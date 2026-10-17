"""Async Claude structured-output helper (JSON-schema output, validated here rather than in the SDK)."""
import asyncio
import base64
import weakref
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

import anthropic
from langsmith.wrappers import wrap_anthropic
from pydantic import BaseModel, ValidationError

from src.config import LLM_MODEL

T = TypeVar("T", bound=BaseModel)

_TRUNCATION_STOP_REASONS = ("max_tokens", "model_context_window_exceeded")

_clients: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, anthropic.AsyncAnthropic]" = (
    weakref.WeakKeyDictionary()
)


class LLMError(Exception): ...


class LLMRefusalError(LLMError): ...


class LLMTruncatedError(LLMError): ...


class LLMOutputError(LLMError): ...


@dataclass(frozen=True)
class LLMUsage:
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int


@dataclass(frozen=True)
class StructuredResult(Generic[T]):
    output: T
    usage: LLMUsage
    stop_reason: str
    model: str


def get_client() -> anthropic.AsyncAnthropic:
    """Return an AsyncAnthropic client cached per running event loop (must be called inside a loop).
    Wrapped for LangSmith: calls show up as LLM runs inside the graph trace when LANGSMITH_TRACING=true."""
    loop = asyncio.get_running_loop()
    client = _clients.get(loop)
    if client is None:
        client = wrap_anthropic(anthropic.AsyncAnthropic())
        _clients[loop] = client
    return client


def text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def image_block(data: bytes, media_type: str = "image/png") -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": base64.standard_b64encode(data).decode("ascii"),
        },
    }


async def generate_structured(
    output_model: type[T],
    *,
    system: str,
    content: str | list[dict],
    max_tokens: int,
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None,
    model: str | None = None,
) -> StructuredResult[T]:
    output_config: dict = {
        "format": {
            "type": "json_schema",
            "schema": anthropic.transform_schema(output_model),
        }
    }
    if effort is not None:
        output_config["effort"] = effort

    response = await get_client().messages.create(
        model=model or LLM_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": content}],
        output_config=output_config,
    )

    stop_reason = response.stop_reason
    if stop_reason == "refusal":
        raise LLMRefusalError("Model refused the request (stop_reason=refusal)")
    if stop_reason in _TRUNCATION_STOP_REASONS:
        raise LLMTruncatedError(
            f"Output truncated (stop_reason={stop_reason}, max_tokens={max_tokens})"
        )

    text = "".join(block.text for block in response.content if block.type == "text")
    if not text:
        raise LLMOutputError(f"Response contained no text block (stop_reason={stop_reason})")

    try:
        output = output_model.model_validate_json(text)
    except ValidationError as e:
        raise LLMOutputError(
            f"Output failed validation against {output_model.__name__}: {e}"
        ) from e

    usage = response.usage
    return StructuredResult(
        output=output,
        usage=LLMUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_input_tokens=usage.cache_read_input_tokens or 0,
            cache_creation_input_tokens=usage.cache_creation_input_tokens or 0,
        ),
        stop_reason=stop_reason,
        model=response.model,
    )
