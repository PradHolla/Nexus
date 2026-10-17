"""Shared transient-error classification and LangGraph retry policy (ingestion and quiz graphs)."""
import anthropic
import httpx
from langgraph.types import RetryPolicy
from qdrant_client.common.client_exceptions import ResourceExhaustedResponse
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse


def is_transient(exc: BaseException) -> bool:
    """True for network/overload failures worth retrying; False for bugs, bad requests and LLM output errors."""
    if isinstance(exc, anthropic.APIConnectionError):  # includes APITimeoutError
        return True
    if isinstance(exc, anthropic.APIStatusError):  # 429, 5xx, overloaded, service unavailable, deadline exceeded
        return exc.status_code == 429 or exc.status_code >= 500
    if isinstance(exc, UnexpectedResponse):
        return exc.status_code is not None and (exc.status_code == 429 or exc.status_code >= 500)
    if isinstance(exc, ResourceExhaustedResponse):
        return True
    if isinstance(exc, ResponseHandlingException):  # qdrant wraps transport errors (and response parse errors)
        return isinstance(exc.source, (httpx.TransportError, OSError))
    return isinstance(exc, httpx.TransportError)


RETRY = RetryPolicy(max_attempts=3, retry_on=is_transient)
