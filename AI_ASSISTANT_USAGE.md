For the development of Nexus, I used a combination of Gemini, the Gemini CLI, and GitHub Copilot acting primarily to assist me with the tasks. Even though core architectural decisions were mine, AI helped me accelerate the development and manage my time. 

The core system architecture and engineering decisions were mine. I designed:

- The hybrid concurrency model: Separating the CPU-bound document parsing (ProcessPoolExecutor) from the I/O-bound AWS Bedrock embeddings (ThreadPoolExecutor) to prevent bottlenecks.

- The Agentic Orchestration: The 3-stage loop (Planner, Generator, Critic) and the specific rubric the Critic uses to reject administrative trivia or hallucinations.(I am still playing around with the system prompts)

- Deployment Architecture: Using EC2 m6a.xlarge for memory safety, mounting SQLite for state persistence, utilizing IAM Roles instead of hardcoded keys, and using Caddy as a reverse proxy for the UI.

I used AI to accelerate my implementation without worrying about syntax, as it is pretty evident that an LLM generated code in my repo. My AI uses:

- Write boilerplate Pydantic schemas and ensure they strictly matched the JSON outputs I wanted the LLMs to generate.(Annoying piece of work, forcing LLMs to adhere to a format)

- Update my syntax for newer library methods (specifically for boto3, FastAPI SSE streaming, and the latest PyMuPDF4LLM chunking methods).

- Building React/Tailwind UI components, I am not a frontend dev, so AI helped me spin up a beautiful frontend.(My favorite part of AI code generation, I no longer have to generate garbage frontends with Streamlit, Gradio etc. lol)

- Since LLMs were asked to generate outputs in JSONs, it meant fixing a lot of trailing comma JSON parsing errors and resolving Pydantic array validation mismatches between the Generator and the Critic. AI really made this tedious task easier.