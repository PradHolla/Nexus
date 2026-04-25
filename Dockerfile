# Use a stable Python base image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libreoffice \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install 'uv' for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install project dependencies
RUN uv sync --frozen --no-install-project

# Copy the rest of the backend source code
COPY . .

# Ensure the database and temp directories exist
RUN mkdir -p /tmp && chmod 777 /tmp

# Expose the FastAPI port
EXPOSE 8000

# Set environment variables (Placeholders - will be overridden by docker-compose/env)
ENV AWS_REGION=us-east-1
ENV QDRANT_HOST=qdrant
ENV QDRANT_PORT=6333

# Run the FastAPI server
# Note: We use 'uv run' to ensure the virtualenv created by uv is used.
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
