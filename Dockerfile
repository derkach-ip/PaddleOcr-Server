FROM paddlepaddle/paddle:3.2.2-gpu-cuda12.9-cudnn9.9

WORKDIR /workspace

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_SYSTEM_PYTHON=1

# Copy only dependency file first (changes less frequently)
COPY pyproject.toml /workspace/pyproject.toml

# Install dependencies (cached unless pyproject.toml changes)
RUN uv pip install -r pyproject.toml --no-cache && \
    uv pip install paddlepaddle-gpu --index-url https://www.paddlepaddle.org.cn/packages/nightly/cu129/ --no-cache

# Copy application code last (changes frequently, but doesn't invalidate dependency cache)
COPY server.py /workspace/server.py

EXPOSE 8080

CMD ["python", "/workspace/server.py"]
