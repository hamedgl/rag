FROM python:3.14-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.23 /uv /bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
# deps first: cached until pyproject/uv.lock change (torch alone is ~3 GB)
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-install-project
COPY rag ./rag
# model weights come from the mounted HF cache; never call the Hub at runtime
ENV PATH=/app/.venv/bin:$PATH HF_HUB_OFFLINE=1
CMD ["uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
