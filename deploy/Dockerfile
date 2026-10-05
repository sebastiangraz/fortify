# GPU image for the fortify service. The torch cu128 wheels bundle the CUDA runtime;
# the host only needs the NVIDIA driver + container toolkit (docker run --gpus all).
#   docker build -f deploy/Dockerfile -t fortify .
#   docker run --gpus all -p 8765:8765 -e FORTIFY_TOKEN=... -v fortify-weights:/weights fortify
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy \
    FORTIFY_WEIGHTS=/weights HF_HOME=/weights/hf

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra models --extra serve --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --extra models --extra serve

EXPOSE 8765
CMD ["uv", "run", "--no-sync", "fortify", "serve", "--host", "0.0.0.0", "--port", "8765"]
