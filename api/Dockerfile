FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# opencv-python-headless (a rembg dependency) still dynamically links libGL
# on Debian despite the "headless" name -- without this, importing it fails
# at runtime with "libGL.so.1: cannot open shared object file".
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
RUN uv sync --no-install-project

COPY . .
RUN uv sync

# Bakes the CLIP and background-removal model weights into this layer
# instead of re-downloading several hundred MB on every container start.
RUN uv run python warm_models.py

EXPOSE 8000

# `fastapi run` is the current standard entrypoint (bundled via
# fastapi[standard], runs on Uvicorn under the hood, production mode by
# default -- no --reload, add --workers N to scale across cores).
CMD ["uv", "run", "fastapi", "run", "main.py", "--host", "0.0.0.0", "--port", "8000"]
