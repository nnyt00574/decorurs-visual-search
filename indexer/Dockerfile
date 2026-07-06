# uv ships as a standalone binary -- copy it from Astral's official image
# rather than pip-installing it, which is the current recommended pattern.
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
# opencv-python-headless (a rembg dependency) still dynamically links libGL
# on Debian despite the "headless" name -- without this, importing it fails
# at runtime with "libGL.so.1: cannot open shared object file".
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# Install deps in their own layer so code edits don't bust the cache
COPY pyproject.toml ./
RUN uv sync --no-install-project
# Copy only the files warm_models.py needs (itself + clip_service.py, which
# only imports third-party packages) so later app-code changes don't bust
# this layer -- it downloads/bakes several hundred MB of model weights.
COPY warm_models.py clip_service.py ./
RUN uv run python warm_models.py
COPY . .
RUN uv sync
# Runs fetch_catalog.py then index_products.py, in that order
CMD ["sh", "-c", "uv run python fetch_catalog.py && uv run python index_products.py"]
