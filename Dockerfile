# Offline runtime image for Fraud Decision Lab (building needs network or cached dependencies).
# Tags and apt repositories are mutable; this is not a bit-for-bit reproducible image.
# A release can separately record reviewed image digests and an image SBOM.
# Build: make docker. Run: docker run --rm --network none fraud-decision-lab doctor
# For an offline demo and copying its report to the host, see README.md.
# Fully qualified image names so the file works with Docker and Podman.
# uv is only mounted during the build steps below; it is not left in the final image.
FROM ghcr.io/astral-sh/uv:0.12.3 AS uv

FROM docker.io/library/python:3.12-slim

# libgomp1: OpenMP runtime that XGBoost needs on Linux.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_NO_CACHE=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependencies first (cached layer), then the project itself.
COPY pyproject.toml uv.lock .python-version README.md LICENSE ./
RUN --mount=type=bind,from=uv,source=/uv,target=/usr/local/bin/uv \
    uv sync --frozen --no-dev --no-install-project
COPY src ./src
# Configs and the bundled synthetic dataset, so `demo` works offline inside the image.
COPY configs ./configs
COPY data/demo ./data/demo
RUN --mount=type=bind,from=uv,source=/uv,target=/usr/local/bin/uv \
    uv sync --frozen --no-dev

# Run as an unprivileged user. Only original synthetic data is baked in; mount external data at run time.
RUN useradd --create-home --uid 1000 lab && mkdir -p /app/data /app/outputs && chown -R lab /app/data /app/outputs
USER lab

ENTRYPOINT ["fraudlab"]
CMD ["doctor"]
