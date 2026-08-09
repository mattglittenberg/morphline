# morphline runtime image (BUILD_PLAN §2.6).
#
# amd64 only for v1.0. Most of the scientific Python stack ships arm64 wheels
# now, but "no compilation surprises" is not a guarantee to plan around, so
# arm64 is a timeboxed follow-on rather than a promise. Local development on
# ARM can run this image under emulation; for tabular data at this scale,
# slower is still seconds.
#
# Reproducibility rests on uv.lock, which is committed. "Reproducible" without
# a lockfile is a claim, not a fact.

# ---- builder ---------------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /build

# Dependencies first, so a source-only change does not invalidate the layer
# that resolves and installs the entire scientific stack.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ---- runtime ---------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="morphline" \
      org.opencontainers.image.description="BIDS-aware longitudinal neuroimaging derivatives pipeline" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/OWNER/morphline"

# procps supplies `ps`, which Nextflow's task wrapper uses to collect trace
# metrics. Without it every task logs a warning and the trace is incomplete.
RUN apt-get update \
    && apt-get install -y --no-install-recommends procps \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 morphline

COPY --from=builder --chown=morphline:morphline /build/.venv /opt/morphline/.venv

ENV PATH="/opt/morphline/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR=/tmp/matplotlib

USER morphline
WORKDIR /work

# Fails the build if the CLI cannot start, rather than deferring the failure to
# the first Nextflow task that tries to use it.
RUN morphline version

ENTRYPOINT []
CMD ["morphline", "--help"]
