# syntax=docker/dockerfile:1
#
# Two stages so the runtime image carries no compiler or build cache.
#   docker build -t kubeai-platform:0.4.0 .
#   docker run --rm -p 8080:8080 kubeai-platform:0.4.0
#
# For a GPU host add `--gpus all` and, on Linux, the NVIDIA container toolkit.

FROM python:3.12-slim AS builder
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install ".[web]"
# Fail the build early if the package does not import.
RUN python -c "import kubeai, kubeai.serving, kubeai.scheduler, kubeai.workflow; print(kubeai.__version__)"

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    KUBEAI_MODEL=kubeai-tiny
WORKDIR /app

# Non-root by default: the gateway never needs to write outside its workdir.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin kubeai \
    && mkdir -p /app/.kubeai && chown -R 10001:10001 /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /build /build

USER 10001
EXPOSE 8080

# Readiness is expressed through /readyz (a model is actually loaded) rather than
# merely "the process is up", which is the usual cause of 502s during rollout.
HEALTHCHECK --interval=15s --timeout=4s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/readyz', timeout=3).status == 200 else 1)"

ENTRYPOINT ["kubeai"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080"]
