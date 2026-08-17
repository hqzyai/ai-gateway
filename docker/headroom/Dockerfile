FROM python:3.13-slim@sha256:8fef26df932191825664e4957ff488c96dfe64918327634a357a55facbc994d3

ARG HEADROOM_VERSION=0.33.0

RUN groupadd --gid 1000 headroom \
    && useradd --uid 1000 --gid 1000 --create-home headroom \
    && python -m pip install --no-cache-dir "headroom-ai[proxy]==${HEADROOM_VERSION}" \
    && headroom --version \
    && python -c "from headroom._core import DiffCompressor, SmartCrusher; assert DiffCompressor and SmartCrusher" \
    && mkdir -p /home/headroom/.headroom \
    && chown -R headroom:headroom /home/headroom

COPY headroom_sidecar.py /usr/local/bin/headroom_sidecar.py

USER headroom
WORKDIR /home/headroom

ENV HEADROOM_NO_SUBSCRIPTION_TRACKING=true \
    HEADROOM_TELEMETRY=off \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

VOLUME ["/home/headroom/.headroom"]
EXPOSE 8788

HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=12 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8787/readyz', timeout=3)"]

ENTRYPOINT ["python", "/usr/local/bin/headroom_sidecar.py"]

LABEL org.opencontainers.image.source="https://github.com/headroomlabs-ai/headroom" \
      org.opencontainers.image.version="${HEADROOM_VERSION}" \
      org.opencontainers.image.licenses="Apache-2.0"
