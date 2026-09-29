FROM python:3.11.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    HF_HOME=/home/wrs/.cache/huggingface

WORKDIR /app

COPY scripts/requirements-linux.txt /tmp/requirements.txt
RUN python -m pip install -r /tmp/requirements.txt \
    && python -m playwright install --with-deps chromium \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 wrs \
    && mkdir -p /home/wrs/.cache/huggingface \
    && chown -R wrs:wrs /home/wrs /ms-playwright

COPY --chown=wrs:wrs . /app

USER wrs
EXPOSE 8090

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8090/health/live', timeout=4)"

CMD ["python", "-m", "uvicorn", "scripts.server:app", "--host", "0.0.0.0", "--port", "8090"]
