# Trade Intelligence Module - API service image.
#
# The 8 GB raw BACI dataset is deliberately NOT baked into the image. Build the
# parquet cache on the host with `python scripts/build_cache.py`, then mount it:
#
#   docker build -t trade-intelligence .
#   docker run -p 8000:8000 -v "$(pwd)/cache:/app/cache:ro" trade-intelligence
#
# The cache is a few tens of megabytes, so it can also be COPYed in for a
# self-contained image if you prefer.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRADE_CACHE_DIR=/app/cache

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY core/ ./core/
COPY agent/ ./agent/
COPY api/ ./api/
COPY scripts/ ./scripts/
COPY frontend/ ./frontend/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
