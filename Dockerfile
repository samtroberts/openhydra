FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        curl \
        git \
        python3-libtorrent \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml /app/

RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt \
    && pip install torch transformers bitsandbytes libtorrent

COPY . /app

EXPOSE 50051

ENTRYPOINT ["python", "-m", "peer.server"]
