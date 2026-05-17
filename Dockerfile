# CUDA 12.1 base — matches vast.ai / runpod default RTX 4090 images.
# Builds in ~3 min on a server, then a `docker run` brings you to a ready
# training prompt without re-installing torch + lightning + miditoolkit.
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# python3-pip on Ubuntu 22.04 installs pip for Python 3.10 (the distro default),
# NOT for 3.11. We bootstrap pip directly via ensurepip so every subsequent
# `python -m pip` call targets Python 3.11 — avoiding a silent mismatch where
# torch/lightning are installed for 3.10 but the `python` symlink runs 3.11.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev git ca-certificates \
        libsndfile1 libfluidsynth3 fluidsynth \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && python -m ensurepip --upgrade \
    && python -m pip install --upgrade pip

WORKDIR /app

# Layer 1: heavy frozen deps. Touch `requirements.lock` to invalidate.
COPY requirements.lock /app/requirements.lock
RUN python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.4.1 torchaudio==2.4.1 \
    && python -m pip install -r /app/requirements.lock

# Layer 2: source. Changes here don't reinstall the heavy stuff.
COPY pyproject.toml /app/pyproject.toml
COPY src /app/src
COPY configs /app/configs
COPY scripts /app/scripts
COPY tests /app/tests
RUN python -m pip install -e .

# Sensible defaults — override at `docker run` time as needed.
ENV PYTHONIOENCODING=utf-8
CMD ["python", "scripts/train.py", "--config", "configs/config.yaml"]
