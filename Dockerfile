# CUDA 12.4 base — matches the training server environment (PyTorch + CUDA 12.4).
# Compatible with host drivers 12.4+ (local RTX 3050 uses 12.7 driver → ok).
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Ubuntu 22.04 ships Python 3.11.0rc1 (a pre-release) whose pip and packaging
# layers conflict with PyTorch at runtime (typing_extensions, torch.version, etc.).
# Creating a venv gives a completely clean Python environment isolated from the
# Ubuntu system packages — the standard fix for this class of apt/pip conflict.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv gcc git ca-certificates \
        libsndfile1 libfluidsynth3 fluidsynth \
    && rm -rf /var/lib/apt/lists/* \
    && python3.11 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip

# All subsequent python / pip calls use the venv, never the system Python.
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Layer 1: heavy frozen deps. Touch `requirements.lock` to invalidate.
COPY requirements.lock /app/requirements.lock
RUN pip install --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.4.1 torchaudio==2.4.1 \
    && pip install -r /app/requirements.lock

# Triton's gcc link step needs `-lcuda` to resolve, but the nvidia/cuda runtime
# image only ships libcuda.so at /usr/local/cuda-12.4/compat/ (forward-compat stub),
# which isn't on the default linker search path. Symlink it into /usr/local/lib
# and register the compat dir with ldconfig so both link-time and runtime resolve.
RUN ln -sf /usr/local/cuda-12.4/compat/libcuda.so /usr/local/lib/libcuda.so \
    && echo "/usr/local/cuda-12.4/compat" > /etc/ld.so.conf.d/cuda-compat.conf \
    && ldconfig

# Layer 2: source. Changes here don't reinstall the heavy stuff.
COPY pyproject.toml /app/pyproject.toml
COPY src /app/src
COPY configs /app/configs
COPY scripts /app/scripts
COPY tests /app/tests
RUN pip install -e .

# Sensible defaults — override at `docker run` time as needed.
ENV PYTHONIOENCODING=utf-8
CMD ["python", "scripts/train.py", "--config", "configs/config.yaml"]
