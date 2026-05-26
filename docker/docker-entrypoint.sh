#!/bin/bash
set -e

# ── WSL2 libcuda.so ──────────────────────────────────────────────────────────
# On WSL2, the real CUDA driver lives at /usr/lib/wsl/lib/libcuda.so.1
# (injected by the Windows NVIDIA driver). The Dockerfile pre-bakes a
# forward-compat stub symlink, but that stub only supports running
# pre-compiled code — Triton's JIT kernel compilation needs the full driver
# API (cuModuleLoad, cuLinkCreate, etc.) which only the real library has.
#
# Always replace the stub with the real WSL2 driver when available.
# On native Linux servers the WSL2 path doesn't exist, so this is a no-op.
WSL_CUDA="/usr/lib/wsl/lib/libcuda.so.1"
STUB="/usr/local/lib/libcuda.so"
if [ -f "$WSL_CUDA" ]; then
    ln -sf "$WSL_CUDA" "$STUB"
    ldconfig 2>/dev/null || true
fi

# ── /usr/local/cuda symlink ──────────────────────────────────────────────────
# Triton detects CUDA_HOME via /usr/local/cuda. The nvidia/cuda runtime image
# ships /usr/local/cuda-12.x but omits the bare symlink that Triton expects.
if [ ! -e /usr/local/cuda ]; then
    CUDA_DIR=$(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -1)
    if [ -n "$CUDA_DIR" ]; then
        ln -sf "$CUDA_DIR" /usr/local/cuda
    fi
fi

# ── WSL2: drop the CUDA forward-compat layer ─────────────────────────────────
# The Dockerfile registers /usr/local/cuda-12.x/compat on the ldconfig path so
# a too-old native-Linux driver can still run CUDA 12.4 code. On WSL2 the real
# driver (12.7) is NEWER than CUDA 12.4, so the compat stub is unnecessary — and
# worse, having BOTH the compat libcuda.so and the real WSL2 libcuda.so.1 loaded
# in one process makes torch.compile/Triton SIGSEGV the instant it compiles a
# kernel. Remove the compat dir from the linker path when on WSL2 so only the
# real driver is ever loaded. No-op on native Linux (file won't exist there
# once this branch is skipped).
if [ -f "$WSL_CUDA" ]; then
    rm -f /etc/ld.so.conf.d/cuda-compat.conf
    ldconfig 2>/dev/null || true
fi

# ── Stack size ───────────────────────────────────────────────────────────────
# Triton's multi-threaded JIT compilation can hit the default 8MB stack limit
# on WSL2. Unlimited stack is the standard fix; harmless on native Linux.
ulimit -s unlimited 2>/dev/null || true

exec "$@"
