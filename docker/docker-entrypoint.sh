#!/bin/bash
set -e

# On WSL2, libcuda.so lives at /usr/lib/wsl/lib/ (mounted from the Windows NVIDIA driver).
# Triton's gcc link step needs it at a standard linker search path.
# Create the symlink once at container startup — harmless on native Linux where
# libcuda.so is already in /usr/local/lib or /usr/lib/x86_64-linux-gnu.
WSL_CUDA="/usr/lib/wsl/lib/libcuda.so.1"
STUB="/usr/local/lib/libcuda.so"
if [ -f "$WSL_CUDA" ] && [ ! -f "$STUB" ]; then
    ln -sf "$WSL_CUDA" "$STUB"
    ldconfig 2>/dev/null || true
fi

exec "$@"
