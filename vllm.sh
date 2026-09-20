#!/usr/bin/env bash
#
# vllm.sh — run vLLM CLI commands via the official vllm/vllm-openai Docker
# image (nixpkgs' CUDA-enabled vllm is marked broken -- see flake.nix history).
#
# Usage:
#   ./vllm.sh serve <model> [vllm args...]   start the OpenAI-compatible server
#   ./vllm.sh --help                         vLLM's own CLI help
#   ./vllm.sh <anything vllm's CLI accepts>
#
# <model> is a HuggingFace model id or path -- vLLM does not read this repo's
# library/*.yml or weights/*.gguf (those are llama.cpp-specific formats).
#
# Env vars:
#   PORT       host port for the OpenAI server (default: 8000)
#   NAME       container name (default: vllm) -- set a distinct one per
#              instance to run more than one at a time
#   GPU        CDI device index to pin to, e.g. "0" or "1" (default: all
#              GPUs visible) -- run one instance per GPU for data-parallel
#              replicas, since this host's 3070+3060 are too different in
#              size/speed for tensor parallelism to make sense across them
#   HF_TOKEN   forwarded in if set, for gated HuggingFace models
#
# Requires GPU passthrough (nvidia-container-toolkit + CDI) on the host --
# this repo's flake enables that via hardware.nvidia-container-toolkit.enable
# in the machine's NixOS config, not something this script sets up itself.
# Uses `--device nvidia.com/gpu=all` (the CDI device reference), not the
# older `--gpus all` flag -- on this host's Docker/CDI setup, `--gpus all`
# resolves against the wrong vendor and fails with "AMD CDI spec not found"
# even though only NVIDIA CDI is registered.
set -euo pipefail

PORT="${PORT:-8000}"
NAME="${NAME:-vllm}"
CACHE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)/.vllm/hf-cache"
mkdir -p "$CACHE_DIR"

args=(
  --rm
  --name "$NAME"
  # vllm/vllm-openai's own ENTRYPOINT is ["vllm", "serve"] -- override it to
  # plain "vllm" so this script's own args (e.g. "serve <model>") aren't
  # appended after an implicit "serve", which eats the model tag as an
  # extra "serve" positional and errors on the real model as unrecognized.
  --entrypoint vllm
  --device "nvidia.com/gpu=${GPU:-all}"
  -p "${PORT}:8000"
  -v "${CACHE_DIR}:/root/.cache/huggingface"
  # This host's two GPUs (3070 + 3060) are different models; vLLM warns
  # without this and PCI_BUS_ID is the deterministic, driver-agreed order.
  -e "CUDA_DEVICE_ORDER=PCI_BUS_ID"
  # Without this, a card with little headroom past the model weights (e.g.
  # this host's 3070 running an 8B model) livelocks: PyTorch's allocator
  # retries the same handful of block sizes forever instead of failing
  # cleanly, because free memory is fragmented across small unusable chunks
  # rather than one contiguous span. expandable_segments lets it grow one
  # segment instead of hunting for a contiguous block, which fixes this.
  -e "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
)
[[ -n "${HF_TOKEN:-}" ]] && args+=(-e "HF_TOKEN=${HF_TOKEN}")
[[ -t 1 ]] && args+=(-t)

docker rm -f "$NAME" >/dev/null 2>&1 || true
exec docker run "${args[@]}" vllm/vllm-openai:latest "$@"
