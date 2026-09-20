#!/usr/bin/env bash
#
# vllm-dual-gpu.sh — starts the known-working two-replica vLLM config found
# by hand: Meta-Llama-3.1-8B-Instruct-AWQ-INT4 on both GPUs, one instance
# per card, for maximum parallel request capacity (see vllm.sh's own
# comments for the GPU-pinning/expandable_segments background).
#
# Each flag below was arrived at empirically on this host's 3070+3060 --
# they are NOT vLLM defaults and won't transfer to a different model or
# GPU pair without re-deriving them the same way:
#
#   GPU1 (3060, 12GB) -- --max-model-len 28672
#     32768 (the model's max) doesn't fit once CUDA graph capture (~0.63 GiB)
#     is accounted for; 28672 is the largest that does, at the default
#     --gpu-memory-utilization (0.92). ~63 tok/s, 1.05x concurrency.
#
#   GPU0 (3070, 8GB) -- --max-model-len 2048 --gpu-memory-utilization 0.97
#     Weights alone (5.33 GiB) are already ~70% of this card's 8GB, so far
#     less room survives for KV cache + graph capture than on the 3060.
#     0.97 utilization and 2048 context is the largest combination that
#     fit -- pushing either further re-triggers the OOM vllm.sh's
#     expandable_segments fix resolves; this is a *real* capacity limit,
#     not the allocator bug. ~78 tok/s, 1.50x concurrency.
#
# Usage: ./vllm-dual-gpu.sh
# Stops both on Ctrl+C. Logs: .vllm/gpu0.log, .vllm/gpu1.log
set -euo pipefail

DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MODEL="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
mkdir -p "$DIR/.vllm"

HOST_IP="${HOST_IP:-$(tailscale ip -4 2>/dev/null)}"
if [[ -z "$HOST_IP" ]]; then
  echo "error: could not determine Tailscale IP (is 'tailscale up' active?)." >&2
  exit 1
fi
export HOST_IP

cleanup() {
  echo "stopping vllm-gpu0 vllm-gpu1 ..."
  docker rm -f vllm-gpu0 vllm-gpu1 >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

GPU=1 NAME=vllm-gpu1 PORT=8001 "$DIR/vllm.sh" serve "$MODEL" \
  --served-model-name llama3.1-8b-awq \
  --max-model-len 28672 \
  > "$DIR/.vllm/gpu1.log" 2>&1 &
PID1=$!

GPU=0 NAME=vllm-gpu0 PORT=8000 "$DIR/vllm.sh" serve "$MODEL" \
  --served-model-name llama3.1-8b-awq \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.97 \
  > "$DIR/.vllm/gpu0.log" 2>&1 &
PID0=$!

echo "starting -- gpu0 (3070): http://${HOST_IP}:8000  gpu1 (3060): http://${HOST_IP}:8001  (tailnet only)"
echo "Ctrl+C to stop both"

# Stream both logs to stdout. A third background job, deliberately not
# among the PIDs `wait` below blocks on -- otherwise Ctrl+C would have to
# kill `tail` too before the script could exit.
tail -n +1 -f "$DIR/.vllm/gpu0.log" "$DIR/.vllm/gpu1.log" &
TAILPID=$!

wait "$PID0" "$PID1"
kill "$TAILPID" 2>/dev/null || true
