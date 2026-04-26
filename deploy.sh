#!/usr/bin/env bash
set -euo pipefail

NODE="rh-h100-02"
SESSION="rohan-qwen3-06b"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

VLLM_CMD="CUDA_VISIBLE_DEVICES=4,5,6,7 /home/lab/rawhad/venvs/vllm_venv/bin/vllm serve Qwen/Qwen3-0.6B \
  --served-model-name qwen3-0.6b \
  --host 0.0.0.0 \
  --port 8100"

# ===
# DEPLOYMENT CONFIGS THAT YOU (#claude-code) SHOULD OPTIMIZE
# ===
SERVER_DEPLOYMENT_CONFIG="--tensor-parallel-size 4 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.95 \
  --enforce-eager"
# ===

echo "Deploying qwen3-0.6b on ${NODE} GPU 4,5,6,7..."
ssh $SSH_OPTS "$NODE" "tmux new-session -d -s ${SESSION} '${VLLM_CMD} ${SERVER_DEPLOYMENT_CONFIG} 2>&1 | tee /tmp/vllm-qwen3-06b.log'"
echo "Started tmux session: ${SESSION}"
echo "Waiting for server to be ready..."

for i in $(seq 1 60); do
  sleep 5
  if ssh $SSH_OPTS "$NODE" "curl -s http://localhost:8100/v1/models" &>/dev/null; then
    echo "Server is ready at ${NODE}:8100"
    exit 0
  fi
  echo "  ...still loading ($((i*5))s)"
done

echo "Timed out waiting for server. Cleaning up..."
ssh $SSH_OPTS "$NODE" "tmux kill-session -t ${SESSION}" || true
ssh $SSH_OPTS "$NODE" "pkill -f 'vllm serve Qwen/Qwen3-0.6B'" || true
ssh $SSH_OPTS "$NODE" "rm -f /tmp/vllm-qwen3-06b.log" || true
echo "Deployment crashed: server did not become ready within 300 seconds."
exit 1
