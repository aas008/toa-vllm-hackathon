#!/usr/bin/env bash
set -euo pipefail

NODE="rh-h100-02"
SESSION="rohan-qwen3-06b"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

echo "Killing tmux session ${SESSION} on ${NODE}..."
ssh $SSH_OPTS "$NODE" "tmux kill-session -t ${SESSION} 2>/dev/null" && echo "Done." || echo "Session not found."
