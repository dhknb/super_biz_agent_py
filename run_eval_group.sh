#!/usr/bin/env bash
# 跑单组测评：run_eval_group.sh <retriever> <tag>
set -u
cd /home/dong/projects/super_biz_agent_py

export HTTP_PROXY=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897
export NO_PROXY=localhost,127.0.0.1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

RETRIEVER="$1"
TAG="$2"
LOG="/tmp/eval_${TAG}.log"

.venv/bin/python -m tests.eval.run_eval \
  --retriever "$RETRIEVER" \
  --golden-set tests/eval/golden_set_expanded.jsonl \
  --top-k 5 \
  --tag "$TAG" > "$LOG" 2>&1
CODE=$?

echo "=== exit=$CODE ==="
echo "rerank_fail_count=$(grep -c 'Rerank 失败' "$LOG" || true)"
echo "cudnn_error_count=$(grep -c 'libcudnn' "$LOG" || true)"
echo "=== report ==="
grep -Ev 'DeprecationWarning|Failed to initialize Async|INFO|DEBUG|^[[:space:]]*$' "$LOG" | tail -45
