#!/usr/bin/env bash
set -u
cd /home/dong/projects/super_biz_agent_py

echo "=== compile ==="
.venv/bin/python -m py_compile tests/eval/run_eval.py tests/eval/multi_query.py && echo "COMPILE OK"

echo "=== MODEL_NAME refs ==="
grep -rn "MODEL_NAME" tests/eval/ || echo "(none)"

echo "=== load_cross_encoder refs ==="
grep -rn "load_cross_encoder" tests/eval/ || echo "(none)"

echo "=== resolve path test ==="
.venv/bin/python -c "
from tests.eval.multi_query import resolve_rerank_path, LOCAL_RERANK_DIR
print('LOCAL_RERANK_DIR =', LOCAL_RERANK_DIR)
print('resolved =', resolve_rerank_path())
"
