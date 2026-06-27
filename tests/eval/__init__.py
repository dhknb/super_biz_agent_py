"""RAG offline evaluation package.

Goal: build a reproducible, CI-friendly evaluation pipeline for the
retrieval + answer-generation stack of super_biz_agent_py.

Pipeline:
    golden_set.jsonl  ->  run_eval.py  ->  metrics.py  ->  reports/*.json

See README.md for usage.
"""
