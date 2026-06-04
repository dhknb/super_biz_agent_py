#!/bin/bash
cd /home/dong/projects/super_biz_agent_py || exit 1
.venv/bin/python -m pytest tests/ -v --tb=short 2>&1 | tee /tmp/test_out.txt
