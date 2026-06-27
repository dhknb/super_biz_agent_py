"""Check rerank top docs"""
import json, sys
path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    data = json.load(f)
print(f"Report: {data['tag']}")
for c in data["cases"]:
    print(f"  {c['case_id']:8s} top-5: {c['retrieved_doc_ids'][:5]}")
