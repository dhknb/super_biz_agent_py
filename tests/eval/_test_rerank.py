"""测试 DashScope Rerank REST API 是否可用"""
import json
import httpx
from app.config import config

url = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"

query = "CPU 使用率持续高于 90% 该怎么排查处理?"
passages = [
    "今天学习了苍穹外卖项目的开发流程，从需求分析到代码实现，收获很多。",
    "CPU 使用率持续高于 90% 的处理方法：1) 通过 top 定位高 CPU 进程；2) 检查是否有死循环。",
    "磁盘满了需要清理日志文件和使用 du 定位大文件。",
    "Java 面向对象编程中，封装继承多态是三大特性。",
]

resp = httpx.post(
    url,
    headers={
        "Authorization": f"Bearer {config.dashscope_api_key}",
        "Content-Type": "application/json",
    },
    json={
        "model": "gte-rerank",
        "input": {
            "query": query,
            "documents": passages,
        },
        "parameters": {
            "top_n": 3,
            "return_documents": True,
        },
    },
    timeout=30.0,
)
print(f"HTTP {resp.status_code}")
body = resp.json()
if resp.is_success and "output" in body:
    print("✅ Rerank API 调用成功!")
    for item in body["output"]["results"]:
        idx = item["index"]
        score = item["relevance_score"]
        print(f"  score={score:.4f} idx={idx} text={passages[idx][:60]}")
else:
    print("❌ 失败:")
    print(json.dumps(body, ensure_ascii=False, indent=2))
