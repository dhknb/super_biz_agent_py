"""DashScope Embeddings 分批逻辑单元测试

mock 掉 OpenAI 客户端,只验证 embed_documents 的分批 + 顺序拼接行为。
不需要真实 DashScope API,纯逻辑测试。
"""

from typing import List
from unittest.mock import MagicMock

import pytest

from app.services.vector_embedding_service import DashScopeEmbeddings


def _make_service(batch_size: int = 10) -> DashScopeEmbeddings:
    """构造一个 mock 掉 OpenAI 客户端的 service 实例,避免触发真实 API。"""
    service = DashScopeEmbeddings.__new__(DashScopeEmbeddings)
    service.client = MagicMock()
    service.model = "text-embedding-v4"
    service.dimensions = 4
    service.batch_size = batch_size
    return service


def _fake_response(texts: List[str], dim: int = 4):
    """模拟 DashScope 返回:每条 text → 一个简单的向量。"""
    response = MagicMock()
    response.data = []
    for i, _ in enumerate(texts):
        item = MagicMock()
        item.index = i
        # 用一个跟下标关联的特征向量,便于断言顺序
        item.embedding = [float(i)] * dim
        response.data.append(item)
    return response


def test_embed_documents_empty_returns_empty() -> None:
    service = _make_service()
    assert service.embed_documents([]) == []
    service.client.embeddings.create.assert_not_called()


def test_embed_documents_single_batch_no_split() -> None:
    """文档数 <= batch_size 时,应该只调一次 API。"""
    service = _make_service(batch_size=10)
    texts = [f"text-{i}" for i in range(5)]
    service.client.embeddings.create.side_effect = lambda **kw: _fake_response(kw["input"])

    embeddings = service.embed_documents(texts)

    assert service.client.embeddings.create.call_count == 1
    assert len(embeddings) == 5
    # 顺序应与输入一致
    for i, vec in enumerate(embeddings):
        assert vec[0] == float(i)


def test_embed_documents_splits_into_batches() -> None:
    """文档数 > batch_size 时,必须分批调用。"""
    service = _make_service(batch_size=10)
    texts = [f"text-{i}" for i in range(25)]
    service.client.embeddings.create.side_effect = lambda **kw: _fake_response(kw["input"])

    embeddings = service.embed_documents(texts)

    # 25 条 / 10 一批 = 3 批 (10 + 10 + 5)
    assert service.client.embeddings.create.call_count == 3
    assert len(embeddings) == 25

    # 检查每批传入的 input 长度
    calls = service.client.embeddings.create.call_args_list
    assert len(calls[0].kwargs["input"]) == 10
    assert len(calls[1].kwargs["input"]) == 10
    assert len(calls[2].kwargs["input"]) == 5


def test_embed_documents_preserves_order_across_batches() -> None:
    """跨批次的拼接顺序必须严格保持输入顺序。"""
    service = _make_service(batch_size=3)
    texts = [f"text-{i}" for i in range(8)]
    service.client.embeddings.create.side_effect = lambda **kw: _fake_response(kw["input"])

    embeddings = service.embed_documents(texts)

    assert len(embeddings) == 8
    # 每条向量第一位 == 它在该批次内的下标
    # 拼接后应该是: [0,1,2, 0,1,2, 0,1] (每批从 0 重新计数,因为 fake 用的是 batch 内 index)
    expected = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0]
    actual = [vec[0] for vec in embeddings]
    assert actual == expected


def test_embed_documents_handles_out_of_order_response() -> None:
    """DashScope 返回顺序不保证,代码必须按 item.index 排回去。"""
    service = _make_service(batch_size=10)
    texts = ["a", "b", "c"]

    def shuffled_response(**kw):
        response = MagicMock()
        # 故意打乱:返回 [index=2, index=0, index=1]
        items = []
        for idx in [2, 0, 1]:
            item = MagicMock()
            item.index = idx
            item.embedding = [float(idx)] * 4
            items.append(item)
        response.data = items
        return response

    service.client.embeddings.create.side_effect = shuffled_response

    embeddings = service.embed_documents(texts)

    # 应该按 0,1,2 的顺序排好
    assert embeddings[0][0] == 0.0
    assert embeddings[1][0] == 1.0
    assert embeddings[2][0] == 2.0


def test_embed_documents_batch_failure_raises() -> None:
    """某一批失败要 raise RuntimeError,不能静默吞掉。"""
    service = _make_service(batch_size=10)
    texts = [f"text-{i}" for i in range(15)]

    call_count = {"n": 0}

    def maybe_fail(**kw):
        call_count["n"] += 1
        if call_count["n"] == 2:  # 第二批故意挂
            raise RuntimeError("API down")
        return _fake_response(kw["input"])

    service.client.embeddings.create.side_effect = maybe_fail

    with pytest.raises(RuntimeError, match="批次 2/2"):
        service.embed_documents(texts)


def test_init_rejects_invalid_batch_size() -> None:
    """超出 DashScope 上限的 batch_size 必须拒绝。"""
    with pytest.raises(ValueError, match="batch_size"):
        DashScopeEmbeddings(api_key="fake-key", batch_size=11)
    with pytest.raises(ValueError, match="batch_size"):
        DashScopeEmbeddings(api_key="fake-key", batch_size=0)
