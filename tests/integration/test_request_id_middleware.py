"""request_id 中间件的端到端契约测试。

验证四件事（都是「对外承诺」，坏掉了排障链路就断）：
1. 响应头一定带 X-Request-ID
2. 上游传入的合法 id 会被复用（跨系统串联）
3. 非法 id（含换行 / 超长）被拒绝并换成新生成的 —— 防日志伪造
4. 请求之间的 id 互不相同，且不会互相继承
"""

from app.core.request_context import REQUEST_ID_HEADER


class TestResponseHeader:
    def test_response_carries_request_id(self, client) -> None:
        response = client.get("/health")
        assert response.headers.get(REQUEST_ID_HEADER)

    def test_ids_differ_across_requests(self, client) -> None:
        """不还原 ContextVar 会让后一个请求继承前一个的 id，
        那比没有 id 更糟 —— 它把两个请求的日志错误地关联在一起。"""
        first = client.get("/health").headers[REQUEST_ID_HEADER]
        second = client.get("/health").headers[REQUEST_ID_HEADER]
        assert first != second


class TestIncomingHeader:
    def test_valid_incoming_id_is_reused(self, client) -> None:
        response = client.get("/health", headers={REQUEST_ID_HEADER: "gw-abc-123"})
        assert response.headers[REQUEST_ID_HEADER] == "gw-abc-123"

    def test_newline_injection_is_rejected(self, client) -> None:
        """带换行的 id 若被放行，攻击者可凭空伪造一整行日志。"""
        malicious = "abc\nFAKE INFO forged log line"
        response = client.get("/health", headers={REQUEST_ID_HEADER: malicious})
        returned = response.headers[REQUEST_ID_HEADER]
        assert returned != malicious
        assert "\n" not in returned

    def test_overlong_id_is_rejected(self, client) -> None:
        """超长值会撑爆数据库列、污染日志宽度。"""
        response = client.get("/health", headers={REQUEST_ID_HEADER: "x" * 200})
        assert len(response.headers[REQUEST_ID_HEADER]) <= 64

    def test_blank_id_falls_back_to_generated(self, client) -> None:
        response = client.get("/health", headers={REQUEST_ID_HEADER: "   "})
        assert response.headers[REQUEST_ID_HEADER].strip()
