# 计划 5：安全与多租户实施文档

## 1. 目标

企业 RAG 最大的风险不是“答错”，而是“把不该给这个用户看的内容答出来”。所以安全与多租户是企业落地必须补齐的能力。

本计划目标：

- API 有认证，不再是任何人都能调用。
- 用户属于租户。
- 文档属于租户和知识库。
- 检索时强制按 `tenant_id` 和权限过滤。
- 上传、删除、重建索引都有审计日志。
- 为后续 RBAC/ABAC 做好扩展。

你完成后，系统应该做到：

- 用户 A 只能检索自己租户的文档。
- 不同租户即使问题相同，也不会互相召回文档。
- 没有认证的请求会被拒绝。
- 每个 document/chunk/vector metadata 都带 `tenant_id`。

## 2. 当前项目现状

当前 API 没有认证：

- `app/api/chat.py`：任何请求都能问。
- `app/api/file.py`：任何请求都能上传文件。
- `app/services/vector_search_service.py`：检索没有 tenant filter。
- `app/services/vector_store_manager.py`：写入 Milvus 时没有租户隔离字段。

当前适合本地演示，但不适合企业环境。

## 3. 推荐安全模型

先做简单但可扩展的模型：

```text
Tenant 租户
  -> User 用户
  -> KnowledgeBase 知识库
  -> Document 文档
  -> Chunk 分片
```

权限规则第一版：

- 用户必须登录。
- 用户只能访问自己 `tenant_id` 下的知识库。
- 文档上传时自动绑定当前用户的 `tenant_id`。
- 检索时必须加 `tenant_id` filter。
- 管理员可以管理本租户所有文档。
- 普通用户只能查询本租户允许的知识库。

先不要一开始就做复杂 ABAC。企业系统可以先从租户隔离 + 简单角色做起。

## 4. 新增依赖

`pyproject.toml` 增加：

```toml
"python-jose[cryptography]>=3.3.0",
"passlib[bcrypt]>=1.7.4",
```

如果你只做内部 API Key 模式，初期可以不加 `passlib`。

## 5. 新增配置

修改 `app/config.py`：

```python
jwt_secret_key: str = "please-change-me"
jwt_algorithm: str = "HS256"
jwt_expire_minutes: int = 60 * 24
auth_enabled: bool = True
```

`.env.example`：

```bash
# Auth
AUTH_ENABLED=true
JWT_SECRET_KEY=change-this-in-production
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=1440
```

生产环境必须把 `JWT_SECRET_KEY` 放到密钥管理系统或安全环境变量中。

## 6. 新增安全上下文模型

新增文件：`app/models/security.py`

```python
from enum import StrEnum

from pydantic import BaseModel, Field


class UserRole(StrEnum):
    ADMIN = "admin"
    USER = "user"


class CurrentUser(BaseModel):
    user_id: str = Field(..., description="用户 ID")
    tenant_id: str = Field(..., description="租户 ID")
    role: UserRole = Field(UserRole.USER, description="用户角色")
    knowledge_base_ids: list[str] = Field(default_factory=list)
```

这个 `CurrentUser` 是整个安全体系最关键的上下文对象。后面所有 API 和 Service 都应该从它拿 `tenant_id`，不要让前端自己传 `tenant_id`。

## 7. 新增认证依赖

新增文件：`app/core/security.py`

```python
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from app.config import config
from app.models.security import CurrentUser, UserRole


bearer_scheme = HTTPBearer(auto_error=False)


def decode_access_token(token: str) -> dict:
    try:
        return jwt.decode(
            token,
            config.jwt_secret_key,
            algorithms=[config.jwt_algorithm],
        )
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效或过期的访问令牌",
        ) from exc


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)]
) -> CurrentUser:
    if not config.auth_enabled:
        return CurrentUser(
            user_id="dev-user",
            tenant_id="dev-tenant",
            role=UserRole.ADMIN,
            knowledge_base_ids=["default"],
        )

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少 Authorization Bearer Token",
        )

    payload = decode_access_token(credentials.credentials)

    user_id = payload.get("sub")
    tenant_id = payload.get("tenant_id")
    role = payload.get("role", UserRole.USER)
    knowledge_base_ids = payload.get("knowledge_base_ids", [])

    if not user_id or not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="令牌缺少用户或租户信息",
        )

    return CurrentUser(
        user_id=user_id,
        tenant_id=tenant_id,
        role=UserRole(role),
        knowledge_base_ids=knowledge_base_ids,
    )
```

学习阶段可以先手工生成 JWT，不急着做登录接口。

## 8. 生成测试 Token

新增脚本：`scripts/create_dev_token.py`

```python
from datetime import datetime, timedelta, timezone

from jose import jwt

from app.config import config


payload = {
    "sub": "user-001",
    "tenant_id": "tenant-a",
    "role": "admin",
    "knowledge_base_ids": ["default"],
    "exp": datetime.now(timezone.utc) + timedelta(minutes=config.jwt_expire_minutes),
}

token = jwt.encode(payload, config.jwt_secret_key, algorithm=config.jwt_algorithm)
print(token)
```

使用：

```bash
python scripts/create_dev_token.py
```

请求时：

```bash
curl -X POST "http://localhost:9900/api/chat_rag" \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"Id":"s1","Question":"CPU 使用率过高怎么处理？"}'
```

## 9. 改造上传接口：绑定租户

在 `app/api/file.py` 中引入：

```python
from app.core.security import get_current_user
from app.models.security import CurrentUser
```

接口签名改成：

```python
@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    current_user: CurrentUser = Depends(get_current_user),
):
    ...
```

如果你已经按计划 1 接了数据库，那么创建文档时要写入：

```python
document = repo.create_document(
    tenant_id=current_user.tenant_id,
    created_by=current_user.user_id,
    knowledge_base_id="default",
    filename=safe_filename,
    original_filename=file.filename,
    file_path=str(file_path.resolve()),
    file_ext=file_extension,
    file_size=len(content),
    content_hash=content_hash,
)
```

对应 `KnowledgeDocument` 增加字段：

```python
tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
knowledge_base_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
created_by: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
```

## 10. 改造 Chunk 和 Milvus metadata

`KnowledgeChunk` 增加：

```python
tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
knowledge_base_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
```

写入 Milvus 时 metadata 必须带：

```python
metadata={
    "tenant_id": document.tenant_id,
    "knowledge_base_id": document.knowledge_base_id,
    "document_id": document.id,
    "chunk_id": chunk.id,
    "chunk_index": chunk.chunk_index,
    "_source": document.file_path,
    "_file_name": document.filename,
}
```

这是隔离的关键。只在关系数据库里有 `tenant_id` 不够，因为向量检索发生在 Milvus，Milvus 查询也必须过滤。

## 11. 改造检索服务：强制租户过滤

修改 `app/services/vector_search_service.py`，新增参数：

```python
def retrieve_documents(
    self,
    query: str,
    top_k: int = 3,
    tenant_id: str | None = None,
    knowledge_base_ids: list[str] | None = None,
) -> list[Document]:
    ...
```

构造 Milvus filter：

```python
def _build_milvus_filter(
    self,
    tenant_id: str | None,
    knowledge_base_ids: list[str] | None,
) -> str | None:
    if not tenant_id:
        return None

    expr = f'metadata["tenant_id"] == "{tenant_id}"'

    if knowledge_base_ids:
        kb_values = ", ".join(f'"{kb_id}"' for kb_id in knowledge_base_ids)
        expr += f' and metadata["knowledge_base_id"] in [{kb_values}]'

    return expr
```

向量检索器加 filter：

```python
search_kwargs = {"k": candidate_k}
expr = self._build_milvus_filter(tenant_id, knowledge_base_ids)
if expr:
    search_kwargs["expr"] = expr

vector_retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
```

BM25 语料也要过滤，否则 BM25 会把其他租户文档混进来。把 `_load_documents_from_milvus()` 改成：

```python
def _load_documents_from_milvus(self, tenant_id: str | None = None) -> list[Document]:
    collection = milvus_manager.get_collection()
    expr = 'id != ""'
    if tenant_id:
        expr = f'id != "" and metadata["tenant_id"] == "{tenant_id}"'

    iterator = collection.query_iterator(
        batch_size=self.BM25_BATCH_SIZE,
        expr=expr,
        output_fields=["id", "content", "metadata"],
    )
```

注意：当前 BM25 retriever 是全局缓存，只按 `doc_count` 缓存。多租户后必须按租户缓存。

```python
self._bm25_retrievers: dict[str, BM25Retriever] = {}
self._bm25_doc_counts: dict[str, int] = {}
```

租户 cache key：

```python
cache_key = tenant_id or "__global__"
```

## 12. 改造 RAG API：从用户上下文拿租户

在 `app/api/chat.py`：

```python
from fastapi import Depends

from app.core.security import get_current_user
from app.models.security import CurrentUser
```

接口：

```python
@router.post("/chat_rag")
async def chat_rag(
    request: ChatRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    result = await rag_answer_service.answer(
        request.question,
        top_k=config.rag_top_k,
        tenant_id=current_user.tenant_id,
        knowledge_base_ids=current_user.knowledge_base_ids,
    )
    return {"code": 200, "message": "success", "data": result}
```

`RagAnswerService.answer()` 增加参数：

```python
async def answer(
    self,
    question: str,
    top_k: int = 5,
    tenant_id: str | None = None,
    knowledge_base_ids: list[str] | None = None,
) -> dict:
    retrieval = vector_search_service.retrieve_with_sources(
        question,
        top_k=top_k,
        tenant_id=tenant_id,
        knowledge_base_ids=knowledge_base_ids,
    )
```

这样前端没有机会伪造 `tenant_id`，因为它只来自 JWT。

## 13. 文档列表也要租户过滤

Repository 不应该暴露全局列表。

```python
def list_documents(self, tenant_id: str) -> list[KnowledgeDocument]:
    stmt = (
        select(KnowledgeDocument)
        .where(KnowledgeDocument.tenant_id == tenant_id)
        .where(KnowledgeDocument.status != DocumentStatus.DELETED)
    )
    return list(self.db.scalars(stmt).all())
```

API：

```python
@router.get("/documents")
async def list_documents(
    db: Session = Depends(get_db),
    current_user: CurrentUser = Depends(get_current_user),
):
    repo = KnowledgeRepository(db)
    docs = repo.list_documents(current_user.tenant_id)
    ...
```

## 14. 新增权限检查函数

新增文件：`app/core/permissions.py`

```python
from fastapi import HTTPException, status

from app.models.security import CurrentUser, UserRole


def require_admin(user: CurrentUser) -> None:
    if user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="需要管理员权限",
        )


def ensure_knowledge_base_access(user: CurrentUser, knowledge_base_id: str) -> None:
    if user.role == UserRole.ADMIN:
        return
    if knowledge_base_id not in user.knowledge_base_ids:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="无权访问该知识库",
        )
```

删除文档、重建索引建议要求管理员：

```python
require_admin(current_user)
```

## 15. 新增审计日志

最低成本做法：先用数据库表记录审计。

ORM：

```python
class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid4()))
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(128), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(64), nullable=False)
    resource_id: Mapped[str] = mapped_column(String(64), nullable=False)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
```

服务：

```python
class AuditLogService:
    def record(
        self,
        db: Session,
        *,
        user: CurrentUser,
        action: str,
        resource_type: str,
        resource_id: str,
        detail: str | None = None,
    ) -> None:
        db.add(
            AuditLog(
                tenant_id=user.tenant_id,
                user_id=user.user_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                detail=detail,
            )
        )
        db.commit()
```

上传成功后：

```python
audit_log_service.record(
    db,
    user=current_user,
    action="document.upload",
    resource_type="document",
    resource_id=document.id,
    detail=document.filename,
)
```

## 16. API Key 模式可选方案

如果你暂时不想做 JWT，也可以先做内部 API Key：

`.env`：

```bash
INTERNAL_API_KEYS=tenant-a:key-a,tenant-b:key-b
```

请求：

```bash
X-API-Key: key-a
```

但长期更推荐 JWT，因为它能携带：

- `sub`
- `tenant_id`
- `role`
- `knowledge_base_ids`
- `exp`

## 17. 安全测试

新增测试：未认证不能访问。

```python
def test_chat_rag_requires_auth(client, monkeypatch):
    monkeypatch.setattr("app.config.config.auth_enabled", True)

    response = client.post(
        "/api/chat_rag",
        json={"Id": "s1", "Question": "hello"},
    )

    assert response.status_code == 401
```

测试租户过滤：

```python
def test_retrieval_uses_tenant_filter(mocker):
    from app.services.vector_search_service import vector_search_service

    mocked_store = mocker.patch(
        "app.services.vector_search_service.vector_store_manager.get_vector_store"
    )

    vector_search_service.retrieve_documents(
        "cpu",
        top_k=3,
        tenant_id="tenant-a",
        knowledge_base_ids=["default"],
    )

    retriever_kwargs = mocked_store.return_value.as_retriever.call_args.kwargs
    expr = retriever_kwargs["search_kwargs"]["expr"]
    assert 'metadata["tenant_id"] == "tenant-a"' in expr
```

测试不同租户无法看到文档：

```python
def test_list_documents_filters_by_tenant(db_session):
    repo = KnowledgeRepository(db_session)
    # 创建 tenant-a 和 tenant-b 的文档
    docs = repo.list_documents("tenant-a")
    assert all(doc.tenant_id == "tenant-a" for doc in docs)
```

## 18. 常见坑

第一，不能信任前端传来的 `tenant_id`。  
所有租户信息必须来自后端解析出来的 `CurrentUser`。

第二，只过滤数据库不够。  
RAG 的泄漏往往发生在向量检索阶段，所以 Milvus 查询必须带 metadata filter。

第三，BM25 缓存不能跨租户复用。  
当前项目 BM25 是全局缓存，多租户后要按租户拆开。

第四，删除文档也要租户校验。  
`DELETE /documents/{id}` 必须先查文档是否属于当前租户。

第五，日志不能打印完整 Token、API Key、用户隐私内容。  
日志里只记录脱敏信息。

## 19. 验收标准

完成后检查：

- 未带 Token 访问受保护 API 返回 401。
- Token 中的 `tenant_id` 能进入 `CurrentUser`。
- 上传文档时写入 `tenant_id`。
- Milvus metadata 中包含 `tenant_id` 和 `knowledge_base_id`。
- 检索时 Milvus filter 包含当前用户的 `tenant_id`。
- BM25 只加载当前租户语料。
- 文档列表只返回当前租户文档。
- 管理操作有审计日志。

## 20. 学习重点

这一阶段你要掌握：

- JWT 的基本结构和校验流程。
- FastAPI 认证依赖如何把用户上下文注入 API。
- RBAC 和租户隔离的区别。
- 为什么 RAG 权限必须下沉到检索层。
- Milvus metadata filter 的使用方式。
- 审计日志在企业系统里的作用。

