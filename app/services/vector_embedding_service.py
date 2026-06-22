"""向量嵌入服务模块 - 基于 LangChain Embeddings 标准接口"""

from typing import List

from langchain_core.embeddings import Embeddings
from openai import OpenAI
from loguru import logger

from app.config import config


class DashScopeEmbeddings(Embeddings):
    """阿里云 DashScope Text Embedding (OpenAI 兼容模式)
    
    实现 LangChain 标准 Embeddings 接口:
    - embed_documents(texts: List[str]) → List[List[float]]: 批量嵌入文档
    - embed_query(text: str) → List[float]: 嵌入单个查询
    """

    # DashScope text-embedding-v4 服务端硬限制:单次请求最多 10 条
    # 这里设为 10 是上限;如果想留 buffer 可以降到 8
    MAX_BATCH_SIZE = 10

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-v4",
        dimensions: int = 1024,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        batch_size: int = MAX_BATCH_SIZE,
    ):
        """
        初始化 DashScope Embeddings

        Args:
            api_key: DashScope API Key
            model: 嵌入模型名称
            dimensions: 向量维度
            batch_size: 单次 API 调用的最大文本数(DashScope 上限 10)
        """
        if not api_key or api_key == "your-api-key-here":
            raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

        if batch_size < 1 or batch_size > self.MAX_BATCH_SIZE:
            raise ValueError(
                f"batch_size 必须在 1 ~ {self.MAX_BATCH_SIZE} 之间(DashScope 服务端限制)"
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size

        # 打印初始化信息
        masked_key = self._mask_api_key(api_key)
        logger.info(
            f"DashScope Embeddings 初始化完成 - "
            f"模型: {model}, 维度: {dimensions}, batch_size: {batch_size}, "
            f"Base URL: {base_url}, API Key: {masked_key}"
        )

    @staticmethod
    def _mask_api_key(api_key: str) -> str:
        """掩码 API Key 用于日志"""
        if len(api_key) > 8:
            return f"{api_key[:8]}...{api_key[-4:]}"
        return "***"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量嵌入文档列表 (LangChain 标准接口)。

        DashScope text-embedding-v4 单次最多 10 条,因此这里**内部按 batch_size 分批调用**,
        最后按原始顺序拼回去。对调用方完全透明 —— 不管你传 5 条还是 500 条,
        都能正确得到等长的向量列表。

        Args:
            texts: 文本列表(可以任意长度)

        Returns:
            List[List[float]]: 嵌入向量列表,顺序与 texts 一致
        """
        if not texts:
            return []

        total = len(texts)
        batch_size = self.batch_size
        num_batches = (total + batch_size - 1) // batch_size

        logger.info(
            f"批量嵌入 {total} 个文档,分 {num_batches} 批(batch_size={batch_size})"
        )

        all_embeddings: List[List[float]] = []
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, total)
            batch = texts[start:end]
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    dimensions=self.dimensions,
                    encoding_format="float",
                )
                # DashScope 返回的 data 不保证顺序,需要按 index 排回原顺序
                batch_embeddings = [None] * len(batch)
                for item in response.data:
                    batch_embeddings[item.index] = item.embedding
                all_embeddings.extend(batch_embeddings)
                logger.debug(
                    f"  批次 {batch_idx + 1}/{num_batches} 完成 ({len(batch)} 条)"
                )
            except Exception as e:
                logger.error(
                    f"批次 {batch_idx + 1}/{num_batches} 嵌入失败 "
                    f"(texts[{start}:{end}]): {e}"
                )
                raise RuntimeError(
                    f"批量嵌入失败 (批次 {batch_idx + 1}/{num_batches}): {e}"
                ) from e

        # 完整性校验:输入输出条数必须严格一致
        if len(all_embeddings) != total:
            raise RuntimeError(
                f"嵌入结果数与输入数不一致: 输入 {total}, 输出 {len(all_embeddings)}"
            )

        logger.info(
            f"批量嵌入完成: 共 {total} 个向量,维度: {len(all_embeddings[0])}"
        )
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        嵌入单个查询文本 (LangChain 标准接口)
        
        Args:
            text: 查询文本
            
        Returns:
            List[float]: 嵌入向量
        """
        if not text or not text.strip():
            raise ValueError("查询文本不能为空")
        
        try:
            logger.debug(f"嵌入查询, 长度: {len(text)} 字符")
            
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self.dimensions,
                encoding_format="float"
            )
            
            embedding = response.data[0].embedding
            logger.debug(f"查询嵌入完成, 维度: {len(embedding)}")
            
            return embedding
            
        except Exception as e:
            logger.error(f"查询嵌入失败: {e}")
            raise RuntimeError(f"查询嵌入失败: {e}") from e


# 全局单例
vector_embedding_service = DashScopeEmbeddings(
    api_key=config.dashscope_api_key,
    model=config.dashscope_embedding_model,
    dimensions=1024,
    base_url=config.dashscope_api_base,
)
