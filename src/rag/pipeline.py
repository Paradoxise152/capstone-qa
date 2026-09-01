"""
RAG 检索管线：Dense + BM25 → RRF 融合 → Cohere Reranker 精排
迁移自 projects/langchain-rag/src/retriever.py + reranker.py
整个管线的数据流：
  文档列表 → [BM25稀疏检索 + Dense向量检索] → RRF融合 → Cohere精排 → Top-5结果
"""
# os：Python 标准库，用来读环境变量（如 COHERE_API_KEY）
import os

# BM25Retriever：经典的"关键词精确匹配"检索器。来自 langchain_community（社区贡献的集成包）。
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings

# Chroma：轻量级向量数据库。
# 工作原理：把文档的向量存进去→查询时把用户问题也向量化→找最相似的 Top-K 文档。
from langchain_chroma import Chroma

# EnsembleRetriever：把多个检索器的结果"融合"起来（粗排）。
# 工作原理：并行调用多个检索器→用 RRF 算法把各自的排名合并成统一排名。
from langchain_classic.retrievers import EnsembleRetriever

# ContextualCompressionRetriever："压缩检索器"——Reranker 的包装器。
# 工作原理：先用 base_retriever 粗排召回一批文档，再用 base_compressor（Reranker）精排保留 top_n 个。
from langchain_classic.retrievers import ContextualCompressionRetriever

# CohereRerank：Cohere 公司提供的重排序 API（精排）。
# 工作原理：把 query 和每个候选文档拼接成一对→交给 Cross-Encoder 模型打分→按分数重排。
from langchain_cohere import CohereRerank

# Document：LangChain 的文档对象，包含 page_content（文本）和 metadata（元数据）。
from langchain_core.documents import Document

# 从项目配置文件 ..config导入常量
from ..config import (
    OPENAI_API_KEY,   
    EMBEDDING_MODEL,  
    DENSE_K,          # 向量检索返回的文档数（粗排阶段），默认 10
    BM25_K,           # BM25 检索返回的文档数（粗排阶段），默认 10
    RERANK_TOP_N,     # 精排后最终保留的文档数，默认 5
)


def _load_embeddings():
    """
    加载 Embedding 模型（带下划线前缀表示"内部使用"）。
    OpenAIEmbeddings 会创建一个客户端，调用 OpenAI 的 Embedding API。
    每次调用 .embed_query("文本") 或 .embed_documents(["文档1","文档2"])，
    都会把文本发给 OpenAI，返回对应的嵌入后向量。
    """
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)


def build_retrieval_pipeline(documents: list[Document]):
    """
    构建完整 RAG 检索管线，输入文档列表，输出一个可调用的检索器。
    参数:
        documents: Document 对象列表，每个 Document 包含 page_content（文本）和 metadata
    返回:
        一个 retriever 对象，调用 .invoke("查询") 返回 Top-K 相关文档
    管线流程：
        Step 1: BM25 关键词检索（粗排）
        Step 2: Dense 向量检索（粗排）
        Step 3: RRF 融合两路结果（粗排合并）
        Step 4: Cohere Reranker 精排（可选，需要 COHERE_API_KEY）
    """
    # ── Step 1: BM25 关键词检索 ──
    # BM25Retriever.from_documents() 内部做的事：
    #     对每篇文档分词，建倒排（词->文档）索引，排名。
    # 结果：擅长精确匹配 "LangGraph Checkpointer" 这种专有名词
    bm25 = BM25Retriever.from_documents(documents)
    bm25.k = BM25_K  # 粗排返回 10 个候选

    # ── Step 2: Dense 向量检索 ──
    # _load_embeddings() 返回 OpenAIEmbeddings 客户端
    embeddings = _load_embeddings()

    # Chroma.from_documents() 内部做的事：
    #   1. 对每篇文档调用 embeddings.embed_documents() → 得到 1536 维向量
    #   2. 把向量存到本地 SQLite 数据库（Chroma 的存储后端）
    #   3. 建向量索引（默认用 HNSW 算法，快速近似最近邻搜索）
    vectorstore = Chroma.from_documents(documents, embedding=embeddings)

    # .as_retriever() 把向量数据库包装成"检索器"对象
    # search_kwargs={"k": DENSE_K} 表示每次检索返回 10 个最相似的文档
    # 这个检索器可以被 EnsembleRetriever 统一管理
    dense = vectorstore.as_retriever(search_kwargs={"k": DENSE_K})

    # ── Step 3: RRF 融合 ──
    # EnsembleRetriever 内部做的事：
    #   1. 并行调用 bm25.invoke(query) 和 dense.invoke(query)
    #   2. 用 RRF 算法合并两个排名：
    #      RRF(doc) = 1/(60+rank_bm25) + 1/(60+rank_dense)
    #   3. 按 RRF 分数从高到低排序
    # weights=[0.5, 0.5] 表示两路等权重（可调：FAQ 场景 Dense 权重高，API 文档 BM25 权重高）
    ensemble = EnsembleRetriever(retrievers=[bm25, dense], weights=[0.5, 0.5])

    # ── Step 4: Cohere Reranker 精排（可选）──
    cohere_key = os.getenv("COHERE_API_KEY")
    if cohere_key:
        # CohereRerank 内部做的事：
        #   1. 把 (query, doc1), (query, doc2), ... 逐个发给 Cohere 的 rerank API
        #   2. Cohere 用 Cross-Encoder 模型对每对打分（不是简单的向量相似度）
        #   3. 按分数从高到低重排，只保留 top_n=5 个
        compressor = CohereRerank(
            model="rerank-v3.5",       
            top_n=RERANK_TOP_N,       
            cohere_api_key=cohere_key,  
        )

        # ContextualCompressionRetriever 把粗排+精排串联起来：
        #   base_retriever（ensemble）→ 先粗排召回 ~20 个候选
        #   base_compressor（compressor）→ 再精排，只保留 5 个最相关的
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble,
        )

    return ensemble