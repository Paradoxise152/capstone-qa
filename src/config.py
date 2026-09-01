import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ──
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = "gpt-4o-mini"

# ── RAG ──
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
DENSE_K = 10          # 粗排：向量检索 Top-K
BM25_K = 10             # 粗排：BM25 检索 Top-K
RERANK_TOP_N = 5        # 精排后保留

# ── Data ──
KNOWLEDGE_FILE = "data/knowledge_base.md"