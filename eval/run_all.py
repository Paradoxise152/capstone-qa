"""
评估一键运行：检索 + 路由 + 正确性

运行方式：
  python -m eval.run_all           # 全部评估
  python -m eval.run_all --rag     # 仅 RAG
  python -m eval.run_all --route   # 仅路由
  python -m eval.run_all --agent   # 仅 Agent 端到端
"""
import sys
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

from src.config import CHUNK_SIZE, CHUNK_OVERLAP, KNOWLEDGE_FILE
from src.rag.pipeline import build_retrieval_pipeline


def run_all():
    mode = sys.argv[1] if len(sys.argv) > 1 else "--all"

    print("=" * 50)
    print("  企业知识库 Agent — 评估报告")
    print("  openevals v0.2.0")
    print("=" * 50)

    # 加载知识库 + 构建检索器
    loader = TextLoader(KNOWLEDGE_FILE)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    retriever = build_retrieval_pipeline(chunks)

    if mode in ("--all", "--rag"):
        print("\n📚 RAG 检索评估")
        print("-" * 30)
        from eval.openevals_rag import run_retrieval_only
        run_retrieval_only(retriever)

    if mode in ("--all", "--route"):
        print("\n🧭 Supervisor 路由评估")
        print("-" * 30)
        from eval.openevals_agent import run_routing_checks
        llm = ChatOpenAI(model="gpt-4o-mini")
        run_routing_checks(llm)

    print("\n✅ 评估完成")


if __name__ == "__main__":
    run_all()