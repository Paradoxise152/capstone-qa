"""
RAG 检索评估：Precision（精确率）和 Recall（召回率）

来自 08.md 评估工程 + Capstone-04
"""
from langchain_core.documents import Document
from src.rag.pipeline import build_retrieval_pipeline
from src.config import KNOWLEDGE_FILE


# 评估数据集：每行 (问题, 期望命中的关键词)
EVAL_DATASET = [
    ("LangGraph 怎么部署？", ["Docker", "Python 3.12", "pip install"]),
    ("产品多少钱？", ["基础版", "专业版", "企业版", "999", "4999"]),
    ("怎么重置密码？", ["忘记密码", "邮箱", "重置链接"]),
    ("退款政策是什么？", ["7 天", "全额退款", "折算"]),
    ("苹果有什么营养？", ["维生素 C", "膳食纤维"]),
    ("工单状态有哪些？", ["待处理", "处理中", "已完成", "已关闭"]),
]


def evaluate():
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from src.config import CHUNK_SIZE, CHUNK_OVERLAP

    loader = TextLoader(KNOWLEDGE_FILE)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)

    retriever = build_retrieval_pipeline(chunks)

    total_correct = 0
    total_retrieved = 0
    total_expected = 0

    for query, expected_keywords in EVAL_DATASET:
        results = retriever.invoke(query)
        texts = " ".join(d.page_content for d in results)
        hits = sum(1 for kw in expected_keywords if kw in texts)
        total_correct += hits
        total_retrieved += len(results)
        total_expected += len(expected_keywords)

    precision = total_correct / total_retrieved if total_retrieved else 0
    recall = total_correct / total_expected if total_expected else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print(f"评测数据集: {len(EVAL_DATASET)} 条")
    print(f"Precision: {precision:.1%}  Recall: {recall:.1%}  F1: {f1:.1%}")
    return {"precision": precision, "recall": recall, "f1": f1}


if __name__ == "__main__":
    evaluate()