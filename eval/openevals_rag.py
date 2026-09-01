"""
RAG 检索评估（openevals 升级版）— 替代手写 Precision/Recall

使用 LangChain 官方 openevals 库：
- retrieval_relevance: 检索到的文档是否与问题相关
- groundedness: Agent 回答是否有据可查
- correctness: Agent 回答是否准确

安装：pip install openevals
"""

from openevals.llm import create_llm_as_judge
from openevals.prompts import (
    RAG_RETRIEVAL_RELEVANCE_PROMPT,
    RAG_GROUNDEDNESS_PROMPT,
    CORRECTNESS_PROMPT,
)


def build_retrieval_evaluator(model: str = "openai:gpt-4o-mini"):
    """检索相关性：搜出来的文档是否与问题相关"""
    return create_llm_as_judge(
        prompt=RAG_RETRIEVAL_RELEVANCE_PROMPT,
        model=model,
        feedback_key="retrieval_relevance",
    )


def build_groundedness_evaluator(model: str = "openai:gpt-4o-mini"):
    """回答有据性：Agent 的回答是否基于检索文档，而非编造"""
    return create_llm_as_judge(
        prompt=RAG_GROUNDEDNESS_PROMPT,
        model=model,
        feedback_key="groundedness",
    )


def build_correctness_evaluator(model: str = "openai:gpt-4o-mini"):
    """回答正确性：Agent 回答 vs 期望答案"""
    return create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        model=model,
        feedback_key="correctness",
    )


# 评估数据集
EVAL_CASES = [
    {
        "question": "LangGraph 怎么部署？",
        "reference": "pip install langgraph，定义 StateGraph，Docker Compose 部署",
    },
    {
        "question": "产品多少钱？",
        "reference": "基础版免费，专业版999元/月，企业版4999元/月",
    },
    {
        "question": "怎么重置密码？",
        "reference": "登录→忘记密码→邮箱→重置链接→新密码",
    },
    {
        "question": "退款政策是什么？",
        "reference": "7天内全额退款，超过按剩余天数折算",
    },
    {
        "question": "苹果有什么营养？",
        "reference": "苹果富含维生素C和膳食纤维",
    },
]


def run_rag_evals(retriever, agent):
    """运行完整 RAG 评估"""
    retrieval_eval = build_retrieval_evaluator()
    groundedness_eval = build_groundedness_evaluator()
    correctness_eval = build_correctness_evaluator()

    results = {"retrieval_relevance": [], "groundedness": [], "correctness": []}

    for case in EVAL_CASES:
        q = case["question"]
        ref = case["reference"]

        # 1. 检索
        docs = retriever.invoke(q)
        doc_texts = " | ".join(d.page_content[:100] for d in docs)

        # 2. Agent 回答（模拟：实际应调用 graph.invoke）
        if agent:
            answer = agent.invoke({"messages": [{"role": "user", "content": q}]})  # type: ignore[arg-type]
            answer_text = answer["messages"][-1].content  # type: ignore[index]
            answer_text = str(answer_text)
        else:
            answer_text = f"[模拟回答] {doc_texts[:200]}"

        # 3. 三项评估
        r = retrieval_eval(
            inputs={"question": q},
            context={"documents": [doc_texts]},
        )
        results["retrieval_relevance"].append(r["score"])  # type: ignore[index]

        g = groundedness_eval(
            outputs={"answer": answer_text},
            context={"documents": [doc_texts]},
        )
        results["groundedness"].append(g["score"])  # type: ignore[index]

        c = correctness_eval(
            inputs=q,
            outputs=answer_text,
            reference_outputs=ref,
        )
        results["correctness"].append(c["score"])  # type: ignore[index]

        print(f"  [{q[:30]}...] retrieval={r['score']} grounded={g['score']} correct={c['score']}")  # type: ignore[index]

    # 汇总
    for key in results:
        scores = results[key]
        avg = sum(1 for s in scores if s) / len(scores) if scores else 0
        print(f"{key}: {avg:.1%} ({sum(1 for s in scores if s)}/{len(scores)})")

    return results


def run_retrieval_only(retriever):
    """仅运行检索评估（不需要 Agent，更快）"""
    return run_rag_evals(retriever, agent=None)