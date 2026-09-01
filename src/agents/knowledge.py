"""Knowledge Agent：RAG 三路召回 + 精排 — 处理知识问答"""

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from ..config import LLM_MODEL, OPENAI_API_KEY


def build_knowledge_agent(retriever):
    """创建知识库 Agent，绑定 RAG 检索管线"""
    llm = ChatOpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY)

    def retrieve(query: str) -> str:
        """从知识库检索相关内容"""
        docs = retriever.invoke(query)
        return "\n\n---\n".join(d.page_content for d in docs)

    return create_agent(
        llm,
        tools=[retrieve],  # type: ignore[arg-type]
        system_prompt="""你是企业知识库助手。回答问题时：
1. 先用检索工具查知识库
2. 基于检索结果回答，不要编造
3. 如果知识库里没有相关信息，诚实告知""",
    )