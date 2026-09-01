"""backend/src/agent_bridge.py

把 CLI 版 src.supervisor 的同步 graph 包装成 FastAPI 可用的异步调用。
这就是"给 CLI 版加后端外壳"的最薄一层——不重写 Agent 逻辑，只换入口。
"""

from __future__ import annotations

import asyncio
import sys
from functools import lru_cache
from pathlib import Path

from langchain_core.messages import HumanMessage

# 把上层 capstone-enterprise-qa/ 加进 sys.path，让 backend 能 import src.*
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # capstone-enterprise-qa/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.agents.faq import build_faq_agent  # type: ignore
from src.agents.knowledge import build_knowledge_agent  # type: ignore
from src.agents.ticket import build_ticket_agent  # type: ignore
from src.config import (  # type: ignore
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    KNOWLEDGE_FILE,
    LLM_MODEL,
    OPENAI_API_KEY,
)
from src.rag.pipeline import build_retrieval_pipeline  # type: ignore
from src.supervisor import build_supervisor  # type: ignore
from langchain.agents import create_agent  # type: ignore
from langchain_community.document_loaders import TextLoader  # type: ignore
from langchain_openai import ChatOpenAI  # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore


@lru_cache(maxsize=1)
def get_graph():
    """构造 Supervisor 图（lifespan 期间只构造一次）

    @lru_cache 保证单例——避免每个请求重建 5 个 Agent + Supervisor，
    否则每个请求都重新加载知识库、初始化 LLM 客户端，慢且费 Token。
    """
    loader = TextLoader(KNOWLEDGE_FILE)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    retriever = build_retrieval_pipeline(chunks)

    knowledge_agent = build_knowledge_agent(retriever)
    ticket_agent = build_ticket_agent()
    faq_agent = build_faq_agent()
    general_agent = create_agent(
        ChatOpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY),  # type: ignore[arg-type]
        tools=[],
        system_prompt="你是通用助手。处理问候、闲聊和无法归类的问题。",
    )

    graph = build_supervisor(knowledge_agent, ticket_agent, faq_agent, general_agent)
    return graph


async def ainvoke_agent(message: str, thread_id: str = "default") -> tuple[str, str]:
    """异步调 Supervisor 图，返回 (answer, routed_agent)

    关键：用 ainvoke（异步版本）而非 invoke（同步）。
    LangGraph 的 ainvoke 内部用 httpx 异步调 LLM API，不卡 event loop。
    """
    graph = get_graph()
    config: dict = {"configurable": {"thread_id": thread_id}}

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=message)]},  # type: ignore[arg-type]
        config,  # type: ignore[arg-type]
    )

    answer = result["messages"][-1].content
    routed_agent = str(result.get("next_agent", "general"))
    return str(answer), routed_agent


async def ainvoke_agent_with_timeout(
    message: str, thread_id: str, timeout_s: float = 60.0
) -> tuple[str, str]:
    """带超时的异步调用——防止 LLM 卡住整条请求

    生产级必须设超时。LLM API 可能因网络/限流卡 60 秒+，
    不设超时 → 用户请求挂死 → 连接池耗尽 → 服务不可用。
    """
    return await asyncio.wait_for(
        ainvoke_agent(message, thread_id),
        timeout=timeout_s,
    )
