"""
Step 2 Supervisor：路由 Knowledge / Ticket / FAQ / General / Human 五个 Agent

三层路由兜底策略：
  ① LLM 语义判断（覆盖 90% 场景）
  ② 硬规则正则兜底（TK-XXXX-XXXX 工单号直接走 ticket，不受 LLM 幻觉影响）
  ③ 低置信度降级（LLM 输出不在合法值里 → general 反问）
"""
import sqlite3
from typing import Annotated, TypedDict, Literal, cast
import operator

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI

from .config import LLM_MODEL, OPENAI_API_KEY
from .hitl import human_approval_required
from .routing_rules import hard_rule_route, route_fallback, VALID_AGENTS

# 支持的 Agent 列表
ALL_AGENTS = Literal["knowledge", "ticket", "faq", "general", "human"]


class CapstoneState(TypedDict):
    messages: Annotated[list, operator.add]
    next_agent: str


llm = ChatOpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY)


def supervisor_node(state: CapstoneState) -> dict:
    """Supervisor：三层路由策略
    ① 硬规则兜底（正则匹配，100% 可靠）
    ② LLM 语义判断（覆盖 90% 场景）
    ③ 低置信度降级（输出不在合法值 → general 反问）
    """
    last_msg = state["messages"][-1]
    content = cast(str, last_msg.content) if hasattr(last_msg, "content") else str(last_msg)

    # ── 第一层：硬规则兜底 ──
    hard_result = hard_rule_route(content)
    if hard_result:
        return {"next_agent": hard_result}

    # ── 第二层：LLM 语义判断 ──
    response = llm.invoke([
        {"role": "system", "content": (
            "你是路由 Supervisor。根据用户消息输出下一步 Agent：\n"
            "- 问知识/文档/怎么/什么是/what/how → knowledge\n"
            "- 工单/订单/查工单/创建工单 → ticket\n"
            "- 密码/退款/发票/API限额/套餐/FAQ/常见问题 → faq\n"
            "- 转人工/人工客服/投诉 → human\n"
            "- 问候/闲聊/不能处理的问题 → general\n"
            "只输出一个词。"
        )},
        {"role": "user", "content": content},
    ])
    llm_result = cast(str, response.content).strip().lower()

    # ── 第三层：低置信度降级 ──
    if llm_result in VALID_AGENTS:
        return {"next_agent": llm_result}
    # LLM 输出不在合法值 → 走 general 反问用户
    return {"next_agent": "general"}


def route(state: CapstoneState) -> ALL_AGENTS:
    """路由函数：将 supervisor 输出的 next_agent 映射到节点名"""
    return route_fallback(state["next_agent"])  # type: ignore[return-value]


def build_supervisor(knowledge_agent, ticket_agent, faq_agent, general_agent):
    """构建 Supervisor 图 — 5 路路由 + 三层兜底"""

    def knowledge_node(state: CapstoneState) -> dict:
        result = knowledge_agent.invoke({"messages": state["messages"]})  # type: ignore[arg-type]
        return {"messages": [result["messages"][-1]]}

    def ticket_node(state: CapstoneState) -> dict:
        result = ticket_agent.invoke({"messages": state["messages"]})  # type: ignore[arg-type]
        return {"messages": [result["messages"][-1]]}

    def faq_node(state: CapstoneState) -> dict:
        result = faq_agent.invoke({"messages": state["messages"]})  # type: ignore[arg-type]
        return {"messages": [result["messages"][-1]]}

    def general_node(state: CapstoneState) -> dict:
        result = general_agent.invoke({"messages": state["messages"]})  # type: ignore[arg-type]
        return {"messages": [result["messages"][-1]]}

    def human_node(state: CapstoneState) -> dict:
        """HITL 转人工审批：暂停图执行，等待人工确认"""
        return human_approval_required(state)

    builder = StateGraph(CapstoneState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("knowledge", knowledge_node)
    builder.add_node("ticket", ticket_node)
    builder.add_node("faq", faq_node)
    builder.add_node("general", general_node)
    builder.add_node("human", human_node)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges("supervisor", route, {
        "knowledge": "knowledge",
        "ticket": "ticket",
        "faq": "faq",
        "general": "general",
        "human": "human",
    })
    builder.add_edge("knowledge", "supervisor")
    builder.add_edge("ticket", "supervisor")
    builder.add_edge("faq", "supervisor")
    builder.add_edge("general", "supervisor")
    builder.add_edge("human", "supervisor")

    conn = sqlite3.connect("capstone_memory.db", check_same_thread=False)
    return builder.compile(checkpointer=SqliteSaver(conn))