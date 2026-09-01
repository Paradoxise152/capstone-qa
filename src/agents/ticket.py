"""Ticket Agent：绑定工单 MCP 工具 — 查询/创建/统计工单"""

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from ..config import LLM_MODEL, OPENAI_API_KEY


def build_ticket_agent():
    """创建工单 Agent，直接注入 mock 工单函数作为工具"""
    llm = ChatOpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY)

    # 工单数据（mock，与 ticket_mcp.py 共享同一份数据）
    TICKETS = {
        "TK-2024-0001": {"title": "登录页面白屏", "status": "处理中", "priority": "高", "category": "技术"},
        "TK-2024-0002": {"title": "账单金额错误", "status": "待处理", "priority": "紧急", "category": "账单"},
        "TK-2024-0003": {"title": "API 调用超限", "status": "已完成", "priority": "中", "category": "技术"},
        "TK-2024-0004": {"title": "退款未到账", "status": "处理中", "priority": "高", "category": "账单"},
    }

    def query_ticket(ticket_id: str) -> str:
        """按工单号查询工单详情"""
        if ticket_id in TICKETS:
            t = TICKETS[ticket_id]
            return f"工单 {ticket_id}: {t['title']} | 状态:{t['status']} | 优先级:{t['priority']} | 分类:{t['category']}"
        return f"工单 {ticket_id} 不存在"

    def search_tickets(keyword: str) -> str:
        """按关键词搜索工单"""
        results = [f"{tid}: {t['title']}" for tid, t in TICKETS.items() if keyword.lower() in t["title"].lower()]
        return "找到: " + "; ".join(results) if results else f"未找到含'{keyword}'的工单"

    def create_ticket(title: str, priority: str = "中", category: str = "其他") -> str:
        """创建新工单"""
        tid = f"TK-2024-{len(TICKETS)+1:04d}"
        TICKETS[tid] = {"title": title, "status": "待处理", "priority": priority, "category": category}
        return f"工单已创建: {tid}"

    def ticket_stats() -> str:
        """统计工单状态分布"""
        from collections import Counter
        statuses = Counter(t["status"] for t in TICKETS.values())
        return "工单统计: " + ", ".join(f"{s}:{c}" for s, c in statuses.items())

    return create_agent(
        llm,
        tools=[query_ticket, search_tickets, create_ticket, ticket_stats],
        system_prompt="""你是工单助手。负责：
- 查询工单状态（按工单号 TK-XXXX-XXXX 或关键词）
- 创建新工单
- 统计工单分布
如果用户没有提供足够信息，主动询问。""",
    )