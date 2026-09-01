"""
Mock 工单 MCP Server — 模拟企业内部工单系统

不依赖真实数据库，用内存字典模拟：
- 查询工单（按工单号或关键词）
- 创建工单
- 统计工单
"""
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationCapabilities
from mcp.server.stdio import stdio_server
import mcp.types as types

# 内存工单数据库（mock）
TICKETS = {
    "TK-2024-0001": {"title": "登录页面白屏", "status": "处理中", "priority": "高", "category": "技术"},
    "TK-2024-0002": {"title": "账单金额错误", "status": "待处理", "priority": "紧急", "category": "账单"},
    "TK-2024-0003": {"title": "API 调用超限", "status": "已完成", "priority": "中", "category": "技术"},
    "TK-2024-0004": {"title": "退款未到账", "status": "处理中", "priority": "高", "category": "账单"},
}

server = Server("ticket-system")


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="query_ticket",
            description="按工单号查询工单详情。参数 ticket_id 格式如 TK-2024-0001",
            inputSchema={
                "type": "object",
                "properties": {
                    "ticket_id": {"type": "string", "description": "工单号"}
                },
                "required": ["ticket_id"],
            },
        ),
        types.Tool(
            name="search_tickets",
            description="按关键词搜索工单（标题匹配）",
            inputSchema={
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "搜索关键词"}
                },
                "required": ["keyword"],
            },
        ),
        types.Tool(
            name="create_ticket",
            description="创建新工单",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "priority": {"type": "string", "enum": ["低", "中", "高", "紧急"]},
                    "category": {"type": "string", "enum": ["技术", "产品", "账单", "其他"]},
                },
                "required": ["title"],
            },
        ),
        types.Tool(
            name="ticket_stats",
            description="统计工单状态分布",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "query_ticket":
        tid = arguments["ticket_id"]
        if tid in TICKETS:
            t = TICKETS[tid]
            return [types.TextContent(type="text", text=f"工单 {tid}: {t['title']} | 状态:{t['status']} | 优先级:{t['priority']}")]
        return [types.TextContent(type="text", text=f"工单 {tid} 不存在")]

    elif name == "search_tickets":
        kw = arguments["keyword"].lower()
        results = [f"{tid}: {t['title']}" for tid, t in TICKETS.items() if kw in t["title"].lower()]
        return [types.TextContent(type="text", text="找到 " + "; ".join(results) if results else "未找到匹配工单")]

    elif name == "create_ticket":
        tid = f"TK-2024-{len(TICKETS)+1:04d}"
        TICKETS[tid] = {"title": arguments["title"], "priority": arguments.get("priority", "中"), "category": arguments.get("category", "其他"), "status": "待处理"}
        return [types.TextContent(type="text", text=f"工单已创建: {tid}")]

    elif name == "ticket_stats":
        from collections import Counter
        statuses = Counter(t["status"] for t in TICKETS.values())
        text = "工单统计: " + ", ".join(f"{s}:{c}" for s, c in statuses.items())
        return [types.TextContent(type="text", text=text)]

    raise ValueError(f"未知工具：{name}")


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, InitializationCapabilities(sampling=types.SamplingCapability(), roots=types.RootsCapability()))

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())