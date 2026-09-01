"""
路由硬规则：100% 可靠的正则匹配兜底

独立模块，不依赖 LLM 初始化——便于单元测试。
"""
import re

# 支持的 Agent 列表
VALID_AGENTS = {"knowledge", "ticket", "faq", "general", "human"}

# ── 硬规则兜底：100% 可靠，不受 LLM 幻觉影响 ──
TICKET_ID_PATTERN = re.compile(r"TK-\d{4}-\d{4}", re.IGNORECASE)

HARD_RULES = [
    # (正则, 路由目标)
    (TICKET_ID_PATTERN, "ticket"),                          # 明确工单号 → 直接走 ticket
    (re.compile(r"转人工|人工客服|我要投诉|解决不了"), "human"),  # 转人工关键词
]


def hard_rule_route(user_input: str) -> str | None:
    """硬规则检查：命中直接返回，未命中返回 None 交给 LLM

    >>> hard_rule_route("查 TK-2024-0001")
    'ticket'
    >>> hard_rule_route("转人工")
    'human'
    >>> hard_rule_route("LangGraph 怎么部署")
    None
    """
    for pattern, target in HARD_RULES:
        if pattern.search(user_input):
            return target
    return None


def route_fallback(next_agent: str) -> str:
    """低置信度降级：LLM 输出不在合法值 → general"""
    if next_agent in VALID_AGENTS:
        return next_agent
    return "general"