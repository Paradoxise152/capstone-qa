"""
Supervisor 三层路由兜底策略测试

测试三层策略：
  ① 硬规则兜底（TK-XXXX-XXXX → ticket，不受 LLM 幻觉影响）
  ② LLM 语义判断（覆盖 90% 场景，需要 API Key，离线跳过）
  ③ 低置信度降级（LLM 输出不在合法值 → general 反问）

运行：python -m pytest tests/test_routing.py -v
"""
import os
import pytest


# ── 第一层：硬规则兜底测试（不需要 API Key）──

# 导入独立的路由规则模块，不触发 supervisor.py 里的 LLM 初始化
from src.routing_rules import hard_rule_route, VALID_AGENTS, route_fallback


class TestHardRule:
    """硬规则应该 100% 命中，不依赖 LLM"""

    def test_ticket_id_pattern(self):
        """含 TK-XXXX-XXXX 工单号 → 直接走 ticket"""
        assert hard_rule_route("查一下工单 TK-2024-0001") == "ticket"
        assert hard_rule_route("TK-2024-0001 是什么问题") == "ticket"
        assert hard_rule_route("帮我查 TK-2024-9999 的状态") == "ticket"

    def test_ticket_id_case_insensitive(self):
        """工单号大小写不敏感"""
        assert hard_rule_route("tk-2024-0001") == "ticket"
        assert hard_rule_route("查 Tk-2024-0001") == "ticket"

    def test_human_keyword(self):
        """转人工关键词 → 直接走 human"""
        assert hard_rule_route("转人工") == "human"
        assert hard_rule_route("我要投诉你们产品") == "human"
        assert hard_rule_route("这个问题解决不了") == "human"

    def test_no_hard_rule_returns_none(self):
        """未命中硬规则 → 返回 None 交给 LLM"""
        assert hard_rule_route("LangGraph 怎么部署") is None
        assert hard_rule_route("怎么重置密码") is None
        assert hard_rule_route("你好") is None
        assert hard_rule_route("产品多少钱") is None

    def test_edge_case_ticket_in_question(self):
        """边界 case：含工单号但实际是知识问答 → 硬规则强制走 ticket
        这是已知限制：硬规则优先于 LLM 语义判断，
        但工单号出现基本都意味着工单操作，所以可接受"""
        assert hard_rule_route("TK-2024-0001 是什么状态") == "ticket"
        assert hard_rule_route("TK-2024-0001 这个工单系统怎么用") == "ticket"


# ── 第三层：低置信度降级测试（不需要 API Key）──

class TestLowConfidenceFallback:
    """LLM 输出非法值时 route 函数降级到 general"""

    def test_route_valid_agents(self):
        """合法值正常路由"""
        for agent in VALID_AGENTS:
            assert route_fallback(agent) == agent

    def test_route_invalid_falls_to_general(self):
        """非法值降级到 general"""
        assert route_fallback("unknown") == "general"
        assert route_fallback("") == "general"
        assert route_fallback("invalid_agent") == "general"


# ── 三层策略集成测试 ──

class TestThreeLayerStrategy:
    """三层策略集成：硬规则优先 → LLM 判断 → 低置信度降级"""

    def test_hard_rule_overrides_llm(self):
        """硬规则优先于 LLM——含工单号即使 LLM 想路由到 knowledge 也会走 ticket"""
        assert hard_rule_route("TK-2024-0001 怎么用") == "ticket"
        # 不含工单号 → 硬规则返回 None，才会调 LLM
        assert hard_rule_route("工单系统怎么用") is None

    def test_fallback_chain(self):
        """兜底链：硬规则 None → LLM → 低置信度 general"""
        # 1. 硬规则命中
        assert hard_rule_route("TK-2024-0001") == "ticket"
        # 2. 硬规则未命中，需要 LLM（这里只测硬规则返回 None）
        assert hard_rule_route("你好") is None
        # 3. LLM 输出非法值时 route_fallback 降级到 general
        assert route_fallback("invalid") == "general"

    def test_hard_rules_coverage(self):
        """验证硬规则覆盖的关键场景"""
        # 工单号
        assert hard_rule_route("查 TK-2024-0001") == "ticket"
        # 转人工
        assert hard_rule_route("转人工") == "human"
        assert hard_rule_route("我要投诉") == "human"
        assert hard_rule_route("解决不了") == "human"


# ── 第二层：LLM 路由测试（需要 API Key，离线跳过）──

@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="需要 OPENAI_API_KEY",
)
class TestLLMRouting:
    """LLM 语义判断——需要 API Key 才能跑"""

    def test_llm_routes_knowledge(self):
        from src.supervisor import supervisor_node
        state = {"messages": [{"role": "user", "content": "LangGraph 怎么部署"}], "next_agent": ""}
        result = supervisor_node(state)
        assert result["next_agent"] == "knowledge"

    def test_llm_routes_faq_not_knowledge(self):
        """关键边界 case：'怎么退款' 应走 faq 不是 knowledge"""
        from src.supervisor import supervisor_node
        state = {"messages": [{"role": "user", "content": "怎么退款"}], "next_agent": ""}
        result = supervisor_node(state)
        assert result["next_agent"] == "faq"