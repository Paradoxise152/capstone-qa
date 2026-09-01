"""
Supervisor 路由评估（openevals trajectory match）

验证 Supervisor 是否正确路由到目标 Agent。
例如："怎么退款"应该路由到 faq，不是 knowledge。
"""

import json
from openevals.trajectory import create_trajectory_match_evaluator


def build_route_evaluator():
    """严格匹配模式：工具调用顺序必须完全一致"""
    return create_trajectory_match_evaluator(
        trajectory_match_mode="strict",
        # 宽松匹配参数内容（"San Francisco" vs "SF" 视为相同）
        tool_args_match_mode="ignore",
    )


# 路由测试用例：(用户输入, 期望Agent)
ROUTING_TESTS = [
    ("LangGraph 怎么部署？", "knowledge"),
    ("查工单 TK-2024-0001", "ticket"),
    ("怎么重置密码？", "faq"),
    ("你们产品多少钱？", "knowledge"),  # 定价在知识库里，不在 FAQ
    ("怎么退款？", "faq"),              # 退款在 FAQ
    ("转人工", "human"),
    ("你好", "general"),
]


def evaluate_routing(results: list[tuple[str, str]]):
    """对比实际路由 vs 期望路由"""
    correct = sum(1 for actual, expected in results if actual == expected)
    total = len(results)
    print(f"Routing accuracy: {correct}/{total} = {correct/total:.1%}")
    for actual, expected in results:
        status = "✅" if actual == expected else "❌"
        print(f"  {status} expected={expected} actual={actual}")
    return correct / total if total else 0


def run_routing_checks(supervisor_llm):
    """运行路由检查（不启动完整 Agent，只测 Supervisor LLM）"""
    results = []
    for question, expected in ROUTING_TESTS:
        response = supervisor_llm.invoke([
            {"role": "system", "content": (
                "你是路由 Supervisor。根据用户消息输出下一步 Agent：\n"
                "- 问知识/文档/怎么/什么是/what/how → knowledge\n"
                "- 工单/订单/TK-/查工单/创建工单 → ticket\n"
                "- 密码/退款/发票/API限额/套餐/FAQ/常见问题 → faq\n"
                "- 转人工/人工客服/投诉 → human\n"
                "- 问候/闲聊 → general\n"
                "只输出一个词。"
            )},
            {"role": "user", "content": question},
        ])
        actual = response.content.strip().lower()
        results.append((actual, expected))

    return evaluate_routing(results)


if __name__ == "__main__":
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini")
    run_routing_checks(llm)