"""FAQ Agent：预设常见问题匹配 — 快速回答高频问题"""

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from ..config import LLM_MODEL, OPENAI_API_KEY

# 预设 FAQ 库
FAQ_DB = {
    "重置密码": "登录页面 → 忘记密码 → 输入注册邮箱 → 点击邮件中的重置链接 → 设置新密码。整个过程约 2 分钟。",
    "升级套餐": "控制台 → 账户设置 → 套餐管理 → 选择目标套餐 → 确认支付。新套餐即时生效。",
    "退款政策": "购买后 7 天内可全额退款。超过 7 天按剩余天数折算退款。退款原路返回，3-5 个工作日到账。",
    "发票申请": "控制台 → 费用中心 → 发票管理 → 填写开票信息 → 提交。电子发票 1 个工作日内发送到注册邮箱。",
    "API限额": "基础版 1000次/月，专业版 50000次/月，企业版无限调用。在控制台可查看当前用量。",
    "数据导出": "控制台 → 数据管理 → 导出 → 选择格式(CSV/JSON) → 确认。大数据量导出可能需要几分钟。",
}


def build_faq_agent():
    llm = ChatOpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY)

    def lookup_faq(question: str) -> str:
        """在 FAQ 库中查找匹配问题"""
        for key, answer in FAQ_DB.items():
            if key in question:
                return f"【{key}】{answer}"
        return "FAQ 库中未找到匹配问题，建议转人工处理。"

    def list_faq_topics() -> str:
        """列出所有 FAQ 主题"""
        return "常见问题包括：" + "、".join(FAQ_DB.keys())

    return create_agent(
        llm,
        tools=[lookup_faq, list_faq_topics],
        system_prompt="""你是 FAQ 助手。先用 lookup_faq 匹配用户问题。
如果匹配成功，直接返回答案。如果失败，诚实告知并建议转人工。
不要编造 FAQ 库中不存在的答案。""",
    )