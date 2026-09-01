"""
Human-in-the-Loop 模块：敏感操作暂停审批

来自 Capstone-03：HITL 转人工审批
"""
from langgraph.types import interrupt


def human_approval_required(state: dict) -> dict:
    """
    需要人工审批时调用。
    暂停图执行，等待外部输入 'approved' 或 'rejected'。
    """
    last_msg = state["messages"][-1]
    user_question = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    # interrupt() 暂停图执行，返回提示信息给调用方
    # 调用方通过 Command(resume="approved") 或 Command(resume="rejected") 恢复
    decision = interrupt({
        "type": "human_approval",
        "message": f"需要人工处理的问题：{user_question[:200]}",
        "options": ["approved（转人工处理）", "rejected（告知用户稍后联系）"],
    })

    if decision == "approved":
        return {
            "messages": [{
                "role": "assistant",
                "content": "已转接人工客服，请稍候。预计等待时间：2 分钟。工单号：TK-ESC-001"
            }],
            "escalated": True,
        }
    else:
        return {
            "messages": [{
                "role": "assistant",
                "content": "抱歉无法立即处理，已记录您的问题。客服将在 1 个工作日内通过邮件联系您。"
            }],
            "escalated": False,
        }