"""
记忆压缩策略：Buffer + Summary 增量混合

三种策略：
  A. Token 窗口截断（trim_messages，保留最新 N 条 + system prompt）
  B. LLM 摘要压缩（旧历史 → summary 文本）
  C. Buffer + Summary 增量混合（生产级，每次追加新摘要而非全量重算）

Capstone 默认启用策略 C，通过 --no-compress 可关闭用于对比测试。
"""
from typing import Annotated, TypedDict
import operator

from langchain_openai import ChatOpenAI
from langchain_core.messages import trim_messages, SystemMessage

from .config import LLM_MODEL, OPENAI_API_KEY

# 压缩阈值：超过 N 条消息触发压缩
COMPRESS_THRESHOLD = 20
# 保留最近 N 条原始消息
KEEP_RECENT = 10


class CompressibleState(TypedDict):
    messages: Annotated[list, operator.add]
    summary: str


def _get_llm():
    return ChatOpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY)


# ── 策略 A：Token 窗口截断（最简单，不需要 LLM）──

def trim_by_token_window(messages: list, max_tokens: int = 4000) -> list:
    """保留最新消息，超出 max_tokens 的旧消息被丢弃，system prompt 永远保留"""
    llm = _get_llm()
    return trim_messages(
        messages,
        max_tokens=max_tokens,
        strategy="last",
        token_counter=llm,
        include_system=True,
    )


# ── 策略 B：LLM 摘要压缩（全量重算，简单但有信息损失风险）──

def summarize_and_compress(messages: list, keep_recent: int = KEEP_RECENT) -> list:
    """旧历史 → LLM 生成摘要 → 摘要 + 最近 N 条"""
    if len(messages) <= keep_recent:
        return messages

    llm = _get_llm()
    old = messages[:-keep_recent]
    recent = messages[-keep_recent:]

    summary = llm.invoke(
        f"用 2-3 句话总结以下对话的关键信息（用户身份、讨论话题、待办事项）：\n\n{old}"
    ).content

    return [
        SystemMessage(content=f"历史对话摘要：{summary}"),
        *recent,
    ]


# ── 策略 C：Buffer + Summary 增量混合（生产级，推荐）──

def hybrid_compress(state: CompressibleState) -> dict:
    """Buffer + Summary 增量混合

    - messages 长度 <= COMPRESS_THRESHOLD：不压缩，返回空
    - 超过阈值：把最近 10 条 + 现有摘要 → LLM 生成新摘要
    - 保留最新 10 条原始消息 + 更新后的摘要

    增量式 vs 全量式：
    - 全量式（策略 B）：每次把所有旧历史喂给 LLM 重新总结 → 信息可能丢失
    - 增量式（策略 C）：基于现有摘要 + 新增对话更新 → 信息保留更好
    """
    messages = state.get("messages", [])
    current_summary = state.get("summary", "")

    if len(messages) <= COMPRESS_THRESHOLD:
        return {}  # 不需要压缩

    llm = _get_llm()
    new_summary = llm.invoke(
        f"现有摘要：{current_summary}\n\n新增对话：{messages[-KEEP_RECENT:]}\n\n"
        f"请更新摘要，保留关键信息（用户身份、讨论话题、待办事项）。"
    ).content

    return {
        "messages": messages[-KEEP_RECENT:],  # 只保留最近 10 条
        "summary": new_summary,
    }


# ── 便捷函数：估算 messages 的 token 数（用于对比测试）──

def estimate_tokens(messages: list) -> int:
    """粗略估算 token 数：中文 1 字 ≈ 1.5 token，英文 1 词 ≈ 1.3 token"""
    total_chars = 0
    for msg in messages:
        content = msg.content if hasattr(msg, "content") else str(msg)
        total_chars += len(str(content))
    # 中文为主的粗略估算
    return int(total_chars * 1.5)