"""backend/src/tasks.py

Celery 任务定义（05.md 新增）。

这些函数在 Worker 进程里跑，不在 FastAPI 进程。

关键：Agent 推理是长时任务（30-60s），放 Worker 跑不阻塞 FastAPI event loop。
"""

from __future__ import annotations

import asyncio
import logging

from .celery_app import app
from .agent_bridge import ainvoke_agent_with_timeout

logger = logging.getLogger(__name__)


@app.task(bind=True, max_retries=3, name="src.tasks.run_agent_task")
def run_agent_task(self, message: str, thread_id: str = "default") -> dict:
    """Agent 推理任务（在 Worker 进程跑）

    bind=True：self 是 Celery 任务实例，可用 self.retry()
    max_retries=3：失败最多重试 3 次
    name=...：显式命名，避免 Celery 自动用模块名+函数名生成（重构时易断）

    返回 dict 而不是 ChatResponse 对象，因为 Celery Backend 用 JSON 序列化。

    幂等性：Agent 推理本身无副作用（只读 Checkpointer + 调 LLM API），
    Worker 崩溃重投不会产生副作用叠加，所以不需要额外幂等保护。
    如果任务里有写 DB/发邮件等副作用，必须用 Idempotency-Key（04.md 学过）。
    """
    logger.info(
        f"[task {self.request.id}] 开始推理：message={message!r}, thread_id={thread_id}"
    )
    try:
        # Celery Worker 是同步上下文，但 agent_bridge 是异步函数
        # 用 asyncio.run 起一个临时 event loop 跑异步函数
        answer, routed = asyncio.run(
            ainvoke_agent_with_timeout(message, thread_id, timeout_s=120.0)
        )
        result = {"answer": answer, "routed_agent": routed, "thread_id": thread_id}
        logger.info(f"[task {self.request.id}] 完成：routed={routed}")
        return result
    except TimeoutError as e:
        logger.warning(f"[task {self.request.id}] 超时，将重试：{e}")
        # 指数退避：2^retries 秒后重试（2/4/8 秒）
        raise self.retry(exc=e, countdown=2 ** self.request.retries)
    except Exception as e:
        logger.error(f"[task {self.request.id}] 失败：{e}")
        # 其他异常也重试，但记日志
        raise self.retry(exc=e, countdown=2 ** self.request.retries)


@app.task(name="src.tasks.health_check")
def health_check() -> str:
    """Worker 健康检查任务（用于验证 Worker 是否在线）"""
    return "pong"
