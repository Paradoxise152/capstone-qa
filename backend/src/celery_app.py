"""backend/src/celery_app.py

Celery 实例 + 配置（05.md 新增）。

Broker/Backend 都用 Redis（04.md 已起，复用），不同 db 号区分：
  db=0 缓存（cache.py 用）
  db=1 Broker（任务消息）
  db=2 Backend（任务结果）

可靠性配置（生产级关键）：
  - task_acks_late=True：Worker 执行完才 ACK（默认取走就 ACK，崩溃会丢任务）
  - task_reject_on_worker_lost=True：Worker 进程崩溃时 reject 任务重投
  - worker_prefetch_multiplier=1：避免一个 Worker 抢一堆任务
"""

from __future__ import annotations

import os

from celery import Celery

# 复用 04.md 的 Redis，但用不同 db 号
REDIS_BASE = os.getenv("REDIS_URL", "redis://localhost:6379")
BROKER_URL = f"{REDIS_BASE}/1"
BACKEND_URL = f"{REDIS_BASE}/2"

app = Celery(
    "capstone_backend",
    broker=BROKER_URL,
    backend=BACKEND_URL,
    include=["src.tasks"],  # Worker 启动时自动导入任务定义
)

# ── 可靠性配置（生产级关键）──
app.conf.update(
    # 任务执行完才 ACK（默认是取走就 ACK，Worker 崩溃会丢任务）
    task_acks_late=True,
    # Worker 进程崩溃时 reject 任务（让 Broker 重投给其他 Worker）
    task_reject_on_worker_lost=True,
    # 预取只取 1 个（避免一个 Worker 抢一堆任务，其他 Worker 闲着）
    worker_prefetch_multiplier=1,
    # 结果序列化格式
    result_serializer="json",
    accept_content=["json"],
    # 结果保留 1 小时（前端轮询拿结果的窗口）
    result_expires=3600,
    # 时区
    timezone="Asia/Shanghai",
    enable_utc=False,
)
