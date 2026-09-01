"""backend/src/cache.py

Redis 异步客户端 + Cache-Aside 辅助函数 + 幂等创建。

用 redis.asyncio（异步版本，不卡 event loop）。
"""

from __future__ import annotations

import json
import os
import random
from typing import Any

import redis.asyncio as redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# decode_responses=True：返回 str 而不是 bytes，方便直接 json.loads
redis_client = redis.from_url(REDIS_URL, decode_responses=True)


async def cache_get(key: str) -> Any | None:
    """查缓存，命中返回反序列化后的值，未命中返回 None"""
    val = await redis_client.get(key)
    if val is None:
        return None
    return json.loads(val)


async def cache_set(key: str, value: Any, ttl: int = 600) -> None:
    """写缓存 + TTL（秒）。TTL 加随机抖动防雪崩"""
    jittered_ttl = ttl + random.randint(0, 60)  # 0-60s 随机抖动
    await redis_client.set(key, json.dumps(value), ex=jittered_ttl)


async def cache_delete(key: str) -> None:
    """删缓存（Cache-Aside 写 DB 后调用，而非更新缓存）"""
    await redis_client.delete(key)


async def idempotent_create(idempotency_key: str, ttl: int = 3600) -> str | None:
    """幂等创建：用 SETNX 抢占幂等 key

    返回 None = 抢到（可以继续创建），返回非 None = 之前已处理过（值是 ticket_id 或标记）
    本篇简化版：只记录"处理过"标记，05.md 会升级为存完整响应。
    """
    # SETNX（nx=True）：key 不存在才设置，返回 True；已存在返回 False
    acquired = await redis_client.set(idempotency_key, "processing", ex=ttl, nx=True)
    if acquired:
        return None  # 抢到，继续创建
    # 已存在：返回已存的值
    existing = await redis_client.get(idempotency_key)
    if existing == "processing":
        # 其他请求正在处理中（罕见情况），本篇简化返回 None 让它也走创建逻辑
        # 生产应该等一会重试或返回 409
        return None
    return existing
