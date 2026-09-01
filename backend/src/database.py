"""backend/src/database.py

PostgreSQL 异步引擎 + Session 工厂。
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/capstone_qa",
)

# echo=True 打印 SQL（开发用），生产关掉
# pool_size=10 + max_overflow=20 = 最高 30 个并发 DB 连接
engine = create_async_engine(DATABASE_URL, echo=False, pool_size=10, max_overflow=20)

# async_sessionmaker：异步 Session 工厂
AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI 依赖：每个请求一个 AsyncSession，请求结束自动关闭"""
    async with AsyncSessionLocal() as session:
        yield session
