"""backend/src/models.py

SQLAlchemy 2.0 ORM 模型定义。
"""

from __future__ import annotations

import random
from datetime import datetime

from sqlalchemy import String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """SQLAlchemy 2.0 声明式基类"""


class Ticket(Base):
    """工单表

    设计：内部自增主键 id（高效索引） + 业务键 ticket_id（对外暴露）
    """

    __tablename__ = "tickets"

    # 内部自增主键
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    # 业务键（对外暴露，唯一索引）
    ticket_id: Mapped[str] = mapped_column(
        String(16), unique=True, nullable=False, index=True
    )

    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # 状态：open / processing / closed
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="open", index=True
    )
    priority: Mapped[str] = mapped_column(String(10), nullable=False, default="中")
    category: Mapped[str | None] = mapped_column(String(50), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now(), onupdate=func.now()
    )

    def __repr__(self) -> str:
        return f"<Ticket {self.ticket_id}: {self.title}>"


def generate_ticket_id() -> str:
    """生成业务键 TK-YYYY-NNNN

    简化版：随机 4 位。生产应该用 DB 序列或 Redis 自增保证唯一性，
    随机有极小概率冲突，靠唯一索引兜底（冲突时重试）。
    """
    year = datetime.now().year
    n = random.randint(1, 9999)
    return f"TK-{year}-{n:04d}"
