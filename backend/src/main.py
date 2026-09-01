"""capstone-enterprise-qa / backend —— 后端外壳入口

04.md 阶段：加 Redis 缓存层
  - /chat 加 Cache-Aside 缓存（knowledge/faq 命中省 LLM 调用）
  - /tickets POST 加幂等 key（Idempotency-Key 防重复创建）
  - TTL + 随机抖动防雪崩

累进：05.md 起接 Celery 异步任务，06.md 起接 SSE 流式响应
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Literal, cast

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from .agent_bridge import ainvoke_agent_with_timeout, get_graph
from .cache import cache_get, cache_set, idempotent_create
from .database import get_db
from .models import Ticket, generate_ticket_id
from .tasks import run_agent_task
from celery.result import AsyncResult

load_dotenv()

# 工单状态 Literal 类型（ORM 返回 str，运行时是这些值，cast 给 Pydantic 响应模型）
TicketStatus = Literal["open", "processing", "closed"]


# ── lifespan：应用启动/关闭时跑的钩子 ──


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时构造 Supervisor 图（只一次），关闭时清理资源"""
    print("[lifespan] 构造 Supervisor 图 + 5 Agent...")
    graph = get_graph()  # @lru_cache 保证只构造一次
    app.state.graph = graph
    print("[lifespan] ✅ 就绪")
    yield
    print("[lifespan] 关闭中...")


app = FastAPI(
    title="capstone-enterprise-qa backend",
    description="服务端方向 L2 咬合：给 CLI 版 Agent 系统加后端外壳",
    version="0.5.0",
    lifespan=lifespan,
)


# ── 请求/响应模型（沿用 01.md 契约）──


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="用户消息")
    thread_id: str = Field(default="default", description="会话命名空间，无状态靠它取历史")


class ChatResponse(BaseModel):
    answer: str
    routed_agent: Literal["knowledge", "ticket", "faq", "general", "human"]
    thread_id: str


class TicketCreateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)


class TicketResponse(BaseModel):
    ticket_id: str
    title: str
    status: Literal["open", "processing", "closed"]


class TicketListResponse(BaseModel):
    items: list[TicketResponse]
    total: int


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class HistoryResponse(BaseModel):
    thread_id: str
    messages: list[HistoryMessage]


# ── 异步任务相关模型（05.md 起）──


class ChatTaskResponse(BaseModel):
    """POST /chat/async 异步任务模式：立即返回 task_id"""
    task_id: str
    status: Literal["pending"] = "pending"
    message: str  # 回显用户消息
    thread_id: str


class TaskStatusResponse(BaseModel):
    """GET /chat/tasks/{task_id} 轮询任务状态"""
    task_id: str
    status: Literal["pending", "started", "success", "failure"]
    result: dict | None = None  # status=success 时有值
    error: str | None = None  # status=failure 时有值


# ── 端点 ──


@app.get("/health")
async def health() -> dict[str, str]:
    """健康检查端点（K8s Readiness/Liveness Probe 用，10.md）"""
    return {"status": "ok"}


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """发起对话 —— 04.md 加 Redis 缓存热点问题

    Cache-Aside：先查缓存命中直接返回（省 LLM 调用），未命中调 Agent 后回填。
    只缓存 knowledge/faq 路由（知识问答答案稳定），ticket（有状态）和 general（闲聊）不缓存。
    """
    import hashlib

    cache_key = f"rag:cache:{hashlib.sha256(f'{req.thread_id}:{req.message}'.encode()).hexdigest()[:16]}"

    # ── 1. 查缓存 ──
    cached = await cache_get(cache_key)
    if cached is not None:
        # 命中：直接返回缓存结果（省一次 LLM 调用 + 省 Token）
        return ChatResponse(
            answer=cached["answer"],
            routed_agent=cached["routed_agent"],
            thread_id=req.thread_id,
        )

    # ── 2. 未命中：调 Agent ──
    try:
        answer, routed = await ainvoke_agent_with_timeout(
            req.message, req.thread_id, timeout_s=60.0
        )
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Agent 推理超时（60s）")
    except Exception as e:
        # 生产级应该记结构化日志（09.md），这里先简化
        raise HTTPException(status_code=500, detail=f"Agent 调用失败: {e}")

    # ── 3. 回填缓存（TTL 10 分钟 + 随机抖动防雪崩，cache_set 内部已加抖动）──
    # 只缓存 knowledge/faq（知识问答，答案稳定），不缓存 ticket（工单有状态）、general（闲聊无价值）
    if routed in ("knowledge", "faq"):
        await cache_set(
            cache_key,
            {"answer": answer, "routed_agent": routed},
            ttl=600,  # 10 分钟
        )

    return ChatResponse(
        answer=answer, routed_agent=routed, thread_id=req.thread_id  # type: ignore[arg-type]
    )


@app.get("/api/v1/history/{thread_id}", response_model=HistoryResponse)
async def get_history(thread_id: str) -> HistoryResponse:
    """拉取会话历史 —— 02.md 从 LangGraph Checkpointer 读，03.md 可选迁到 PostgreSQL"""
    graph = app.state.graph
    try:
        config: dict = {"configurable": {"thread_id": thread_id}}
        state = await graph.aget_state(config)
        if state is None or not state.values:
            return HistoryResponse(thread_id=thread_id, messages=[])
        msgs: list[HistoryMessage] = []
        for m in state.values.get("messages", []):
            role = "assistant" if m.type == "ai" else "user"
            msgs.append(HistoryMessage(role=role, content=str(m.content)))  # type: ignore[arg-type]
        return HistoryResponse(thread_id=thread_id, messages=msgs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"拉取历史失败: {e}")


# ── 工单端点：03.md 起接 PostgreSQL 真实持久化 ──


@app.post("/api/v1/tickets", response_model=TicketResponse, status_code=201)
async def create_ticket(
    req: TicketCreateRequest,
    db: AsyncSession = Depends(get_db),
    idempotency_key: str | None = Query(default=None, alias="Idempotency-Key"),
) -> TicketResponse:
    """创建工单 —— 04.md 加幂等 key 防重复创建

    客户端传 Idempotency-Key（query 参数形式，正式版应改 Header），同一 key 重复提交只创建一次。
    01.md 学的幂等性在这里落地。
    """
    # 幂等检查（如果客户端传了 key）
    if idempotency_key:
        existing = await idempotent_create(idempotency_key, ttl=3600)
        if existing is not None:
            # 之前处理过：简化版返回 409，05.md 会升级为返回首次完整响应
            raise HTTPException(
                status_code=409,
                detail=f"Ticket already created with key {idempotency_key}: {existing}",
            )

    ticket = Ticket(
        ticket_id=generate_ticket_id(),
        title=req.title,
        description=req.description or "",
        status="open",
        priority="中",
        category="其他",
    )
    db.add(ticket)
    await db.commit()
    await db.refresh(ticket)  # 拿回 DB 生成的 id / created_at

    # 创建成功后，把幂等 key 标记为"已完成"（值为 ticket_id）
    # 简化版：只存 ticket_id，05.md 会存完整响应
    if idempotency_key:
        from .cache import redis_client
        await redis_client.set(idempotency_key, ticket.ticket_id, ex=3600)

    return TicketResponse(
        ticket_id=ticket.ticket_id,
        title=ticket.title,
        status=cast(TicketStatus, ticket.status),
    )


@app.get("/api/v1/tickets/{ticket_id}", response_model=TicketResponse)
async def get_ticket(
    ticket_id: str,
    db: AsyncSession = Depends(get_db),
) -> TicketResponse:
    """查询单个工单（按业务键）—— 走 ticket_id 唯一索引"""
    stmt = select(Ticket).where(Ticket.ticket_id == ticket_id)
    result = await db.execute(stmt)
    ticket = result.scalar_one_or_none()
    if ticket is None:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")
    return TicketResponse(
        ticket_id=ticket.ticket_id,
        title=ticket.title,
        status=cast(TicketStatus, ticket.status),
    )


@app.get("/api/v1/tickets", response_model=TicketListResponse)
async def list_tickets(
    status_filter: Literal["open", "processing", "closed"] | None = Query(
        default=None, alias="status"
    ),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> TicketListResponse:
    """工单列表（状态过滤 + 分页）—— 状态过滤走 status 索引"""
    stmt = select(Ticket)
    count_stmt = select(func.count()).select_from(Ticket)
    if status_filter is not None:
        stmt = stmt.where(Ticket.status == status_filter)
        count_stmt = count_stmt.where(Ticket.status == status_filter)

    stmt = (
        stmt.order_by(Ticket.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )

    # 并行执行查询 + 计数（asyncio.gather 真正发挥价值）
    items_result, total_result = await asyncio.gather(
        db.execute(stmt),
        db.execute(count_stmt),
    )
    tickets = items_result.scalars().all()
    total = total_result.scalar_one()

    return TicketListResponse(
        items=[
            TicketResponse(
                ticket_id=t.ticket_id,
                title=t.title,
                status=cast(TicketStatus, t.status),
            )
            for t in tickets
        ],
        total=total,
    )


@app.patch("/api/v1/tickets/{ticket_id}", response_model=TicketResponse)
async def update_ticket_status(
    ticket_id: str,
    status: Literal["open", "processing", "closed"],
    db: AsyncSession = Depends(get_db),
) -> TicketResponse:
    """更新工单状态 —— UPDATE ... WHERE ticket_id = ? RETURNING ..."""
    stmt = (
        update(Ticket)
        .where(Ticket.ticket_id == ticket_id)
        .values(status=status)
        .returning(Ticket.ticket_id, Ticket.title, Ticket.status)
    )
    result = await db.execute(stmt)
    row = result.one_or_none()
    if row is None:
        await db.rollback()
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")
    await db.commit()
    return TicketResponse(
        ticket_id=row.ticket_id,
        title=row.title,
        status=cast(TicketStatus, row.status),
    )


@app.get("/")
async def root() -> dict[str, str]:
    return {"name": "capstone-enterprise-qa backend", "version": "0.5.0"}


# ── 异步任务端点（05.md 起）──


@app.post(
    "/api/v1/chat/async",
    response_model=ChatTaskResponse,
    status_code=202,
)
async def chat_async(req: ChatRequest) -> ChatTaskResponse:
    """异步对话 —— 投递 Celery 任务立即返回 task_id

    202 Accepted：请求已接受但未完成，符合 HTTP 语义（01.md 学的）。
    前端拿 task_id 轮询 GET /api/v1/chat/tasks/{task_id}。

    保留 /api/v1/chat 同步版用于缓存命中场景（快问快答），
    长时推理走这个异步端点。06.md 起 /chat 会接 SSE 流式返回 token。
    """
    # 投递 Celery 任务
    result = run_agent_task.delay(req.message, req.thread_id)
    return ChatTaskResponse(
        task_id=result.id,
        status="pending",
        message=req.message,
        thread_id=req.thread_id,
    )


@app.get(
    "/api/v1/chat/tasks/{task_id}",
    response_model=TaskStatusResponse,
)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    """轮询任务状态 —— 前端拿 task_id 查进度

    Celery 状态映射：
      PENDING → 还在队列里没被 Worker 取走
      STARTED → Worker 正在执行
      SUCCESS → 完成，result 字段有值
      FAILURE → 失败，error 字段有值
      RETRY   → 重试中（归到 started）
    """
    result: AsyncResult = run_agent_task.AsyncResult(task_id)
    status_map = {
        "PENDING": "pending",
        "STARTED": "started",
        "SUCCESS": "success",
        "FAILURE": "failure",
        "RETRY": "started",
    }
    celery_status = result.status
    mapped = status_map.get(celery_status, "pending")

    return TaskStatusResponse(
        task_id=task_id,
        status=mapped,  # type: ignore[arg-type]
        result=result.result if mapped == "success" else None,
        error=str(result.result) if mapped == "failure" else None,
    )


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("BACKEND_HOST", "0.0.0.0")
    port = int(os.getenv("BACKEND_PORT", "8000"))
    uvicorn.run("src.main:app", host=host, port=port, reload=True)
