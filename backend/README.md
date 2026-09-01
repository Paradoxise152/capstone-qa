# Capstone Backend —— 企业知识库 Agent 的 Web 服务层

基于 **FastAPI** 构建的异步 Web 服务层，将 CLI 版多 Agent 系统封装为 RESTful API，并逐步引入 Redis 缓存、PostgreSQL 持久化与 Celery 异步任务等生产级能力。

## ✨ 当前能力

- **对话 API**：`POST /api/v1/chat` 调用 Supervisor 多 Agent 系统，返回答案与路由信息
- **Redis Cache-Aside 缓存**：热点知识问答缓存 + TTL 随机抖动防雪崩，节省 LLM 调用与 Token
- **幂等创建**：工单创建支持 `Idempotency-Key`，防止重复提交
- **会话历史**：基于 LangGraph Checkpointer 读取多轮对话上下文
- **异步任务**：Celery 异步化长时 Agent 推理（`task_id` 轮询模式）
- **健康检查**：`GET /health`（K8s Probe 预留）

## 🛠️ 技术栈

| 层 | 技术选型 |
|----|---------|
| Web 框架 | FastAPI + Uvicorn（ASGI 异步） |
| 数据库 | SQLAlchemy 2.0（async）+ asyncpg + Alembic |
| 缓存 | Redis（Cache-Aside + TTL 抖动） |
| 任务队列 | Celery + Redis broker |
| Agent 逻辑 | LangGraph（经 `agent_bridge` 复用 CLI 版 `src/`） |

## 📡 API 概览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/api/v1/chat` | 对话（热点缓存 + 调 Agent） |
| GET | `/api/v1/history/{thread_id}` | 会话历史 |
| POST | `/api/v1/tickets` | 创建工单（幂等） |
| GET | `/api/v1/tickets` | 工单列表 |
| POST | `/api/v1/chat/async` | 异步对话（Celery，返回 `task_id`） |
| GET | `/api/v1/chat/tasks/{task_id}` | 异步任务状态轮询 |

> 注：`async` 系列接口随异步任务能力逐步完善。

## 🚀 快速开始

```bash
cd backend
# Windows：py -3.12 -m venv .venv && .venv\Scripts\activate
# macOS / Linux：python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # 填入 OPENAI_API_KEY 等

# 开发模式（热重载）
uvicorn src.main:app --reload
```

## 📁 目录结构

```
backend/
├── src/
│   ├── main.py          # FastAPI 入口与路由
│   ├── agent_bridge.py  # 桥接 CLI 版 LangGraph Agent
│   ├── cache.py         # Redis 缓存（Cache-Aside + 幂等）
│   ├── database.py      # SQLAlchemy 异步引擎
│   ├── models.py        # ORM 模型（Ticket 等）
│   ├── tasks.py         # Celery 异步任务
│   └── deps.py          # 依赖注入
├── requirements.txt
└── .env.example
```

## 🔗 与 CLI 版关系

本服务层复用 CLI 版（`src/`）的多 Agent 系统与 RAG 管线，通过 `agent_bridge.py` 以异步方式调用 Supervisor 图，实现"CLI 能力 → Web API"的封装升级。

