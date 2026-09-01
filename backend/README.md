# capstone-enterprise-qa / backend —— 后端外壳（服务端方向 L2 咬合）

> 本子目录是「服务端工程与系统设计」方向的 L2 累进项目，给上层 CLI 版 Agent 系统加"后端外壳"。
> 学完 10 个服务端概念后，CLI 版 `capstone-enterprise-qa` 将升级为**全栈 AI 客服系统**。

## 与 CLI 版的关系

```
capstone-enterprise-qa/
├── src/                # CLI 版 Agent 系统（AI 主方向产物，保留不动）
├── data/               # 知识库与测试数据（共享）
├── eval/               # openevals 评估（共享）
└── backend/            # ← 本目录：服务端方向后端外壳
    ├── README.md       # 本文件
    ├── requirements.txt
    ├── .env.example
    └── src/
        ├── main.py     # FastAPI 入口（随文章累进）
        └── ...
```

- CLI 入口（`src/main.py`）保留，AI 主方向调试用
- 后端入口（`backend/src/main.py`）随服务端方向文章累进，最终成为 Web 服务入口

## 学习累进路线（10 个服务端概念 → 10 层后端外壳）

| # | 服务端概念 | 给 Capstone 加的后端外壳层 | 对应文章 |
|---|-----------|---------------------------|---------|
| 1 | HTTP 协议与 API 设计 | 设计 Agent 服务的 RESTful API 契约 | `01.md` |
| 2 | FastAPI + asyncio | 把 Supervisor 路由包装成异步 Web 服务 | `02.md` |
| 3 | PostgreSQL + SQLAlchemy | 工单数据持久化到 PostgreSQL | `03.md` |
| 4 | Redis + MongoDB | 缓存热点检索结果、会话短期记忆 | `04.md` |
| 5 | Celery + 消息队列 | 多步 Agent 推理任务异步化 | `05.md` |
| 6 | SSE / WebSocket | 流式返回 Agent 推理过程 | `06.md` |
| 7 | JWT / OAuth2 / RBAC | API 鉴权，区分用户/客服/管理员 | `07.md` |
| 8 | 网关 / 限流 / 熔断 | 限流防 LLM API 被刷爆、熔断降级 | `08.md` |
| 9 | 可观测性 + 性能调优 | Prometheus 监控 P99、Token 消耗 | `09.md` |
| 10 | Docker / K8s / CI/CD | 容器化全栈、CI 跑评估、金丝雀发布 | `10.md` |

## 运行（当前仅骨架，01.md 后可跑）

```bash
cd backend
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env  # 填入 OPENAI_API_KEY 等
python src/main.py
# 或：uvicorn src.main:app --reload
```
