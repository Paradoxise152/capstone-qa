"""backend/src/deps.py

FastAPI 依赖注入：把 Agent 实例、配置等共享资源注入路由。
"""

from __future__ import annotations

from fastapi import Request


def get_graph(request: Request):
    """从 app.state 拿 Supervisor 图（lifespan 初始化时挂上去）"""
    return request.app.state.graph
