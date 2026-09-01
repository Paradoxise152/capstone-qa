# 企业知识库 Agent 系统

基于 LangChain + LangGraph 的多 Agent 协作系统，支持知识库问答、工单管理、FAQ 匹配、人工审批。

## 架构

```
Supervisor（三层路由兜底 + LLM 意图路由）
  ├── Knowledge Agent  — RAG Dense+BM25 双路召回 + Cohere Reranker 精排
  ├── Ticket Agent     — 工单查询/创建/统计
  ├── FAQ Agent        — 6 个高频问题快速匹配
  ├── General Agent    — 闲聊兜底
  └── Human Node       — interrupt() 暂停审批
```

### 三层路由兜底策略

```
用户输入 → ① 硬规则正则（TK-XXXX-XXXX → ticket，100% 可靠）
         → ② LLM 语义判断（覆盖 90% 场景）
         → ③ 低置信度降级（LLM 输出非法值 → general 反问）
```

> 覆盖 8 个公共概念：RAG、MCP、记忆管理、多Agent编排、工作流、HITL、评估工程

## 快速开始

```bash
# 1. Python 3.12 + uv
uv python install 3.12
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env: OPENAI_API_KEY=sk-xxx（必需）
#           COHERE_API_KEY=xxx（可选，用于精排）

# 3. 运行
python -m src.main
```

## 对话示例

```
You: LangGraph 怎么部署？
Agent: 根据知识库，部署步骤：1. pip install langgraph...

You: 查工单 TK-2024-0001
Agent: 工单 TK-2024-0001: 登录页面白屏 | 状态:处理中 | 优先级:高

You: 怎么重置密码？
Agent: 登录页面 → 忘记密码 → 输入注册邮箱 → 设置新密码（约 2 分钟）

You: 转人工
Agent: ⚠️ 需要人工审批 → 已转接人工客服
```

## 评估

```bash
# 一键评估（检索 + 路由，基于 LangChain openevals）
python -m eval.run_all

# 分项评估
python -m eval.run_all --rag      # RAG 检索（相关性 + 有据性 + 正确性）
python -m eval.run_all --route    # Supervisor 路由准确率

# 单元测试（三层路由兜底 + 记忆压缩策略）
python -m pytest tests/ -v
```

评估维度：retrieval_relevance / groundedness / correctness / routing accuracy

### 记忆压缩 Token 节省（10 轮对话实测）

| 策略 | Token 消耗 | 节省比例 | 说明 |
|------|-----------|---------|------|
| 不压缩 | ~2300 | 0% | 全量 messages，对话越长线性增长 |
| 策略 A：Token 窗口截断 | ~2000 | 13% | 保留最新 N 条 + system prompt |
| **策略 C：Buffer+Summary 增量混合** | **~300** | **87%** | **生产推荐**：保留最近 10 条 + 增量摘要 |

策略 C 优势：
- 增量式更新摘要（基于现有摘要 + 新增对话），信息保留更好
- 保留最近 10 条原始消息，短期上下文完整
- Token 节省 87%，长对话不爆窗口

## 项目结构

```
├── src/
│   ├── main.py              # CLI 入口
│   ├── config.py            # 配置
│   ├── supervisor.py        # Supervisor 路由（5 Agent + 三层兜底）
│   ├── routing_rules.py     # 路由硬规则（独立模块，便于测试）
│   ├── memory.py            # 记忆压缩策略（Buffer+Summary 增量混合）
│   ├── hitl.py              # HITL 转人工
│   ├── agents/
│   │   ├── knowledge.py     # RAG 检索 Agent
│   │   ├── ticket.py        # 工单 Agent
│   │   ├── faq.py           # FAQ Agent
│   │   └── general.py       # （内置在 supervisor）
│   ├── rag/
│   │   └── pipeline.py      # RAG 检索管线
│   └── tools/
│       └── ticket_mcp.py    # MCP Server 完整版
├── data/
│   └── knowledge_base.md    # 知识库文档
├── eval/
│   ├── run_all.py           # openevals 一键评估
│   ├── openevals_rag.py     # RAG 三维评估
│   └── openevals_agent.py   # 路由准确率评估
├── tests/
│   ├── test_routing.py      # 三层路由兜底测试（10 用例）
│   └── test_memory.py       # 记忆压缩策略测试（9 用例）
└── requirements.txt
```

## 技术栈

| 层 | 组件 |
|----|------|
| 检索 | LangChain BM25 + Chroma Dense + Cohere Reranker |
| 编排 | LangGraph StateGraph + Supervisor 模式（三层路由兜底） |
| 记忆 | SQLite Checkpointer + Buffer+Summary 增量压缩（Token 节省 87%） |
| HITL  | LangGraph `interrupt()` API |
| 评估 | openevals 三维评估 + pytest 单元测试（19 passed） |
| LLM | OpenAI GPT-4o-mini |