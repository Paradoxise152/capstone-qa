"""
企业知识库 Agent — CLI 入口

Step 1：Supervisor 路由 + Knowledge Agent（RAG 检索）
"""
import sys
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from .config import CHUNK_SIZE, CHUNK_OVERLAP, KNOWLEDGE_FILE, LLM_MODEL, OPENAI_API_KEY
from .rag.pipeline import build_retrieval_pipeline
from .agents.knowledge import build_knowledge_agent
from .supervisor import build_supervisor


def load_knowledge():
    loader = TextLoader(KNOWLEDGE_FILE)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(docs)


def main():
    print("=" * 50)
    print("  企业知识库 Agent — Step 1")
    print("  Supervisor + Knowledge Agent")
    print("=" * 50)

    # 1. 加载知识库 + 构建检索管线
    print("📚 加载知识库...")
    chunks = load_knowledge()
    print(f"   共 {len(chunks)} 个 chunks")
    retriever = build_retrieval_pipeline(chunks)
    print("   ✅ RAG 检索管线就绪")

    # 2. 创建 Agent
    llm = ChatOpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY)
    knowledge_agent = build_knowledge_agent(retriever)

    from .agents.ticket import build_ticket_agent
    ticket_agent = build_ticket_agent()

    from .agents.faq import build_faq_agent
    faq_agent = build_faq_agent()

    general_agent = create_agent(
        llm, tools=[],
        system_prompt="你是通用助手。处理问候、闲聊和无法归类的问题。",
    )
    graph = build_supervisor(knowledge_agent, ticket_agent, faq_agent, general_agent)
    print("   ✅ Supervisor 就绪（4 Agent）\n")

    # 3. 交互式对话
    config = {"configurable": {"thread_id": "session-001"}}  # type: ignore[arg-type]
    print("💬 输入问题开始对话（输入 /quit 退出）：\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("/quit", "/exit", ""):
            print("👋 再见！")
            break

        result = graph.invoke(  # type: ignore[arg-type]
            {"messages": [{"role": "user", "content": user_input}]},
            config,
        )
        answer = result["messages"][-1].content
        print(f"Agent: {answer}\n")


if __name__ == "__main__":
    main()