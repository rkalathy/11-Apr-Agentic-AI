"""RAG agent demo: a FunctionAgent with a vector-index query tool over local docs."""

import asyncio
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv(".env")

Settings.llm = OpenAI(model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

DATA_DIR = Path(__file__).parent / "data"
INDEX_DIR = Path(__file__).parent / "storage"


def build_or_load_index() -> VectorStoreIndex:
    if INDEX_DIR.exists():
        storage = StorageContext.from_defaults(persist_dir=str(INDEX_DIR))
        return load_index_from_storage(storage)

    docs = SimpleDirectoryReader(str(DATA_DIR)).load_data()
    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir=str(INDEX_DIR))
    print(f"[built] index over {len(docs)} docs -> {INDEX_DIR}")
    return index


def make_agent() -> FunctionAgent:
    index = build_or_load_index()
    query_engine = index.as_query_engine(similarity_top_k=3)

    rag_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="nimbustech_kb",
        description=(
            "Answer questions about NimbusTech (the company, its product Nimbus "
            "Observe, pricing, leadership, integrations, and limitations) using "
            "the internal knowledge base."
        ),
    )

    return FunctionAgent(
        tools=[rag_tool],
        llm=Settings.llm,
        system_prompt=(
            "You answer questions about NimbusTech. ALWAYS use the nimbustech_kb "
            "tool first to retrieve grounded context, then answer concisely. "
            "If the knowledge base doesn't contain the answer, say so explicitly."
        ),
    )


async def main():
    agent = make_agent()
    questions = [
        "Who founded NimbusTech and when?",
        "What's the difference between the Pro and Enterprise plans?",
        "Does Nimbus Observe support AWS GovCloud?",
        "What's the company's stock price?",
    ]
    for q in questions:
        print(f"\n>>> {q}")
        response = await agent.run(q)
        print(response)


if __name__ == "__main__":
    asyncio.run(main())
