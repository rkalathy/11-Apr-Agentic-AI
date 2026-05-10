"""RAG agent demo: a ChatAgent with a retrieval tool over a tiny in-memory index.

MAF doesn't ship a vector store; we build one from scratch with OpenAI embeddings
+ NumPy cosine similarity, then expose retrieval as a regular tool function.
"""

import asyncio
from pathlib import Path
from typing import Annotated
from dotenv import load_dotenv
import numpy as np
from openai import OpenAI
from pydantic import Field
from agent_framework.openai import OpenAIChatClient

load_dotenv(".env")

DATA_DIR = Path(__file__).parent / "data"
EMBED_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 600  # chars

openai_client = OpenAI()


def chunk_text(text: str, size: int = CHUNK_SIZE) -> list[str]:
    """Pack paragraphs into chunks up to `size` chars; tiny paragraphs get merged."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buffer = ""
    for p in paragraphs:
        if len(p) > size:
            if buffer:
                chunks.append(buffer)
                buffer = ""
            for i in range(0, len(p), size):
                chunks.append(p[i : i + size])
            continue
        if not buffer:
            buffer = p
        elif len(buffer) + 2 + len(p) <= size:
            buffer += "\n\n" + p
        else:
            chunks.append(buffer)
            buffer = p
    if buffer:
        chunks.append(buffer)
    return chunks


def embed(texts: list[str]) -> np.ndarray:
    resp = openai_client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)


def build_index() -> tuple[list[str], np.ndarray]:
    chunks: list[str] = []
    for path in sorted(DATA_DIR.glob("*.txt")):
        chunks.extend(chunk_text(path.read_text()))
    vectors = embed(chunks)
    print(f"[index] {len(chunks)} chunks indexed from {DATA_DIR}")
    return chunks, vectors


CHUNKS, VECTORS = build_index()


def search_knowledge_base(
    query: Annotated[str, Field(description="Natural language search query")],
    top_k: Annotated[int, Field(description="Number of chunks to return", ge=1, le=8)] = 5,
) -> str:
    """Search the NimbusTech knowledge base for grounded context."""
    query_vec = embed([query])[0]
    norms = np.linalg.norm(VECTORS, axis=1) * np.linalg.norm(query_vec)
    sims = (VECTORS @ query_vec) / np.where(norms == 0, 1, norms)
    top_idx = np.argsort(-sims)[:top_k]
    hits = [f"[score={sims[i]:.3f}]\n{CHUNKS[i]}" for i in top_idx]
    return "\n\n---\n\n".join(hits)


async def main():
    agent = OpenAIChatClient(model="gpt-4o-mini").as_agent(
        name="nimbustech_kb_agent",
        instructions=(
            "You answer questions about NimbusTech. ALWAYS call the "
            "search_knowledge_base tool first to retrieve grounded context, then "
            "carefully read the returned chunks and answer concisely. Only say "
            "'I don't know' if NONE of the retrieved chunks contains the answer."
        ),
        tools=[search_knowledge_base],
    )

    questions = [
        "Who founded NimbusTech and when?",
        "What's the difference between the Pro and Enterprise plans?",
        "Does Nimbus Observe support AWS GovCloud?",
        "What's the company's stock price?",
    ]
    for q in questions:
        print(f"\n>>> {q}")
        result = await agent.run(q)
        print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
