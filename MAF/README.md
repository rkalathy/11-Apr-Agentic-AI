# Microsoft Agent Framework (MAF) Demos

Three small demos showing the core agent patterns in MAF — Microsoft's unified successor to AutoGen + Semantic Kernel.

| Demo | File | What it shows |
|---|---|---|
| Basic agent | `basic_agent.py` | Single `ChatAgent` with two function tools |
| Multi-agent | `multi_agent.py` | `SequentialBuilder` workflow: Researcher → Writer |
| Agent with RAG | `rag_agent.py` | `ChatAgent` with a custom embedding-based retrieval tool |

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Uses `OPENAI_API_KEY` from the repo-root `.env` (loaded via `load_dotenv("../.env")`).

> MAF also supports Azure OpenAI, Azure AI Agents, and Foundry. Swap `OpenAIChatClient` for `AzureOpenAIChatClient` (in `agent_framework.azure`) and provide the Azure env vars.

## Run

```bash
python basic_agent.py     # tool-using single agent
python multi_agent.py     # researcher → writer sequential workflow
python rag_agent.py       # answers questions using docs in ./data
```

## Notes

- All agents are async — invoke with `asyncio.run(main())`
- Tools are plain Python functions; MAF inspects their signatures and `Annotated[..., Field(description=...)]` to build the JSON schema sent to the model
- `SequentialBuilder` runs participants in order with the prior agent's output passed as input to the next. For more complex routing use `HandoffBuilder`, `ConcurrentBuilder`, or `WorkflowBuilder`
- MAF has no built-in vector store; the RAG demo rolls its own with OpenAI embeddings + NumPy cosine similarity. For production swap in Azure AI Search, pgvector, Chroma, etc.
- Built-in OpenTelemetry observability — set `OTEL_EXPORTER_OTLP_ENDPOINT` to send traces to your collector (Jaeger, Langfuse OTLP endpoint, etc.)

## Comparison to LlamaIndex (`../LlamaIndex`)

| Concept | LlamaIndex | MAF |
|---|---|---|
| Single agent | `FunctionAgent` | `ChatAgent` |
| LLM client | `OpenAI(...)` from `llama_index.llms.openai` | `OpenAIChatClient(...)` from `agent_framework.openai` |
| Tools | `FunctionTool.from_defaults(fn)` | Plain function in `tools=[...]` |
| Multi-agent | `AgentWorkflow` + `can_handoff_to` | `SequentialBuilder` / `HandoffBuilder` / `WorkflowBuilder` |
| RAG | First-class — `VectorStoreIndex` + `QueryEngineTool` | Build it yourself; MAF is orchestration-only |
| Async | Yes, `await agent.run(q)` | Yes, `await agent.run(q)` |
