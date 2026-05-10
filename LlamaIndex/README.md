# LlamaIndex Demos

Three small demos showing the core agent patterns in LlamaIndex:

| Demo | File | What it shows |
|---|---|---|
| Basic agent | `basic_agent.py` | Single `FunctionAgent` with two function tools |
| Multi-agent | `multi_agent.py` | `AgentWorkflow` with handoff between Researcher and Writer |
| Agent with RAG | `rag_agent.py` | `FunctionAgent` with a `QueryEngineTool` over a local vector index |

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Uses `OPENAI_API_KEY` from the repo-root `.env` (loaded via `load_dotenv("../.env")`).

## Run

```bash
python basic_agent.py     # tool-using single agent
python multi_agent.py     # researcher → writer handoff
python rag_agent.py       # answers questions using docs in ./data
```

The RAG demo builds a vector index from `data/*.txt` on first run and persists it under `./storage/`. Delete that folder to rebuild from scratch (e.g. after editing the docs).

## Notes

- All agents are async — use `asyncio.run(main())` to invoke
- `FunctionAgent` is the OpenAI-style tool-calling agent; switch to `ReActAgent` if you want explicit reasoning traces
- In the multi-agent demo, `can_handoff_to=[...]` controls who each agent can pass control to; the orchestrator (`AgentWorkflow`) handles the routing
- The RAG demo uses `text-embedding-3-small` for embeddings; change in `Settings.embed_model` to swap models
