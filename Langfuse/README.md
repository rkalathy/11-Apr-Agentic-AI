# Resume Review Agent — Langfuse Setup

A LangChain agent that reviews resumes, with full Langfuse integration:
- Prompts managed in Langfuse (no hardcoded strings)
- Traces auto-captured via the LangChain callback
- Multi-criteria LLM-as-a-judge scores attached to every trace
- Dataset + experiment runner for offline evaluation

## File layout

| File | Role |
|---|---|
| `seed_prompts.py` | Push/update prompts in Langfuse (system, score, strengths, improvements, judge) |
| `seed_dataset.py` | Create the `resume-review-eval` dataset with 4 sample resumes |
| `resume_review_agent.py` | The agent — fetches prompts, runs, attaches scores |
| `run_experiment.py` | Iterate the agent over the dataset and emit per-run evaluations |
| `requirements.txt` | Python dependencies |

## Prerequisites

- Python 3.13 venv (`venv/` in this folder)
- `.env` at the **repo root** (`../.env`) — but our scripts load from local `.env`. Copy or symlink:

```bash
ln -s ../.env .env
```

Required keys:

```
OPENAI_API_KEY=sk-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_BASE_URL=http://10.60.0.111:30300
```

Install deps:

```bash
pip install -r requirements.txt
```

## One-time setup

```bash
python seed_prompts.py     # creates the 5 prompts in Langfuse (production label)
python seed_dataset.py     # creates the resume-review-eval dataset with 4 items
```

Both are idempotent — re-run safely after editing the source files.

## Production usage

Run the agent on a single resume:

```bash
python resume_review_agent.py
```

What happens:

1. Fetches all prompts from Langfuse (`label="production"`)
2. Invokes the agent — it calls 3 tools: `score_resume`, `identify_strengths`, `suggest_improvements`
3. Attaches scores to the resulting trace:

| Score | Type | Source |
|---|---|---|
| `resume_score` | NUMERIC (0–10) | Regex from agent output |
| `tools_completeness` | BOOLEAN | All 3 tools called |
| `judge_relevance` | NUMERIC | LLM judge — addresses real resume content |
| `judge_specificity` | NUMERIC | LLM judge — concrete vs generic |
| `judge_actionability` | NUMERIC | LLM judge — applicable suggestions |
| `judge_coverage` | NUMERIC | LLM judge — score + strengths + improvements |
| `judge_overall` | NUMERIC | Mean of 4 judge criteria |

## Offline evaluation (experiments)

```bash
python run_experiment.py
```

Runs the agent across all 4 dataset items concurrently, evaluates each output, and creates a Langfuse **dataset run** linking every per-item trace.

Evaluators applied:

| Evaluator | Purpose |
|---|---|
| `eval_tools_completeness` | Did the agent use all 3 tools? |
| `eval_resume_score` | Numeric X/10 from output |
| `eval_resume_score_in_range` | Score within bounds set in `expected_output` |
| `eval_judge` | Multi-criteria LLM judge (5 scores) |

The script prints the direct dashboard URL at the end.

## Where to find things in the Langfuse UI

| Thing | Path |
|---|---|
| Production traces | **Tracing → Traces** (filter by score, date, etc.) |
| Prompts | **Prompts** — edit any prompt and the next agent run picks it up |
| Datasets | **Datasets → resume-review-eval** |
| Experiment runs | Inside the dataset → **Runs** tab |
| Compare runs | **Runs** tab → select 2+ → **Compare** button |

## Editing prompts without code changes

1. Open Langfuse → **Prompts** → e.g. `resume-review/system`
2. Create a new version, save it with the `production` label
3. Next agent invocation uses the new version automatically

To roll back: re-label the previous version as `production` in the UI.

## Common workflows

**Tune a prompt:**
1. Edit prompt in Langfuse UI
2. `python run_experiment.py`
3. UI → dataset → Runs → Compare against the previous run

**Add a new test resume:**
1. Add an entry to `ITEMS` in `seed_dataset.py` (give it a stable `id`)
2. `python seed_dataset.py` — upserts only the new item
3. Next experiment run picks it up

**Switch model:**
1. Change `model="gpt-4o-mini"` in `resume_review_agent.py`
2. `python run_experiment.py` and compare in the UI

## Troubleshooting

- **`Import "langfuse" could not be resolved`** — Pylance can't see the venv. Set the Python interpreter to `Langfuse/venv/bin/python` in your IDE; runtime is unaffected.
- **`401 Incorrect API key`** — `OPENAI_API_KEY` in `.env` is invalid; agent never calls a tool.
- **Evaluator failed: `'dict' object has no attribute 'name'`** — evaluators must return `Evaluation` objects (`from langfuse import Evaluation`), not dicts.
- **No scores show up** — check `langfuse.flush()` is called before the script exits.
