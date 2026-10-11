# RAG over the LangChain docs

A learning project: build the simplest working RAG pipeline, break it on purpose, then make it production-grade.
Questions are answered from the LangChain, LangGraph, Deep Agents and LangSmith docs with citations, and an eval gate catches regressions.
Decisions, results and gotchas live in [PLAN.md](PLAN.md).

## Pipeline

```
docs (MDX, snippets inlined) → clean + header-aware chunking → EmbeddingGemma 2 (CUDA) → Qdrant
question → [condense follow-up] → dense top 20 → rerank (TypeSafe Jev) → top 4 → grounded answer with [source] citations
```

| Path | What |
|---|---|
| `rag/corpus.py` | Load (oss + LangSmith, snippet imports inlined), clean, chunk |
| `rag/store.py` | Embedder; Qdrant (server via `QDRANT_URL`, else embedded); incremental sync by content-addressed chunk IDs |
| `rag/retrieve.py` | Dense search, optional BM25 hybrid / rewrite (RRF), pluggable rerankers (`RERANK_MODEL`) |
| `rag/generate.py` | Grounded answer prompt, condense for multi-turn, LLM judge |
| `rag/api.py`, `rag/chat.html` | FastAPI service (SSE streaming, citations, 👍/👎 feedback, Langfuse tracing) and chat UI |
| `eval/run.py` | Retrieval metrics + judged answers; `--gate` fails below `eval/thresholds.json` |
| `eval/*.jsonl` | `hard`, `main`, `rare`, `multi`, `langsmith` eval sets |
| `eval/*.py` | Drafting, re-pooling, injection, vague-query, cache and cost probes (see PLAN.md) |

## Setup

Needs Python 3.12+ (developed on 3.14) with [uv](https://docs.astral.sh/uv/), Docker with the NVIDIA runtime,
an NVIDIA GPU (~1.5 GB VRAM for the embedder) and an OpenAI-compatible LLM endpoint.

```bash
uv sync
git submodule update --init --filter=blob:none   # docs repo, pinned commit (sparse checkout: src/oss by default)
git -C data/langchain-docs sparse-checkout add src/langsmith src/snippets   # LangSmith pages + shared snippets
```

Create `.env`:

```bash
EMBED_MODEL=google/embeddinggemma-2
LLM_BASE_URL=http://localhost:.../v1
LLM_API_KEY=...
LLM_MODEL=...
JUDGE_MODEL=...          # optional; defaults to LLM_MODEL. Pin it so swapping the generator doesn't move the grader
RERANK_MODEL=jev:choice  # or jev:noul, ce:<hf id>, an LLM name; unset = generator LLM
JEV_API=...              # TypeSafe key, for jev:* rerankers
QDRANT_URL=http://localhost:6333
LANGFUSE_PUBLIC_KEY=...  # = LANGFUSE_INIT_PROJECT_* in langfuse/.env
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=http://localhost:3000
```

Langfuse needs its own secrets in `langfuse/.env` (every `# CHANGEME` in `langfuse/compose.yml`, plus the
`LANGFUSE_INIT_*` org/project/user values). Generate them, e.g. with `openssl rand -hex 32`.

## Run

```bash
make up            # Qdrant + API on the GPU (http://localhost:8000) + Langfuse (http://localhost:3000)
make eval          # all eval sets with rerank; fails on regression
make down
```

The first start embeds the corpus (~24k chunks, a few minutes). Later syncs embed only changed chunks.
After changing the corpus, re-pool eval labels: `uv run python -m eval.repool hard main rare multi langsmith`.

## Status

Phase 2 Milestones 1–5 are done, and M6 (real-world) is mostly done; feedback → eval waits for real ratings.
Current gate (80 items): hit@4 0.958, MRR 0.952, answers 0.938, time to first token ~1.1 s. See [PLAN.md](PLAN.md#milestones).
