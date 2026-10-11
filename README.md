# RAG over the LangChain docs

A learning project: build the simplest working RAG pipeline, break it on purpose, then make it production-grade.
Questions are answered from the LangChain (v1 and v0.3), LangGraph, Deep Agents and LangSmith docs, the LangSmith API reference and two papers (RAG survey, ReAct), with citations, and an eval gate catches regressions.
Decisions, results and gotchas live in [PLAN.md](PLAN.md).

## Pipeline

```
docs (MDX, snippets inlined) → clean + header-aware chunking → EmbeddingGemma 2 (CUDA) → Qdrant
question → [condense follow-up: gpt-oss] → route version (Jev) → dense top 20 → rerank (TypeSafe Jev) → top 4 → grounded answer (GPT-6 Luna) with [source] citations
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
# v0.3 docs (second docs version; mostly notebooks), pinned:
git clone --filter=blob:none --sparse -b v0.3 https://github.com/langchain-ai/langchain.git data/langchain-v03
git -C data/langchain-v03 checkout 9971fd8caafcfddddc94492cdecb02ce67b82da6
git -C data/langchain-v03 sparse-checkout set docs/docs/concepts docs/docs/how_to docs/docs/tutorials
# papers (RAG survey, ReAct)
mkdir -p data/pdfs && for id in 2312.10997 2210.03629; do curl -sL -o data/pdfs/$id.pdf https://arxiv.org/pdf/$id; done
```

Create `.env`:

```bash
EMBED_MODEL=google/embeddinggemma-2
LLM_BASE_URL=http://localhost:.../v1
LLM_API_KEY=...
LLM_MODEL=...            # generator; current: cx/gpt-6-luna (cheapest with no measured loss, see PLAN.md)
JUDGE_MODEL=...          # optional; defaults to LLM_MODEL. Pin it so swapping the generator doesn't move the grader
CONDENSE_MODEL=...       # optional; defaults to LLM_MODEL. Rewrites follow-ups into the search query, so it moves retrieval; current: kaflama
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

Phases 2 and 3 are closed (2026-10-11). Open items and deliberate gaps are listed at the end of PLAN.md.
Current gate (98 items): hit@4 0.955, MRR 0.926, answers 0.908, time to first token ~1.3 s. See [PLAN.md](PLAN.md#milestones).
