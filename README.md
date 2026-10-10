# RAG over the LangChain docs

A learning project: build the simplest working RAG pipeline, break it on purpose, then make it production-grade.
Questions are answered from the LangChain docs with citations, and an eval gate catches regressions.
Decisions, results and gotchas live in [PLAN.md](PLAN.md).

## Pipeline

```
LangChain docs (MDX) → clean + header-aware chunking → EmbeddingGemma 2 (CUDA) → Chroma
question → top-k chunks → grounded LLM answer with [source] citations → LLM judge (eval)
```

| Path | What |
|---|---|
| `rag/corpus.py` | Load, clean (MDX/JSX noise, `ParamField`) and chunk the docs |
| `rag/store.py` | Embedder and persisted Chroma collections (`data/chroma/`) |
| `rag/generate.py` | Grounded answer chain ("Not in docs." when unanswerable) and LLM judge |
| `eval/run.py` | Retrieval metrics (hit@4, MRR@20) + judged answers; `--gate` fails below `eval/thresholds.json` |
| `eval/*.jsonl` | `hard` (10 handmade) and `main` (45 drafted, human-reviewed) eval items |
| `rag.ipynb` | Phase 1 notebook, stage by stage |

## Setup

Needs Python 3.12+ (developed on 3.14) with [uv](https://docs.astral.sh/uv/), an NVIDIA GPU (~1.4 GB VRAM for the embedder) and an OpenAI-compatible LLM endpoint.

```bash
uv sync
git submodule update --init --filter=blob:none   # LangChain docs, pinned commit; read from data/langchain-docs/src/oss
```

Create `.env`:

```bash
EMBED_MODEL=google/embeddinggemma-2
LLM_BASE_URL=http://localhost:.../v1
LLM_API_KEY=...
LLM_MODEL=...
JUDGE_MODEL=...      # optional; defaults to LLM_MODEL. Pin it so swapping the generator doesn't move the grader
```

## Run

```bash
make eval          # full corpus, hard + main sets, fails on regression
make eval-subset   # Phase 1 27-page subset
```

The first run embeds the corpus into `data/chroma/`. If you change the corpus or chunker, use a new collection name or delete `data/chroma/`.

## Status

Phase 1 (notebook POC) is done. Phase 2 Milestone 1 (eval v2) is done: gpt-oss baseline hit@4 0.85, MRR 0.74, answers ~0.87.
Next: Qdrant + incremental ingest, hybrid retrieval + rerank, FastAPI service. See [PLAN.md](PLAN.md#milestones).
