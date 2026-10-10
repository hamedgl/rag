# RAG POC — Plan

Goal: understand core RAG logic by building the simplest working version, then break it on purpose.
Phase 2 (later): production-grade, deployment, real-user issues.

## Decisions

| Area | Choice |
|---|---|
| Embedder | EmbeddingGemma 2 (`google/embeddinggemma-2`, 744M, 768d) via PyTorch + sentence-transformers on CUDA |
| Generator | Gemini 3.8 Flash via local OpenAI-compatible proxy (`LLM_*` in `.env`) → `ChatOpenAI` |
| Corpus | LangChain docs, focused subset (~20–40 pages: concepts + RAG/retrieval), cloned MDX |
| Workspace | Jupyter notebook, `uv` env (Python 3.14) |
| Vector store | `InMemoryVectorStore` → Chroma |
| Chunking | `RecursiveCharacterTextSplitter` (800/100) → `MarkdownHeaderTextSplitter`, compare |
| Retrieval | k=4, then try k=1 / k=15 |
| Prompt | Answer only from context, say "not in docs" otherwise, cite source file |
| Task prefixes | Compare retrieval with and without EmbeddingGemma query/document prefixes |
| Done | 10 known-answer Qs; explain ≥1 retrieval miss + ≥1 hallucination |

## Stages

1. **Setup**: uv env, embedder on GPU, verify LLM endpoint ✅
2. **Ingest**: clone docs, choose subset, clean MDX/JSX noise ✅
3. **Chunk**: naive recursive splitting, inspect the chunks ✅
4. **Embed + store**: in memory, inspect vectors, prefix comparison ✅
5. **Retrieve**: similarity search, read the scores ✅
6. **Generate**: grounded prompt + citations ✅
7. **Evaluate**: 10 Qs, diagnose 2 failures ✅ (retrieval 10/10 hit@4; answers 8/10)
8. **Improve**: header-aware chunking, vary k, move to Chroma ✅

Each stage: concept → user predicts the outcome (multiple choice) → Claude builds and runs → compare.

## Phase 2 topics (later)

Hybrid search (BM25 + vectors), reranking, eval sets and metrics, keeping the index fresh, PDF parsing,
query rewriting, latency and cost, caching, serving, observability, prompt injection through docs,
access control, multi-turn conversation.

## Known gotchas

- Ollama can't run EmbeddingGemma 2 on Linux/NVIDIA: library tags are MLX-only, and ggml GGUF fails with `unknown model architecture: 'gemma-embedding2'` (Ollama 0.40.2). Hence PyTorch.
- HF repo has only the full 744M model (no 270m text-only checkpoint). About 1.4 GB VRAM.
- Needs `pillow` + `torchvision` even for text, because the processor is multimodal.
- `max_seq_length` loads unset (effectively unlimited). Set it to 8192 (model card) explicitly, or long chunks will blow up memory or be handled inconsistently.
- Task prompts are built in: `prompt_name="query"` / `"document"`.
- Similarity of an unrelated pair is about 0.56 (relevant pair about 0.78): cosine scores are relative, not absolute.
- LLM proxy returned a streamed (SSE) response without being asked. Suspect this first if LangChain fails to parse.
- Separator lists can't fix code blocks longer than chunk_size (cut-code stays ~30%). Markdown separators fix mid-sentence starts (25%→14%) but add tiny chunks (14→90). Structure-aware splitting (Stage 8) is the real fix.
- LangChain's markdown separators are regexes: pass `is_separator_regex=True`, otherwise they never match.
- 708 chunk pairs with cosine > 0.97: near-duplicates from provider tabs (same example for OpenAI, Gemini, ...) repeated across pages. They crowd top-k. Fix candidates: dedup at ingest, or keep one provider variant.
- Task prompts A/B on 8 probes: no measurable gain (equal on 5, prompts worse on 2, better on 1). Too few probes to conclude; recheck with Stage 7 eval set.
- "LangChain vs LangGraph" rank 15 was a bad LABEL, not bad retrieval: the answer lives in concepts/products + langchain/philosophy (ranks 3–13). Comparison answers span pages, so one-source labels mislead. The "Feature comparison" heading (rank 10, 151 chars) got split from its table (rank 9).
- Gen k=4 on the comparison question: no refusal; partial but grounded answer (claims verified at page level). k=10 added the "when to use which" guidance from concepts/products. Off-topic (Kubernetes/Helm): clean "Not in docs."
- Gemini 3.x guidance reportedly recommends temperature 1.0 (lower values may cause looping). We use 0 for repeatability; watch for loops or odd output.
- Stage 7 eval: retrieval hit@4 10/10. Answers 8/10. Failures: Q2 over-refusal (right page at rank 1, still said "Not in docs"; likely the chunk says WHAT short-term memory is but not HOW); Q4 lost default (cleaner bug below). No hallucinations: Gemini refused the bait (Q10) and the ingestion gap (Q8) correctly.
- CLEANER BUG: clean_mdx keeps only <ParamField path=...> (27) and drops body=... (270). Parameter names and default="..." attributes are lost. Tool call limit survived only because its description text repeats "(default)". Fix in Stage 8.
- Prompt A/B on eval: prompts helped the paraphrase Q (rank 1 vs 4) and Q10 (1 vs 2), equal elsewhere. Asymmetric prompts matter most when the question's wording differs from the docs'.

## Phase 1 results (2026-10-10)

| Version | Answers (LLM judge) | Change |
|---|---|---|
| v1 | 8/10 | baseline |
| v2 | 9/10 | ParamField fix → Q4 fixed (162 params kept vs 16) |
| v3 | 8/10 | header-aware + heading path → Q2 still refused, Q5 REGRESSED |
| v3, k=1 / 4 / 15 | 6 / 8 / **10** | more context fixed Q2 and Q5 |

- Q2 = chunk-level miss hidden by a page-level hit: top-4 were the wrong sections of the right page (Delete, Customizing, Tools), not the basic "add a checkpointer" usage.
- Loose prompt + temperature 1: made-up answers on both bait questions (invented provider list "from the tutorial"; "4000" plus invented assert code). Strict grounding prompt is what prevented them.
- Loose answer ended with "Skipped: ... Add when ..." (a style not in our prompt) → the proxy may inject its own system prompt. Unverified; check before trusting outputs.
- Chroma scores = cosine DISTANCE (1 - similarity): 0.228 vs 0.772, same ordering.
