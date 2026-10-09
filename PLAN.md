# RAG POC — Plan

Goal: understand core RAG logic by building the simplest working version, then break it on purpose.
Phase 2 (later): production-grade, deployment, real-user issues.

## Decisions

| Area | Choice |
|---|---|
| Embedder | EmbeddingGemma 2 `270m` (text) via Ollama (needs Ollama ≥ 0.40) |
| Generator | Gemini 3.8 Flash via local OpenAI-compatible proxy (`LLM_*` in `.env`) → `ChatOpenAI` |
| Corpus | LangChain docs, focused subset (~20–40 pages: concepts + RAG/retrieval), cloned MDX |
| Workspace | Jupyter notebook, `uv` env |
| Vector store | `InMemoryVectorStore` → Chroma |
| Chunking | `RecursiveCharacterTextSplitter` (800/100) → `MarkdownHeaderTextSplitter`, compare |
| Retrieval | k=4, then try k=1 / k=15 |
| Prompt | Answer only from context, say "not in docs" otherwise, cite source file |
| Task prefixes | Compare retrieval with and without EmbeddingGemma query/document prefixes |
| Done | 10 known-answer Qs; explain ≥1 retrieval miss + ≥1 hallucination |

## Stages

1. **Setup**: uv env, Ollama upgrade, pull embedder, verify LLM endpoint
2. **Ingest**: clone docs, choose subset, clean MDX/JSX noise
3. **Chunk**: naive recursive splitting, inspect the chunks
4. **Embed + store**: in memory, inspect vectors, prefix comparison
5. **Retrieve**: similarity search, read the scores
6. **Generate**: grounded prompt + citations
7. **Evaluate**: 10 Qs, diagnose 2 failures
8. **Improve**: header-aware chunking, vary k, move to Chroma

Each stage: concept (what / why / how it fails) → you write the core logic → inspect outputs.

## Phase 2 topics (later)

Hybrid search (BM25 + vectors), reranking, eval sets and metrics, keeping the index fresh, PDF parsing,
query rewriting, latency and cost, caching, serving, observability, prompt injection through docs,
access control, multi-turn conversation.

## Known gotchas

- Ollama 0.30.8 too old for EmbeddingGemma 2. Upgrade first.
- Embedder context 8,192 tokens (model card; Ollama page says 256K). Text past the limit is cut off silently.
- LLM proxy returned a streamed (SSE) response without being asked. Suspect this first if LangChain fails to parse.
