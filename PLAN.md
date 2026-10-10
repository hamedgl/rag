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

---

# Phase 2: production-grade (confirmed 2026-10-10)

| Area | Decision |
|---|---|
| Strategy | Eval first, then quality, then service |
| Users | Single user (no auth) |
| Corpus | Full LangChain docs (~760 pages) |
| Eval set | ~50 LLM-drafted + human-reviewed, plus the 10 handmade as hard core; all relevant chunks labeled |
| Eval gate | `make eval` with score thresholds; a regression fails |
| Retrieval | query rewrite → hybrid (BM25 + dense) → listwise LLM rerank (20 → 5); dedup at ingest |
| Multi-turn | condense follow-up into a standalone question (shares the rewrite step) |
| Cache | semantic cache, gated on a measured wrong-hit rate |
| Vector DB | Qdrant |
| Embedder | in API process, GPU |
| LLM | 9router `kaflama` as-is. KNOWN RISK: Antigravity route appears to inject ~/.gemini/GEMINI.md (Ponytail rules) |
| Service | FastAPI, streaming, citations, chat UI with 👍/👎 |
| Tracing | Langfuse self-hosted |
| Freshness | incremental by file hash; delete vectors of removed pages |
| Deploy | Docker Compose, local, GPU passthrough |

## Milestones
1. **Eval v2**: full corpus, drafted and reviewed eval set, chunk-level metrics, `make eval`
2. **Ingest v2**: dedup, cleaner v2, incremental hashing, Qdrant
3. **Retrieval v2**: rewrite → hybrid → rerank, one at a time with eval after each
4. **Service**: FastAPI, streaming, citations, multi-turn, UI, feedback
5. **Ops**: Compose, GPU containers, Langfuse, semantic cache, latency/cost
6. **Real-world**: prompt injection via docs, vague queries, feedback → eval, gateway contamination

## Milestone 1 log
- M1.1 hard set (10 Q, chunk-level evidence): subset hit@4 0.75 → full 0.50; MRR 0.53 → 0.47; answers 0.7 both (different Qs). Main issue at scale: the same feature documented for LangChain, LangGraph and Deep Agents (→ metadata filter, M3).
- Generator switched to gpt-oss-120b-medium (via `kaflama`, with Gemini as fallback). Judge pinned: JUDGE_MODEL=ag/gemini-3.8-flash-low. Same accuracy 0.7 but decline_acc 1.0 → 0.5: gpt-oss fails by confident wrong answers (h2) and false attribution (h8) instead of refusing. Uses 【】 citations. Gate reset to the gpt-oss baseline.
- M1.2 eval set: 52 drafted (Gemini) → 45 kept after review (dropped code trivia d4 d8 d13 d20 d29, generic d32 d46). eval/main.jsonl + eval/hard.jsonl = 55 items.
- Baseline (gpt-oss, full corpus): hit@4 0.851, MRR 0.741, answers 0.87–0.89, declines 0.875–1.0. Drafted set is far easier than the handmade one (hit@4 0.92 vs 0.50): questions written from a chunk land near that chunk.
- Noise: over 3 identical runs, retrieval is exactly repeatable; only h8 flips (gpt-oss at temperature 0 isn't deterministic). One flip = 12.5 points of decline_acc (8 declines). Gate set below the noise floor: hit@4 0.85 / MRR 0.74 / answers 0.85 / declines 0.85.

## Milestone 2 log
- M2.1 Qdrant (embedded local mode, `data/qdrant`; server mode with Compose in M5). Same chunks + embedder: metrics identical (hit@4 0.851, MRR 0.741, answers 0.873, declines 0.875). But 3/55 top-4 lists differ, and h1 moved from rank 7 to 6 (an MRR change of 0.0005, lost in rounding). All the swaps are near-ties between duplicate content (chat/TEMPLATE vs vectorstores/TEMPLATE, repeated extensions chunks). Cause: exact search (Qdrant local) vs approximate HNSW (Chroma). Averages hide per-item churn.
- M2.2 Incremental sync. Changed the plan: content-addressed chunk IDs instead of file hashes. ID = uuid5(page, chunk text, nth copy on that page). Each sync re-runs clean + chunk on everything (0.5 s, CPU), embeds only new IDs and deletes stale ones. This also catches cleaner/chunker changes, which a file hash would miss. The copy counter keeps 31 within-page duplicate chunks as separate points. No-op sync takes 12 s (full embed: 335 s). Check: `uv run python -m rag.test_store`.
- Gotcha: `make eval` was a no-op ("'eval' is up to date"), because the eval/ directory shadowed the target. Fixed with `.PHONY`.
- M2.3 Cleaner v2. Audit found the cleaner CORRUPTING CODE: `^import` (meant for MDX snippet imports) deleted 1,541 import lines inside code samples, and the JSX-tag regex stripped 546 tags in 140 code blocks (e.g. `Promise<ReadResult>`). It also dropped 1,910 component titles (Card 898, Tab 429, Accordion 424, Step 153; same bug class as ParamField). Fix: split on ``` fences and run tag/import rules on prose only; keep `title=` as **bold** lines; img → alt text; strip HTML tags and JSX comments. 15% of chunks changed (1,705 added, 1,629 deleted); incremental sync took 91 s.
  Result: metrics IDENTICAL, 0 answer flips, only ±1 rank shuffles. Eval is blind to code and tab-title correctness (evidence quotes are prose, and drafted Qs came from clean-looking chunks). Kept anyway: corrupt code is wrong without a metric. Eval gap: add Qs whose answer is in a code import or a tab label (feedback → eval, M6).
- M2.4 Dedup. Measured first: cosine can't separate duplicates from distinct facts (at 0.96, WatsonxRerank vs WatsonxToolkit = different classes, same boilerplate; even 0.995 pairs sit on different pages, so dropping one breaks that page's citation). Changed the plan:
  - At ingest: drop junk only (138 chunks: printed embedding vectors, 70 from pgvectorstore, plus bare ``` leftovers from splitting).
  - At query time: `corpus.unique()` collapses exact-duplicate bodies in results (373 boilerplate copies across templated integration pages).
  - Near-duplicates are left to the M3 reranker.
  NOT EVALUATED (skipped by choice): the first M3 eval's baseline includes this change.
- M2 done: Qdrant + content-addressed sync + cleaner v2 + dedup. Gate still passes (last measured: 0.851 / 0.741 / 0.873 / 0.875).

## Milestone 3 log
- Eval is now parallel: LLM calls run in 8 threads, retrieval stays sequential (embedded Qdrant). 55 Qs: ~6–11 min → 57 s.
- Baseline incl. M2 dedup: hit@4 0.851, MRR 0.741 (unchanged by dedup), answers 0.855 (1 flip = noise).
- M3.1 naive rewrite (replaces the question for retrieval): hit@4 0.766, MRR 0.678. WORSE overall, but mixed:
  - Wins on vague/symptom Qs: h2 None→1, d23 16→10, d42/d44 3→1.
  - Losses on exact Qs: h3 1→3, d19 1→5.
  - Losses from invented classes (HumanInTheLoopAgent, InterruptHandler, GraphNode) and v0.x legacy APIs (AgentExecutor: d12 2→20). The LLM knows old LangChain better than these v1 docs.
  - Output leaks `<|end|>`, `\x00`, `​`.
  Next: fuse original + rewrite with RRF instead of replacing.
- M3.1b Fuse original + rewrite (RRF, k=60): hit@4 0.809, MRR 0.720. Between the two, not better.
  - Equal-weight RRF lets chunks that both lists rank ~8th beat the original's correct #2 (d12 2→8, d22 2→7).
  - KEY DIAGNOSIS: recall@20 is already 46/47 for the original query (hit@4 40/47). The problem is ORDERING, not recall. Rewrite and hybrid are recall tools, so they have little to fix. Reranker ceiling over the pool: 46/47.
  - Rewrite stays off for single-turn (`--rewrite` flag kept); its code returns for multi-turn condensing (M4).
- M3.2 Listwise LLM rerank (gpt-oss, top 20 → 5, rest keep their order). hit@4 0.851 → 0.979 (= the pool ceiling, 46/47), MRR 0.741 → 0.941–0.952, answers 0.855 → 0.945–0.964, declines 1.0.
  - Fixed: h1 6→1, h5 6→1, d23 16→1, d25 7→1. Still open: h2 (not in the pool at all), d17 (rank 4, wrong answer).
  - Retrieval is no longer deterministic (LLM in the loop), but over 3 runs hit@4 stayed at 0.979; only d18 moved (1↔2).
  - Cost: +1 LLM call per query (~20 chunks ≈ 4–5k tokens); eval 57 s → 95 s.
  - The gateway fell back to Gemini for 1 answer per run (more load → more fallbacks?).
  - Rerank is now default in `make eval`. Gate raised: hit@4 0.95 / MRR 0.92 / answers 0.92 / declines 0.85.
  - Remaining gap: recall (h2). Hybrid/rewrite get judged on that one now; the reranker absorbs their ordering noise.
- M3.3 Hybrid (in-memory rank_bm25 over the same chunks, `\w+` tokens, RRF with dense → cut to 20 → rerank). BM25 alone: recall@20 39/47 (dense 46/47), 26 ms/query, index 0.3 s.
  Hybrid + rerank: recall@20 0.979 (same), hit@4 0.957 (one item, d17 4→15), MRR 0.938. Within noise.
  BM25 beats dense on 6 Qs (h7 `stream_mode` 7→1, d25 7→3), but all 6 are already in the dense pool, where the reranker fixes them. Hybrid only pays when it brings evidence INTO the pool: 0 cases here.
  Kept behind `--hybrid`, off by default. Same eval gap as cleaner v2: no rare-token Qs (error strings, env vars, pasted tracebacks), which is where BM25 classically wins. h2 is missed by both (pure vocabulary mismatch).
- M3.4 Rare-token eval set: `eval/draft_rare.py` drafts Qs that MUST contain an exact rare string (error message, ENV_VAR/config key, import line; rarest token in ≤5 chunks), with evidence pooled over dense ∪ BM25 so labels don't favour one method. 15 drafted → 11 kept (dropped r3 circular, r7 r8 r9 trivia) → eval/rare.jsonl, now part of `make eval`.
  - Results: dense hit@4 1.0 / MRR 0.848; hybrid 1.0 / 0.955; with rerank both 1.0 / 1.0. SATURATED.
  - Seed bias again: a rare string copied from a chunk is literally in that chunk, so dense finds it too. The hard case (docs explain the error in different words, or the string lives only in another page's code) can't be drafted from the answer chunk; it needs real user queries (M6).
  - Decision: hybrid stays OFF (no gain in the final pipeline; reranker covers ordering).
- M3 status: rerank is the win. Rewrite and hybrid kept behind flags. Rewrite returns for multi-turn condensing (M4). Next: M4 service.

## Milestone 4 log
- M4.1 Service: `make serve` → FastAPI on :8000 (rag/api.py) + one-file vanilla chat UI (rag/chat.html).
  - POST /chat streams SSE: sources → token… → done (id, condensed query, model, per-stage timings) | error (in-band, since the HTTP status was already sent).
  - POST /feedback appends 👍/👎 + comment + the WHOLE turn to data/feedback.jsonl (ready-made eval items for M6). Every turn is logged to data/turns.jsonl. Both are gitignored.
  - UI escapes LLM output and adds only its own markup (``` → <pre>, [source] → doc link). No markdown-to-HTML: the answer is LLM text derived from third-party docs (XSS vector, see M6).
  - Citation URL = docs.langchain.com/oss/python/ + source without "python/" (verified 4 patterns → 200).
- Latency (single turn): retrieve 0.06 s, rerank 2.6–3.4 s, first token 3.5–6.2 s. Rerank ≈ 75% of time-to-first-token. Streaming is real (328 events over 2.9 s).
- Multi-turn: "And which one should I use in production?" → condensed to "Which checkpoint backend should I use in production?" → PostgresSaver/MongoDBSaver with correct citations. Condense adds ~1 s.
  - Same mechanism as the M3.1 rewrite, opposite outcome: the new vocabulary came from the assistant's previous (docs-grounded) answer, not from the model's stale memory.
- M4.2 Multi-turn eval: eval/multi.jsonl, 8 handmade items (ellipsis, pronoun ×3, ordinal, correction, topic switch, decline). Fixed history (user + assistant text), a gold `standalone` question for the judge, and evidence quotes. In `make eval`; `--no-condense` = baseline.
  | Variant | answers | hit@4 | recall@20 |
  |---|---|---|---|
  | raw follow-up | 2/8 (only topic switch + decline) | 0.29 | 0.43 |
  | condense v1 | 6/8 | 0.57 | 0.57 |
  | condense v2 | 7/8 | 0.86 | 1.0 |
  - v1 failure: references resolved but the TOPIC was dropped ("look things up in the store", "see the text word by word"). "Don't add terms" made it too literal. v2 requires naming the topic from the conversation, and still bans terms from outside it.
  - m4 is still wrong: "word by word" vs the docs' "LLM tokens" is a vocabulary gap that condense correctly won't bridge (same class as h2).
  - CAVEAT: the prompt was tuned on the same 8 items, so 7/8 is optimistic.
  - Full gate (74 items): hit@4 0.985, MRR 0.924, answers 0.959. MRR keeps landing near 0.92: average 2 runs if it flakes.
- Gotcha: `pkill -f "uvicorn rag.api"` kills the calling shell too (the pattern matches its own command line). Use `pkill -f "[u]vicorn rag.api"`.
- M4.3 Reranker benchmark (74 items, 1 run each; latency = sequential single-user probe, 10 hard Qs, warmed up). `--rerank-model <llm>` or `ce:<hf id>`.
  | Reranker | hit@4 | MRR | answers | latency median/max |
  |---|---|---|---|---|
  | none | 0.846 | 0.744 | 0.919 | 0 |
  | gpt-oss-120b (current) | 0.954 | 0.925 | 0.959 | 3.35 / 5.94 s |
  | claude-haiku-4.5 (kr/) | 0.985 | 0.906 | 0.959 | 1.06 / 1.31 s |
  | bge-reranker-v2-m3 (CE, fp32) | 0.938 | 0.824 | 0.932 | 0.52 / 0.61 s |
  | ms-marco-MiniLM-L-6 (CE) | 0.938 | 0.737 | 0.932 | 0.04 / 0.06 s |
  - gemini-3.5-flash-lite: 429 free-tier quota under 8 parallel calls (no result).
  - The no-smaller-gpt-oss premise was wrong: kaflama is already ag/gpt-oss-120b-medium, the only gpt-oss on the proxy.
  - Haiku ≈ gpt-oss quality at 3× speed. Cross-encoders get evidence into the top 4 but not to rank 1: MiniLM MRR = no rerank. A CE scores topical relevance; only an LLM can follow "contains the answer, not just the topic".
  - Gotcha: the first CE call includes the model load (bge mean 1.14 s > p95 0.77 s in the parallel run). Warm up before timing.
- M4.4 TypeSafe Jev reranker (`jev:noul` / `jev:choice`; env JEV_API; POST api.typesafe.ai/v1/systemone). Jev returns typed answers with probabilities, not text, so the "prompt" becomes questions:
  - noul = pointwise: one yes/no per passage, all 20 in one request, sorted by P(yes).
  - choice = listwise: one pick over 20 passages as options; its probability distribution is the ranking.
  Both keep the "actual answer, not just the topic" criterion. Probabilities are 2-decimal, so ties are broken by dense order. NOT deterministic (identical on 1/5 repeats).
  | Reranker | hit@4 | MRR | answers | latency median |
  |---|---|---|---|---|
  | jev:choice (3 runs) | 0.969–0.985 | 0.939–0.940 | 0.946–0.959 | 0.26 s |
  | jev:noul | 0.954 | 0.876 | 0.946 | 0.25 s |
  - jev:choice = best MRR of all rerankers, 13× faster than gpt-oss (3.35 s), ~$0.0003/query (input only).
  - noul < choice: isolated yes/no can't compare passages (h7: two passages tie at 0.95, and dense order picks the wrong one). Same lesson as CE vs LLM: ranking needs comparison.
  - decline_acc 0.889 in all 3 runs: h8 (ingestion-gap label) every time. A sharper reranker surfaces related provider chunks more reliably, and the generator answers from them. Borderline label, but watch it: better rerank → more temptation on "not in docs" Qs.
  - DEFAULT is now jev:choice (`RERANK_MODEL` in .env; unset → generator LLM). Gate passes (hit@4 0.985, MRR 0.932). Service: rerank 0.3 s, time-to-first-token 3.5–6.2 s → 1.1 s. Answer generation (~0.8 s prefill) is now the biggest wait.
