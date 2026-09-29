# rag/

The RAG engine. Pure Python, no HTTP, no FastAPI, no CLI — this package doesn't
know any of those exist. `api/` and `cli.py` call into it through exactly one
door, `engine.py`, which is why either of them can be deleted without this
folder noticing.

Two things happen here: **ingest** turns a source into searchable chunks, and
**answer** turns a question into a grounded, cited response. Both are one method
call on `RagEngine` (`engine.ingest()`, `engine.ask()` — which routes to
`engine.answer()` / `engine.stream_answer()` or `engine.summarize()` — plus
directly calling any of those three) that walks through every module below in
order.

## Flow 1 — ingest

```
SourceInput
  file_bytes  ─────────────────────────────┐
  arxiv_id    ──▶ ingestion/fetchers/arxiv.py     │
  wikipedia   ──▶ ingestion/fetchers/wikipedia.py │
                          │                        │
                          ▼                        ▼
                 ingestion/resolver.py  (picks the fetcher, then the adapter)
                          │
            ┌─────────────┴──────────────┐
            ▼                             ▼
  ingestion/sources/pdf_source.py   ingestion/sources/wikipedia_source.py
  Docling layout analysis,          "== Heading ==" parsing,
  reading order, block labels       no layout step needed
            │                             │
            └──────────────┬──────────────┘
                            ▼
                     models.ParsedDocument
              (title, blocks[] in reading order, source_type)
                            │
                            ▼
                chunking/section_aware.py
        groups blocks by section, packs to target size,
           carries overlap, drops references/notes
                            │
                            ▼
                        models.Chunk[]
         (chunk text + metadata: section, page, ids)
                            │
                            ▼
                  embedding/bge_m3.py
           one model → dense vector + sparse weights
                            │
                            ▼
               database/qdrant_store.py
        upsert into Qdrant — one collection, two named
           vectors per point (`dense`, `sparse`)
```

Walk it in code: `engine.ingest()` calls, in order,
`resolver.resolve()` → `chunker.chunk()` → `embedder.embed()` →
`store.delete_document()` + `store.upsert_chunks()`. The delete-then-upsert pair
is what makes re-ingesting the same source replace it instead of duplicating it —
`models.document_id_for()` derives a stable ID from the source's origin (arXiv
URL, Wikipedia URL, or `upload:<filename>`), so the same source always maps to
the same document.

An uploaded PDF skips the fetcher step entirely and enters the exact same
`pdf_source.py` used for a downloaded arXiv paper — see `ingestion/base.py`'s
`SourceInput`. Nothing downstream of `resolve()` can tell the two apart.

## Flow 2 — answer

```
question, document_id?
        │
        ▼
retrieval/hybrid.py                              stage 1 — recall
  embed the query (embedding/bge_m3.py)
  search_dense + search_sparse (database/qdrant_store.py)
  fuse the two rankings with RRF (retrieval/rrf.py)
        │
        ▼
   RetrievedChunk[]   (~50 candidates, config: retrieval.candidates)
        │
        ▼
reranking/cross_encoder.py                       stage 2 — precision
  score every (query, chunk) pair together
  sort by rerank_score, keep the top N
        │
        ▼
engine._select_context()
  best rerank_score < retrieval.score_threshold ?
        │
   ┌────┴─────┐
  yes          no
   │            │
   ▼            ▼
ABSTAIN     top N chunks, rank order ──▶ Answer.used_chunks (debug UI reads this)
 (LLM never       │
  called)          ▼
             generation/prompts.py
               order_for_context()  → lost-in-the-middle placement
               build_context()      → "[1] ... [2] ..." + Citation[] to match
                    │
                    ▼
             generation/llm_client.py
               llama.cpp /chat/completions — complete() or stream()
                    │
                    ▼
             response contains "NOT_IN_SOURCE" ?
                │              │
               yes             no
                │              │
                ▼              ▼
            ABSTAIN      Answer(text, citations, used_chunks)
                          citations = only the [n] markers actually
                          used in the text (engine._used_citations)
```

Walk it in code: `engine._select_context()` is shared by both `answer()` and
`stream_answer()` — it's the one place retrieval, reranking, and the abstain
threshold live. From there the two methods diverge only in how they call the
LLM: `answer()` waits for `llm.complete()`; `stream_answer()` iterates
`llm.stream()` and yields an `AnswerEvent` per token, holding back the first few
characters so a streamed `NOT_IN_SOURCE` never leaks onto the screen as a
half-written sentence before the engine recognizes it and abstains instead.

**Why `used_chunks` and the LLM's context aren't in the same order:** rank order
(best score first) is what a human debugging retrieval wants to see. The LLM
gets a different order — `order_for_context()` moves the best chunk to the
front *and* the second-best to the very end, because models are measurably worse
at using information buried in the middle of a long context ("lost in the
middle", see `rag.md` §7.5). Citation markers `[1][2][3]…` follow whatever order
the LLM actually saw, since that's the order it's citing against.

## Flow 3 — summarize

A global question ("what is this paper about") can't be answered from a
handful of retrieved chunks, so `engine.summarize()` skips retrieval entirely
and walks every chunk of the document instead:

```
document_id, question?
        │
        ▼
store.list_chunks()                              every chunk, in document order
        │
        ▼
summarization/map_reduce.py
  group_sections()            chunks → SectionGroup[], one per section
  MapReduceSummarizer._map()  one LLM call per group → SectionSummary[]
  MapReduceSummarizer.reduce()   the summaries → one cited answer
        │
        ▼
Answer(text, citations, section_summaries, verified_claims)
  citations_for()   [n] → the section it names (citations_for/passages_for)
  section_summaries  what each marker points at, for the UI and verification
  verified_claims    checked against each section's real source text, not
                      the summary that could have invented the claim
```

Walk it in code: `engine.summarize()` calls `store.list_chunks()` →
`summarizer.summarize()` → `citations_for()` → `self._verify_summary()`. It
never touches `retrieval/` or `reranking/` — `used_chunks` stays empty here,
since no retrieval ran (see `models.py`'s `Answer.section_summaries`).

## The data model (`models.py`)

The four types every stage passes to the next. Nothing here mentions Qdrant,
FastAPI, or llama.cpp — that's what keeps this package swappable underneath.

```
ParsedDocument            Chunk                    RetrievedChunk             Answer
──────────────            ─────                    ──────────────             ──────
title                      chunk_id                 chunk                      text
source_type    chunking──▶ text        retrieval──▶ score        generation──▶ citations
origin                     document_id              rerank_score               abstained
blocks[]                   metadata                 effective_score            used_chunks
 ├ text                     ├ section               (rerank_score if set,
 ├ block_type               ├ page                   else score)
 ├ section_path             └ section_path
 └ page
```

`document_id_for()` / `chunk_id_for()` (both in `models.py`) derive UUIDs
deterministically from the source's origin, not randomly — that's the whole
mechanism behind "re-ingesting replaces instead of duplicates."

## Every module

| Path | Role |
|---|---|
| `engine.py` | `RagEngine` — the only entry point (`ingest`, `ask`, `answer`, `stream_answer`, `summarize`, `health`) |
| `models.py` | Shared types: `ParsedDocument`, `Chunk`, `RetrievedChunk`, `Answer`, `Citation`... |
| `gpu.py` | `release_cuda_memory()` — drops a model's VRAM so the embedder/reranker/LLM can share one 6 GB card, one at a time (`engine.py`'s `_free_gpu_for_llm`) |
| `ingestion/base.py` | `SourceInput` — enforces "exactly one source" |
| `ingestion/resolver.py` | Picks fetcher + adapter for a `SourceInput`, returns a `ParsedDocument` |
| `ingestion/fetchers/arxiv.py` | arXiv ID → downloaded PDF, via the official `arxiv` package |
| `ingestion/fetchers/wikipedia.py` | Title/URL → clean article text, via the Wikipedia REST API |
| `ingestion/sources/pdf_source.py` | PDF → `ParsedDocument`, via Docling layout analysis |
| `ingestion/sources/wikipedia_source.py` | Wikipedia text → `ParsedDocument`, via heading parsing |
| `chunking/section_aware.py` | `ParsedDocument` → `Chunk[]`, section-bounded with overlap |
| `embedding/bge_m3.py` | Text → dense + sparse vectors (bge-m3) |
| `retrieval/hybrid.py` | Query → candidate `RetrievedChunk[]` (dense / sparse / hybrid) |
| `retrieval/rrf.py` | Reciprocal Rank Fusion — merges two rankings into one |
| `reranking/cross_encoder.py` | Candidates → top-N, scored by a cross-encoder |
| `generation/prompts.py` | System prompt, context ordering, citation numbering |
| `generation/llm_client.py` | Thin HTTP client for an OpenAI-compatible API (Ollama); warns when a prompt fills the context window |
| `generation/vision.py` | Question-conditioned figure descriptions - asks the vision model the user's actual question, at answer time, instead of one fixed caption written at ingest. Off by default (`vision`/`ingestion.figures` in `config.yaml`) |
| `query/router.py` | Keyword routing between retrieval and map-reduce/RAPTOR summarization - a fixed English trigger-phrase match, not an LLM call |
| `verification/` | Sentence-level claim decomposition + one batched LLM-judge call per answer; flags unsupported and uncited claims |
| `summarization/` | Map-reduce, and `summarization.method: raptor` - reduce the document's RAPTOR tree instead of mapping every section |
| `raptor/` | RAPTOR summary tree per document: clustering (no LLM) and bottom-up tree building; nodes stored as chunks with a `level`, hidden from retrieval by default |
| `agents/` | Corrective (LLM-graded retrieval, rewrite and retry) and multi-hop (self-ask follow-up searches) context selection, `agent.mode` |
| `evaluation/` | Multi-paper golden sets (single / comparison / multi-hop / abstain), self-implemented metrics (recall@k, MRR, evidence recall, abstain accuracy), LLM-judged faithfulness and accuracy, per-kind reports |

Every module above is built and covered by tests - nothing in `rag/` is a
stub. One thing deliberately *not* built: a general query-rewriting layer
under `query/` (multi-query, HyDE, step-back, decomposition). The one case
that actually needed it - the multi-hop agent's follow-up search failing to
find an unstated bridge fact - got a narrower, purpose-built version
directly in `agents/multi_hop.py` instead of a general layer here
(`docs/roadmap.md` has the full build history and what was measured at each
step, if that level of detail is ever needed).

## Design rules

- **One-directional dependencies.** `ingestion/` doesn't know `chunking/`
  exists; `chunking/` doesn't know `embedding/` exists. Everything talks through
  `models.py` types and gets composed by `engine.py`. Change an implementation
  (a different reranker, say) and only `core/container.py` needs to know.
- **`rag/` never imports from `api/` or `cli.py`.** Check this with
  `grep -r "import api" rag/` — it should return nothing, always.
- **Fetching and parsing are separate** (`ingestion/fetchers/` vs.
  `ingestion/sources/`) so an uploaded file and a downloaded one run through
  identical parsing code.
- **Two-stage retrieval on purpose.** Stage 1 (`retrieval/`) is cheap and wide —
  it optimizes for recall. Stage 2 (`reranking/`) is expensive and narrow — it
  optimizes for precision. Doing reranking on the full collection instead of ~50
  candidates would work, but far too slowly to be interactive.
- **The abstain threshold is the cheapest hallucination guard.** It runs before
  the LLM is ever called — no tokens, no latency, no chance for the model to
  guess. Everything after it (grounding prompt, `NOT_IN_SOURCE` sentinel) is a
  second layer in case the first one lets through a weak-but-technically-above-threshold
  match.

## Testing

`tests/unit/` drives `RagEngine` with fakes (`FakeRetriever`, `FakeReranker`,
`FakeLLM` in `test_engine.py`) instead of real models — the questions under test
are behavioral ("does it abstain before calling the LLM", "does context land in
lost-in-the-middle order"), not "does bge-m3 produce good embeddings". Only
`database/`'s tests hit a real service (Qdrant); everything in `rag/` is fast
enough to run on every save.
