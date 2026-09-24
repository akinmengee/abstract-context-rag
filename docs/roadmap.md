# Roadmap

Each phase adds one capability and states how its effect is measured. Phase order
can change; the folder layout never encodes it.

Status: ✅ done · 🚧 in progress · ⬜ not started

| # | Phase | What it adds | How it is measured | Status |
|---|---|---|---|---|
| 1 | Baseline | Docling parsing, section-aware chunking, bge-m3, Qdrant, grounded answers from Ollama | Parse QC report, first golden set answers | ✅ |
| 2 | Retrieval depth | Hybrid search, RRF, cross-encoder reranking, chunking variants | Ablation table: dense vs sparse vs hybrid vs hybrid+rerank (recall@k, MRR, abstain accuracy) | ✅ |
| 3 | Query and verification | Query routing, map-reduce summaries, citation verification | Faithfulness before/after verification; abstain accuracy | 🚧 |
| 4 | Agentic RAG | Corrective / self-reflective retrieval, multi-paper questions | Multi-hop question accuracy | ⬜ |
| 5 | Fine-tuning | Embedding → reranker → generator | Retrieval metrics against the base models | ⬜ |
| 6 | RAPTOR / GraphRAG | Hierarchical and graph indexes | Global-question accuracy vs map-reduce | ⬜ |
| 7 | Multimodal | Tables and figures, ColPali | Accuracy on table and number questions | ⬜ |
| 8 | Wikipedia | Single-article ingestion through the Wikipedia adapter | Same golden-set metrics on a second domain | ⬜ |
| 9 | Interfaces | React web app, Flutter mobile app, Docker packaging | Runs from one command; usable from a phone | ⬜ |
| 10 | Deployment (optional) | Auth, rate limiting, HTTPS | — | ⬜ |

## Phase 1 status

Done:
- Source adapters and fetchers: arXiv ID, Wikipedia title/URL, PDF upload — one `ingest()` entry point
- Section-aware chunking with stable chunk IDs and section breadcrumbs in the embedding text
- bge-m3 dense + sparse embeddings, stored in one Qdrant collection
- Hybrid retrieval with RRF, cross-encoder reranking, abstain threshold
- Grounded generation with numbered citations and lost-in-the-middle context ordering
- FastAPI (`/api/v1`), SSE streaming, `abstractrag` CLI, Gradio dev console
- Parse QC report (block types, section outline, reading-order warnings)
- **First real end-to-end run** (2026-09-20): ingested the RAG paper (2005.11401),
  asked two real questions, got correct grounded answers with accurate citations
  (verified against the paper). ~70s per question (see `rag.md` §3 for the
  CUDA/Ollama setup issues this took to get working: CPU-only torch by default,
  `localhost` vs `127.0.0.1` on Windows, Qwen3 thinking-mode token budget).
- **Evaluation harness built and run** (2026-09-20): `rag/evaluation/` (models,
  self-implemented recall@k/MRR/abstain-accuracy metrics, LLM-as-judge
  faithfulness, runner) and `abstractrag eval`. First golden set: 11 questions
  on the RAG paper (8 answerable + 3 out-of-scope). First measured result:
  recall@5 0.88, MRR 0.81, abstain accuracy 0.82 (`mode=hybrid`, judge off).
  Two real findings from that run, both fixed or logged rather than guessed at:
  - `retrieval.score_threshold` was rejecting a correctly-retrieved answer
    (rank 1, score 0.16) while every wrong-abstain case scored ≤ 0.09 - lowered
    0.3 → 0.12. Narrow margin on 11 samples; revisit as the golden set grows.
  - The reranker scores the "3.4 Fact Verification" chunk very low (0.001,
    dropped from rank 9 → 20 of 37) for a question about it, even though the
    chunk is directly on-topic - it just never says the word "dataset". Not a
    pipeline bug; logged as a real case for the reranker fine-tuning phase (§12).

Next:
- Run the parser over the target paper set and review the QC reports
- Ingest more papers, extend the golden set to multi-document comparison
  questions (rag.md §7.7/§11 already describe the format)
- Investigate LLM call latency further (~40-60s of the ~70s total is the Ollama
  call itself; thinking mode is the main suspect)

## Phase 2 status

Done:
- `RagEngine.preview_retrieval()` and `abstractrag eval --retrieval-only`: score
  recall@k/MRR/abstain accuracy without calling the LLM at all (generation is
  the slow part; retrieval metrics never needed it). Cut one ablation run from
  ~190-290s to ~60s (first run per process pays model-load time once; every
  further run just reuses it).
- **Ablation table** (2026-09-24, 11-question golden set, RAG paper, threshold 0.12):

  | Mode | recall@5 | MRR | abstain accuracy |
  |---|---|---|---|
  | dense + rerank | 0.88 | 0.81 | 0.91 |
  | sparse + rerank | 0.88 | 0.81 | 0.91 |
  | hybrid + rerank | 0.88 | 0.81 | 0.91 |
  | hybrid, rerank off | 0.88 | 0.73 | **0.27** |

  **Finding:** on this small a corpus (37 chunks, `retrieval.candidates: 50`),
  stage 1 already returns essentially every chunk regardless of dense/sparse/
  hybrid, so the reranker - which scores the full candidate set - erases any
  difference between retrieval modes. The real finding is reranking itself:
  turning it off does not just lower ranking quality (MRR 0.81 → 0.73), it
  collapses abstain accuracy to 0.27. RRF's fused scores (~0.03, "not
  comparable to cosine similarity" - `rrf.py`) sit on a completely different
  scale than the reranker's 0-1 scores that `score_threshold` is calibrated
  against, so with reranking off almost everything scores below threshold and
  the system abstains even when the answer was retrieved. The comment in
  `rrf.py` already said the abstain decision depends on the reranker score;
  this measures the size of that dependency for the first time. Re-run once
  the corpus has more chunks than `retrieval.candidates` actually filters.

Next:
- Chunking variants (target size, overlap) - not yet ablated, needs a config
  sweep the same way retrieval mode was compared here

## Phase 3 status

Done:
- **Citation verification** (2026-09-24): `rag/verification/` - `claims.py` splits
  an answer into one claim per sentence with its `[n]` markers (pure code, no LLM),
  `verifier.py` checks every claim against the passage it cited in one batched LLM
  call. Verdicts: `supported` / `unsupported` / `uncited`. On by default
  (`verification.enabled`), surfaced in `Answer.verified_claims`, the CLI, and the
  Gradio console. Abstains are never verified - nothing was claimed.
  - Anything the judge does not clearly clear counts as unsupported: a missing
    verdict line, an unreadable one, or a citation to a marker that was never in
    the context. A trust mechanism must not pass a verdict it could not read.
  - **Verified against the real model, both directions**: a true claim came back
    `supported`, and a fabricated one ("trained on 4096 TPUs", absent from the
    passage) came back `unsupported` with the reason "No training on 4096 TPUs".
    A judge that only ever agrees would be worthless, so this check matters more
    than the happy path.
  - Cost: one extra LLM call per answered question. A real two-claim answer went
    from ~70s to ~200s end to end. `ACR_VERIFICATION__ENABLED=false` turns it off.

- **Map-reduce summarisation** (2026-09-24): `rag/summarization/` - `sections.py`
  groups a document's chunks into top-level-section groups (pure code, no LLM),
  `map_reduce.py` summarises each group from its real source text (map), then
  writes the final answer from those summaries (reduce), citing `[n]` = section.
  `RagEngine.summarize()`, `abstractrag summarize --document-id --question`,
  `POST /api/v1/summarize`, and a Gradio "Summarize" tab. Verification resolves
  a marker to the section's real source chunks, never the intermediate summary -
  the same anti-hallucination property citation verification established, applied
  one level up.
  - **The core measurement, run against the real model (RAG paper, 2026-09-24)**:
    asked *"What is the main contribution of this paper?"* through both paths.
    `ask` (top-k retrieval) **abstained** - "This source does not contain that
    information" - the reranker never surfaced a chunk that scores this global
    question above threshold, because no single chunk states the paper's
    contribution; it is spread across the abstract, introduction and results.
    `summarize --question "..."` answered correctly and concisely: *"The main
    contribution is retrieval-augmented generation (RAG), a method that combines
    pre-trained parametric memory with non-parametric memory through a
    general-purpose fine-tuning approach [2]"* (`verification: 1/1 claims
    supported`). This is the failure top-k retrieval cannot fix by construction
    (rag.md 8) and map-reduce was built to solve - confirmed, not assumed.
  - **Question-awareness confirmed**: the same document produced two different
    outputs - a full 9-sentence summary with no `--question`, and the single
    focused sentence above with one. The question reaches the map prompt, not
    just the reduce prompt.
  - **Real cost, and it's higher than planned**: the RAG paper's chunks group
    into **32 sections**, not the ~7 estimated in rag.md 8.1 - `section_path[0]`
    does not collapse "2.1 Models" under "2 Methods" the way assumed; every
    subsection heading is its own top-level group. 32 map calls + 1 reduce =
    **~27 minutes** for a plain summary (a focused question ran faster, ~16 min,
    once Ollama had already served several requests this run - variance between
    the two runs was substantial call-to-call). Revisit the grouping key once
    a second, longer document is ingested (rag.md 8.1, `group_sections`).
  - **Real, unfixed finding: verification silently fails to parse on a summary's
    claim set.** The 9-claim plain-summary run came back `verification: 0/9`,
    every claim `"the judge returned no readable verdict for this claim"` - not
    a hallucination finding, a judge-output-parsing failure. The 1-claim focused
    run parsed fine. Suspected cause: the same Qwen3 "thinking" token-budget
    problem already seen in the map stage (`finish_reason=length`,
    `reasoning_chars` in the thousands, see the two warnings logged during the
    32-section map run) - a 9-claim judge prompt is large enough to hit it even
    though `"think": false` is sent. Not yet fixed; logged here rather than
    guessed at. Candidate fixes for later: raise `llm.max_tokens` for the judge
    call specifically, or verify a summary's claims in smaller batches.

Next:
- Fix or work around the verification parsing failure on larger claim sets
  (see the finding above) before trusting `summarize`'s verification numbers
- Measure it: add a verification metric to the eval harness so "faithfulness
  before/after verification" (this phase's stated measure) becomes a number
- Query routing (specific vs global questions) - the one Phase 3 sub-feature
  still unbuilt; now that both pipelines exist, routing has somewhere to route
