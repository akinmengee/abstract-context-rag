# Roadmap

Each phase adds one capability and states how its effect is measured. Phase order
can change; the folder layout never encodes it.

Status: ✅ done · 🚧 in progress · ⬜ not started

| # | Phase | What it adds | How it is measured | Status |
|---|---|---|---|---|
| 1 | Baseline | Docling parsing, section-aware chunking, bge-m3, Qdrant, grounded answers from Ollama | Parse QC report, first golden set answers | ✅ |
| 2 | Retrieval depth | Hybrid search, RRF, cross-encoder reranking, chunking variants | Ablation table: dense vs sparse vs hybrid vs hybrid+rerank (recall@k, MRR, abstain accuracy) | ✅ |
| 3 | Query and verification | Query routing, map-reduce summaries, citation verification | Faithfulness before/after verification; abstain accuracy | 🚧 |
| 4 | Agentic RAG | Corrective / self-reflective retrieval, multi-paper questions | Multi-hop question accuracy | ✅ |
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
  - **Real finding, attempted fix, still unresolved: verification silently fails
    to parse on a summary's claim set.** The 9-claim plain-summary run came back
    `verification: 0/9`, every claim `"the judge returned no readable verdict for
    this claim"` - not a hallucination finding, a judge-output-parsing failure.
    The 1-claim focused run parsed fine.

- **Query routing** (2026-09-24): `rag/query/router.py::is_global_question()` -
  a pure, LLM-free keyword classifier (multi-word phrases anchored to "this/the
  paper/document/article/study", not bare words - a bare "summary"/"overview"/
  "key result" would wrongly match table- or figure-specific questions).
  `RagEngine.ask(question, document_id)` is the new default entry point for
  `abstractrag ask` and `POST /api/v1/chat`: routes to `summarize()` only when
  both a document is scoped and the question looks global, otherwise `answer()`
  as before. `POST /api/v1/summarize`, `abstractrag summarize`, and
  `/chat/stream` stay unrouted on purpose (streaming has no map-reduce path).
  - **The routing feature's own proof, run against the real model (2026-09-24)**:
    `abstractrag ask "What is the main contribution of this paper?" --document-id
    <id>` - the exact question that made plain `ask` abstain two runs earlier -
    now prints `Global question detected - summarising the whole document...`
    and returns the correct, verified answer (`verification: 1/1 claims
    supported`) in ~13 minutes, with no `--question` flag and no need to know
    `summarize` exists. A specific question through the same `ask` command
    (`"What retriever does this paper use?"`) still answers normally in ~106s,
    confirming routing does not change existing behaviour.

  **Verification fix attempted (2026-09-24) and it did not work on the real
  model - recorded honestly rather than claimed fixed:** `judge_max_tokens`
  (a dedicated, larger token budget for the judge call only - `verifier.py`,
  `VerificationSettings.judge_max_tokens=8192`) was implemented, unit-tested,
  and reviewed clean. Re-running the exact same 9-claim summary against the
  real model produced **the identical result**: `verification: 0/9`, every
  claim unreadable - `judge_max_tokens=8192` made no observable difference.
  The judge call logged no "empty content" warning this time (unlike the map
  calls, which did hit `finish_reason=length` on this same run) - meaning the
  response was non-empty, just never contained a parseable `<n>|YES/NO|<reason>`
  line for any of the 9 claims. Leading suspect, not yet confirmed: `llm.ctx_size`
  (4096, `config.yaml`) is configured but **never actually sent to Ollama** -
  `llm_client.py`'s `_payload()` has no `num_ctx`/`options` field at all - so
  raising `max_tokens` cannot help if the server-side context window is the
  real ceiling. This needs its own investigation (capture the judge's raw
  response, or try wiring `num_ctx` through) rather than a second guess stacked
  on the first one.

  **Root cause found and fixed (2026-09-25) - it was the prompt, not the reply:**
  a summary cites whole sections, and one judge call carried the full text of
  every cited section - most of the paper. Ollama's OpenAI-compatible API cuts
  a prompt longer than its context window from the front, with no error,
  taking the system prompt and its reply format with it (measured directly: a
  6965-token prompt came back as `prompt_tokens: 4096`). Raising the output
  budget could never help. Three fixes, each measured on the same summary:
  - 8k context baked into the Ollama model (a per-request `num_ctx` is
    ignored on `/v1`), and a client warning whenever a prompt fills the window
    - still 0/22: the prompt was ~16k tokens, larger than 8k too.
  - Judge claims in batches that fit `verification.max_prompt_chars`, trimming
    passages only for a single claim that exceeds it on its own - no more
    unreadable verdicts, but 1/11: the instruct model now put all 31 section
    markers at the start of the summary, leaving 10 sentences uncited.
  - Reduce prompt with an explicit citation-format example -
    **15/17 claims supported, 0 unreadable, 0 truncated**, in ~2 minutes (the
    same summary took ~27 minutes in phase 3). The two `unsupported` verdicts
    carry real reasons, e.g. a sentence about "Table 7" whose data never made
    it through parsing.

- **5 code-review bugs fixed (2026-09-25), all found in the query-routing
  commit:**
  - `router.py`: `_NARROW_GLOBAL_PATTERNS` (`main`/`key` phrases) now includes
    "finding(s)"/"result(s)", so "what are the main findings of this paper?"
    routes correctly - it fell through to plain retrieval before.
  - `router.py`: those same `main`/`key` phrases are not anchored to `_DOC`
    (unlike every other pattern), so instead of anchoring them - which would
    have broken "what are the key findings?" with no document mention, an
    existing passing case - a nearby `Table N`/`Figure N` reference now vetoes
    the match, catching the concrete failure case ("what key finding does
    Table 3 report?") without narrowing the deliberately-unanchored ones.
  - `router.py`: the gap between a trigger word and `_DOC` is now
    `[^.!?]{0,30}` instead of `.{0,30}`, so it can no longer bridge a
    sentence boundary ("I read the summary of related work. Does this paper
    also cover X?" no longer misroutes).
  - `ui/app.py` `ask()`: section-summary rows now go to their own Dataframe
    (`ask_summaries_output`, headers `marker/section/summary`), toggled
    visible instead of being packed into the retrieval table's `score`/`page`
    columns under the wrong labels.
  - `verifier.py`: `ClaimVerifier`'s `judge_max_tokens` default is now read
    from `VerificationSettings().judge_max_tokens` instead of a second
    hardcoded `8192`, so it cannot silently drift from the configured value.
  - Verified: 33/33 router tests (10 new cases covering the three bugs above),
    full backend suite 164 passed / 4 skipped, `ruff check` clean, `ui/app.py`
    compiles. Not yet re-run against the real model (routing's own live-model
    proof already exists from the prior run; these are precision/display
    fixes, not new codepaths).

Next (deferred by decision, not blocked):
- Measure it: add a verification metric to the eval harness so "faithfulness
  before/after verification" (this phase's stated measure) becomes a number
- Phase 2 leftover: chunking-variant ablation (target size, overlap)
- `group_sections` still makes one group per subsection (32 for the RAG paper)
- cheap now at ~2 minutes per summary, but still not the grouping planned

## Phase 4 status

Done (2026-09-25):
- **Multi-paper evaluation first** (measurement before technique): DPR
  (2004.04906) and ColBERT (2004.12832) ingested next to the RAG paper - DPR is
  RAG's retriever, so it forms a real two-hop chain; ColBERT adds comparisons
  and a distractor. Golden format moved from one `expected_section` to
  `evidence: [{paper, section}]` + `scope` + `kind` (single / comparison /
  multi_hop / abstain), validated at load time. `rag_paper.json` was migrated
  with `scope: 2005.11401` on every question: "this paper" means nothing once a
  second paper is ingested. New `multi_paper.json`: 19 questions. New metrics:
  **evidence recall** (share of expected `(paper, section)` pairs the LLM saw,
  no LLM needed) and **accuracy** (`CorrectnessJudge` against the reference
  answer - faithfulness cannot see a missing second hop). Reports break every
  number down by kind and record `agent_mode`; a run stops before the first
  question if a golden paper is not ingested.
- **`rag/agents/`, selected with `agent.mode`** - the engine still writes the
  answer, so citations, abstain and verification work the same in every mode:
  - `corrective`: an LLM grader checks whether any retrieved chunk answers the
    question; if none does, or the reranker threshold already failed, the query
    is rewritten and searched once more before abstaining. A *gate*, not a
    filter - see findings.
  - `multi_hop`: self-ask. After the first (corrective) search a planner writes
    follow-up searches one at a time (`SEARCH: ...` / `DONE`, at most 3
    searches), each one graded and its kept chunks pooled in hop order (at most
    8). The first follow-up offers no DONE option.
  - Plain Python, no LangGraph: a bounded loop of three steps does not need a
    graph framework.

**Ablation** (30 questions over three papers, Qwen3-4B-Instruct Q4_K_M, 8k
context, judges at temperature 0, verification off):

| agent.mode | accuracy | single (14) | comparison (4) | multi-hop (4) | abstain (8) | abstain acc. | evidence recall | chunks / answer | time / question |
|---|---|---|---|---|---|---|---|---|---|
| off | 0.83 | 0.93 | 1.00 | 0.00 | 1.00 | 0.87 | 0.52 | 5.0 | ~18s |
| corrective | 0.83 | 0.93 | 1.00 | 0.00 | 1.00 | 0.90 | 0.52 | 5.0 | ~24s |
| **multi_hop** | **0.90** | **1.00** | 0.50 | **0.75** | 1.00 | **0.97** | 0.54 | 3.9 | ~44s |

`multi_hop` is the default in `config.yaml`. Reference point: the same `off`
run on the thinking model used through phase 3 scored accuracy 0.70,
faithfulness 0.53, and took ~105s per question.

Findings:
- **Multi-hop works where a single query cannot.** "How did RAG's retriever
  compare to BM25 in top-20 accuracy?" - `off` abstains (nothing retrieved
  says RAG's retriever is DPR); `multi_hop` searches DPR's training and results
  next and answers "78.4% vs 59.1%" citing DPR 5.1. On the embedding-dimension
  question `off` answered **728** - which RAG's own appendix G really says
  ("21M 728 dimensional vectors", a typo in that paper) - while `multi_hop`
  followed the chain to DPR 3.1 and answered 768.
- **The generator already catches wrong-entity context.** Retrieval-only, all
  three corpus-level abstain questions failed: "What MRR@10 does RAPTOR
  achieve on MS MARCO?" scores 0.77 against ColBERT's MS MARCO results, and
  the reranker threshold cannot tell "wrong system" from "right topic". With
  generation, all three abstained correctly - the grounding prompt's
  NOT_IN_SOURCE rule, not the threshold, is the layer that holds here.
- **Corrective retrieval added nothing on this corpus.** As a *filter*
  (keeping only graded chunks) it cost accuracy, 0.83 -> 0.73 in an interim
  run (judges not yet at temperature 0): it dropped
  chunks an answer needed, often one side of a comparison. As a *gate* it is
  neutral: it rescued the question the threshold wrongly abstained on
  ("Which dataset is used for the fact verification experiments?", via the
  rewrite "fact verification dataset") but that answer was still judged wrong.
  Kept, because multi-hop builds every hop on it.
- **A small model says "done" too early.** With a DONE option on the first
  plan, the planner never searched a second time - it declared a comparison
  with one side, or a chain with an unstated link, complete. Forcing one
  follow-up search costs seconds on the instruct model.
- **Filtering hurts comparisons.** Multi-hop pools only graded chunks to fit
  its budget, and on comparisons that drops a side (1.00 -> 0.50).

Limits, honestly:
- Four questions per kind: one question is 0.25. The multi-hop gain (0 -> 3 of
  4) is real; the comparison drop (4 -> 2 of 4) is within what a few more
  questions could move.
- Evidence recall is a lower bound: a fact often appears in more than one
  section (ColBERT's datasets are in both 4.1 and the introduction), and only
  the listed section counts.
- The model knows these papers from pre-training. `off` answered comparisons
  it had only half the evidence for; faithfulness and verification exist to
  catch exactly that, but accuracy alone can reward it.
- The three-hop question (RAG -> DPR's related work -> ColBERT) is unsolved:
  the planner searched "the concurrent work" without ever naming ColBERT, and
  the query rewrite had used one of the three searches.
- The multi-paper questions and reference answers were drafted in the same
  session that built the agents, by the same assistant, from the real parsed
  sections - a human review of `multi_paper.json` is still owed (rag.md 7.7).

Found and fixed along the way (all measured, details in `rag.md` 3):
- `ollama pull qwen3:4b` is **Qwen3-4B-Thinking-2507**: its template opens every
  reply with `<think>`, and no flag switches that off (`think: false` on `/v1`
  was ignored: ~3000 reasoning tokens and ~57s for a one-word grading reply;
  `/no_think` sometimes still filled all 4096 tokens). The project now runs its
  own Q4_K_M GGUF of **Qwen3-4B-Instruct-2507** from `models/`, packaged with
  `backend/ollama/qwen3-4b-instruct-8k.Modelfile`: the same grading call takes
  0.5s. The phase 1 attempt to import this GGUF had failed only because it was
  given the thinking model's template.
- Ollama's `/v1` silently truncates prompts past the server's context window
  and ignores a per-request `num_ctx` - the root cause of the phase 3
  verification failure (above). 8k context is now baked into the model, and
  the client warns when a prompt fills the window.
- Both judges now run at temperature 0: at 0.1, two near-identical answers
  were judged differently.
- Faithfulness 0.53 on the thinking model was a judge artefact (correct,
  plainly grounded answers such as "BART-large [1][5]" judged unsupported);
  the same pipeline on the instruct model scores 0.78.

Next:
- Grow the golden set (more comparison and multi-hop questions per kind) before
  tuning prompts further - at n=4, tuning would fit noise.
- Comparisons in multi-hop: keep each hop's full context up to the budget
  rather than only graded chunks, and measure.
- Three-hop chains: count the query rewrite outside the search budget, or
  raise `max_searches`, and measure the cost.
