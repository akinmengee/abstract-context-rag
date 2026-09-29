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
| 5 | RAPTOR / GraphRAG | Hierarchical index (RAPTOR); GraphRAG deferred | Global-question accuracy vs map-reduce | ✅ |
| 6 | Wikipedia | Real end-to-end run of the (already-built) Wikipedia adapter, golden set | Same golden-set metrics on a second domain | ✅ |
| 7 | Multimodal | Figure captioning via Docling + a second Ollama vision model (tables already handled by Docling) | Three models tried live: `moondream` hallucinates, `qwen2.5vl:3b` hits an open Ollama GPU bug, `granite3.2-vision:2b` is honest but generic - stays off by default | 🚧 |
| 8 | Fine-tuning | Generator SFT data + Kaggle training script (embedding/reranker deferred) | 465 verified examples, trained on Kaggle, converted to GGUF, measured against the base model on the full 70-question golden set: **regressed** (accuracy 0.79→0.64, abstain accuracy 0.84→0.74) - traced to the training set's 58% abstain skew; base model stays the default | ✅ |
| 9 | Interfaces | React web app (done: accounts, chat history, streaming chat), Flutter mobile app, Docker packaging | Real web UI built; a conversation started on web is ready to continue on a future phone client through the same JWT contract | 🚧 |
| 10 | Deployment (optional) | Auth (done, pulled forward into phase 9 - mobile continuity needed accounts now), rate limiting, HTTPS | — | 🚧 |

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

- **Chunking ablation** (2026-09-28, same 11-question golden set, RAG paper,
  `agent.mode: off` to isolate chunking from agent grading, threshold 0.12):

  | `target_chars` / `overlap_chars` | chunks | recall@5 | MRR | abstain accuracy |
  |---|---|---|---|---|
  | 1800 / 200 (current default) | 37 | 0.88 | 0.81 | 0.91 |
  | 900 / 100 | 43 | 0.88 | 0.75 | **1.00** |
  | 2700 / 300 | 35 | 0.88 | 0.81 | **1.00** |
  | 1800 / 0 | 37 | 0.88 | 0.81 | **1.00** |

  **Finding:** target size and overlap barely move retrieval quality on this
  corpus (37 chunks either way) - recall@5 is identical across every variant,
  MRR only dips for the smallest chunks. The one place a difference showed up
  is the single already-known borderline case (Phase 4's findings: "the
  question the threshold wrongly abstained on"): "Which dataset is used for
  the fact verification experiments?" scores 0.07 (below the 0.12 threshold)
  under the default chunking, but 0.12-0.14 (above it) under all three
  alternatives tried - the exact chunk boundary changes what text the
  cross-encoder scores for the top candidate, and the default happens to be
  the one config, of four, that lands on the wrong side.

  Limits, honestly: one borderline question moving is not strong evidence for
  changing the production default - it is exactly the small-n noise the
  phase 4 findings already warned about, just now demonstrated in the other
  direction (a config *rescuing* the known miss rather than causing it). Not
  changing `config.yaml`'s default on a single data point; worth re-checking
  if the golden set grows past four questions per kind.

Next:
- ~~Chunking variants (target size, overlap)~~ - done above.

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
  sections; reviewed and approved by the user on 2026-09-25 (rag.md 7.7).

Found and fixed along the way (all measured, details in `rag.md` 3):
- `ollama pull qwen3:4b` is **Qwen3-4B-Thinking-2507**: its template opens every
  reply with `<think>`, and no flag switches that off (`think: false` on `/v1`
  was ignored: ~3000 reasoning tokens and ~57s for a one-word grading reply;
  `/no_think` sometimes still filled all 4096 tokens). The project now runs its
  own Q4_K_M GGUF of **Qwen3-4B-Instruct-2507** from `llm/models/`, packaged
  with `llm/models/qwen3-4b-instruct-8k.Modelfile`: the same grading call takes
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

## Phase 4 follow-up: the missing-bridge fix (2026-09-28)

Phase 5's RAPTOR measurement (below) found multi-hop had fallen to 1/6 on the
six-paper corpus, root-caused to a specific mechanism: the planner's follow-up
search jumped straight to the second hop's own terms (e.g. "DPR training
negatives") instead of first confirming the unstated link ("RAG's retriever is
DPR") against a real passage - the model already knew the answer and searched
for its consequences instead of grounding it, so the final context often had
the target fact but not the sentence connecting it to the question.

Done:
- `MultiHopAgent` tracked exactly one chunk pool, reset to empty whenever the
  first search's grade failed - so the planner's own prompt showed it
  "(none yet)" even when that first search *had* retrieved something, just
  nothing graded as sufficient for the full compound question. Now a second,
  ungraded list (`seen`, capped the same as the citation pool) carries every
  hop's retrieved chunks regardless of grading into the planning prompt, so
  the model reasons from what the index actually returned instead of a blank
  slate. The citation-worthy pool is untouched - only graded-sufficient hops
  still land there.
- `_SEARCH_ADVICE` (`rag/agents/prompts.py`) reworded from an example
  ("search for that, e.g. ...") to an instruction with an explicit priority
  rule: identify the link *before* anything else, and never substitute a fact
  the model already knows for one the passages have actually stated.

**Live measurement (2026-09-28)**, the four RAG->DPR bridge questions from
`multi_paper.json`, full pipeline (retrieval, rerank, agent, generation,
verification) against the real corpus - run twice for consistency, through
the Ollama-free `--profile gpu` container (see `docker-compose.yml`'s `llm`
service, verified working end-to-end in the same session) so the result is
independent of which LLM server is behind it:
- **2 of 4 now answer correctly with `supported_claims: 1.00`** (no
  hallucination) - previously these hit the same missing-bridge failure Phase
  5 measured. `evidence recall` across all four: 0.54.
- **2 of 4 still wrongly abstain** - and tracing one of them past what the
  eval report shows (fetching the actual chunk text from Qdrant) found the
  correct passage *was* retrieved, word for word ("78.4% vs. 59.1% for top-20
  accuracy on Natural Questions"), but the final context pool held a
  different, unrelated RAG-paper section instead of the one stating "our
  retriever is based on DPR" - so the generator, given DPR's numbers but
  nothing connecting them to the question, correctly declined to assemble the
  claim itself rather than reaching for its own knowledge. Safe, not useless:
  the remaining gap is the planner still sometimes collapsing the
  link-then-detail search into one step rather than issuing the bridge search
  as its own hop, even with the stronger prompt.

Limits, honestly: four questions, the same n=4 caveat as the phase 4 table
above; this is a small-model reasoning limit that a reworded prompt improved
but did not close, not a retrieval problem (retrieval now reliably finds the
right passage). Incidental fix found while running this live: `engine.py`'s
`_free_gpu_for_llm()` unconditionally dropped and reloaded the embedder and
reranker before every LLM call regardless of device - correct and measured on
the shared-GPU local setup (`rag/gpu.py`), pure waste (~20-27s per call, paid
once per hop) on the Docker deployment where they already run on CPU
(`ACR_EMBEDDING__DEVICE=cpu`); both `unload()` methods now skip on `"cpu"`.

Next: force the bridge search as its own hop deterministically (skip the
planner's discretion for the very first follow-up) rather than only advising
it in the prompt, then re-measure the same four questions.

## Phase 5 status (RAPTOR - swapped with fine-tuning)

Order changed on 2026-09-25: fine-tuning's training step runs on Kaggle (2x
~20GB GPU) and needs the user at every step, while RAPTOR can be built and
measured end to end locally. GraphRAG is deferred, not dropped: it would be a
new subsystem with no reuse, solving the multi-hop problem phase 4 already
reaches at 0.75, and its advantage only shows at a corpus size far beyond six
papers. Design: `rag.md` 7.9.1.

Done (step 0, shared groundwork for phases 5 and 6):
- The six core papers of `rag.md` 11 are ingested: Lost in the Middle
  (2307.03172, 40 chunks), Self-RAG (2310.11511, 55) and RAPTOR (2401.18059,
  43) join RAG, DPR and ColBERT - 238 chunks.
- Golden set grown to 48 questions (24 single, 6 comparison, 6 multi-hop, 12
  abstain) with a new `core_papers.json`. Two old abstain targets became
  answerable once their papers were ingested: the Self-RAG critic question was
  replaced by a GraphRAG one; "What MRR@10 does RAPTOR achieve on MS MARCO?"
  stays an abstain - now a harder one, since the paper is in the corpus but
  never evaluates on MS MARCO.
- Golden questions carry `split: train | eval` (default `eval`), and
  `abstractrag eval --split` scores only `eval` by default - so fine-tuning
  data can never be drawn from the questions it will be scored on.
- Retrieval-only check on the six-paper corpus (`agent.mode off`): recall@5
  0.96 over 24 single questions; evidence recall 0.33 on comparisons and 0.53
  on multi-hop; abstain accuracy 0.67 before generation (the reranker still
  cannot see a wrong-system question - generation catches those, phase 4).

Parse QC notes from the new papers: RAPTOR's "3 METHODS" came through as one
section with no subsections (six chunks), and a few captions or prompt text
became fake sections ("Example:", "Summary found in the parent of that
node:", Self-RAG's "Instructions" / "Demonstrations" / "Perceived utility 3").

Done (RAPTOR, 2026-09-25):
- **Measurement first:** a `global` golden kind (`global.json`: a
  main-contribution and a summary question for each of the six papers,
  scoped, no evidence) and `abstractrag eval --entry ask`, which goes through
  the router so global questions reach `summarize()` like a user's would. Each
  question is timed (engine call incl. verification, judges excluded) and
  scored by the share of claims verification cleared.
- **Tree nodes are chunks:** stored in the same Qdrant collection with a
  top-level `level` payload (0 = leaf) and `source_ids` (the leaves a node
  covers). Retrieval hides them unless `retrieval.include_tree_nodes`; points
  ingested before this change have no `level` and count as leaves, so nothing
  had to be re-ingested. `list_chunks()` returns leaves unless a level is asked.
- `rag/raptor/`: `clustering.py` (Ward-linkage agglomerative clustering on unit
  vectors, deterministic, clusters over twice the target size re-clustered -
  simpler than the paper's GMM + UMAP + BIC, and good enough on this corpus) and
  `tree.py` (summarise each cluster with the map-reduce map prompt, embed the
  summaries, repeat up to `max_levels`). `abstractrag build-tree`; ingest
  builds the tree when `raptor.build_on_ingest`.
- **Variant (a)** - `summarization.method: raptor`: `summarize()` reduces the
  document's level-1 nodes in one call, no map calls at question time. A
  citation still resolves to the leaves under a node for verification.
- **Variant (b)** - `retrieval.include_tree_nodes`: `answer()` may retrieve
  tree nodes next to chunks (collapsed tree); a claim citing a node is verified
  against its leaves' text, never the node's summary.

Tree build: 69 nodes over the six papers (9-15 per paper), 61-93 s per paper,
about 7.5 minutes in total, paid once. With the tree built and hidden, the
retrieval-only numbers were identical to step 0 - no regression.

**Global questions** (12, verification on):

| path | accuracy | supported claims | time / question |
|---|---|---|---|
| map-reduce via router (phase 3) | 0.92 | 0.76 | ~103 s |
| **RAPTOR (a): summarize() from level-1 nodes** | **1.00** | **0.78** | **~32 s** |
| RAPTOR (b): tree nodes in answer(), agent off | 0.50 | 0.46 | ~16 s |
| RAPTOR (b): tree nodes in answer(), agent multi_hop | 0.75 | 0.54 | ~49 s |
| plain top-k (tree hidden), agent multi_hop | 0.58 | 0.42 | ~45 s |

**Specific questions** (48 from `rag_paper`, `multi_paper`, `core_papers`;
agent multi_hop, verification off):

| tree nodes | accuracy | single | comparison | multi-hop | abstain | recall@5 | MRR | evidence recall |
|---|---|---|---|---|---|---|---|---|
| hidden (six-paper baseline) | 0.85 | 1.00 | 0.67 | 0.17 | 1.00 | 0.96 | 0.85 | 0.47 |
| visible | 0.88 | 1.00 | 0.83 | 0.17 | 1.00 | 0.92 | 0.74 | 0.32 |

Defaults chosen from these numbers: `summarization.method: raptor` with
`raptor.build_on_ingest: true` (otherwise new documents silently fall back to
map-reduce); `retrieval.include_tree_nodes: false`; the router stays.

Findings:
- **RAPTOR (a) is the win: at least as accurate as map-reduce, ~3x faster.**
  12/12 against 11/12 and 0.78 against 0.76 supported claims are within one
  question of each other; the latency is not (~32 s against ~103 s). The one
  thing given up is question-aware section summaries - the tree is summarised
  once, without the question - and on these 12 questions it cost nothing.
- **Tree nodes in ordinary retrieval (b) do not replace the router.** With the
  multi-hop agent they lift global questions over plain top-k (0.75 against
  0.58) but stay far below (a), and on specific questions they displace leaf
  chunks: recall@5 0.96 -> 0.92, MRR 0.85 -> 0.74, evidence recall 0.47 ->
  0.32 (partly by construction - a node's section label never matches a single
  expected section). Accuracy moved +1 comparison question, within noise.
- **Verifying tree summaries needs heavy trimming.** A level-1 node covers
  about five leaves (~9k characters); a claim citing several nodes exceeds
  `verification.max_prompt_chars`, so passages are cut (26 trims across the
  12 (a) answers) and occasionally a claim gets no readable verdict and counts
  as unsupported. 0.78 is therefore a lower bound.
- **Clusters cross section lines, as RAPTOR intends** - e.g. RAPTOR's title
  chunk clustered with its summarisation-prompt appendix - so a citation label
  like "ABSTRACT; 1 INTRODUCTION; 2 RELATED WORK (+2 more)" is less readable
  than one section name.
- **Multi-hop fell to 1/6 on the six-paper corpus - a phase 4 gap, not a
  RAPTOR effect** (measured with the tree hidden). In five of six questions the
  pool holds the second hop (DPR's training, DPR's results, Lost in the
  Middle's U-curve, ColBERT's MRR@10) but not the bridge that links the
  question to it ("RAG's retriever is DPR", "RAPTOR cites Liu et al.",
  "Self-RAG uses Contriever-MS MARCO"): the first search never ranks the
  bridge chunk in its top five, and the planner jumps straight to the second
  hop from its own knowledge. Without the bridge the generator abstains - the
  grounded behaviour, and deterministic (3/3 reruns). Phase 4's runs had the
  same missing bridge (evidence recall 0.54) but the model filled the gap from
  its own knowledge; the phase 4 multi-hop score was partly ungrounded.
- Integration tests: with Qdrant's storage on a synced folder (OneDrive), a
  deleted collection's directory sometimes lingers and blocks re-creating it.
  The store tests now use a fresh collection name each and tolerate a failed
  cleanup; the root cause is the storage location.

Limits, honestly:
- 12 global questions: one question is ~0.08. The latency gap is the robust
  result; the accuracy difference is not.
- The correctness judge compares against a reference written for this set;
  summaries are long and a lenient judge can pass a partly wrong one.
- Global questions all use the same two phrasings; other phrasings depend on
  the keyword router (phase 3).

Next:
- Multi-hop bridges: make the planner search for the unstated link first
  ("which retriever does RAG use") instead of jumping to the target, and
  re-measure the 48 specific questions.
- Verification of tree-based summaries: judge a claim against the leaves most
  relevant to it rather than all leaves under every cited node.
- Try the paper's GMM + UMAP clustering only if a larger corpus shows the
  simple clustering falling short.

## Phase 6 status (Wikipedia)

Order changed again on 2026-09-25: Wikipedia and figure captioning (phase 7)
can both be built and run end to end locally, so they moved ahead of
fine-tuning, which needs the user's hand at every Kaggle step. Design:
`rag.md` 7.9.2.

The adapter itself was not new work: `ingestion/fetchers/wikipedia.py` and
`ingestion/sources/wikipedia_source.py` were written in phase 1 (the
TextExtracts API, `== Heading ==` parsing, navigation-section dropping) and
unit-tested, but never run against the real API and never given a golden
set. This phase is that measurement.

Done:
- Ingested "Retrieval-augmented generation" (13 chunks, 4 tree nodes) and
  "Question answering" (10 chunks, 3 tree nodes) for real. Both went through
  the full pipeline unmodified: hybrid retrieval, the multi-hop agent,
  RAPTOR tree building, citation, verification, and `is_global_question()`
  routing to `summarize()` - all English-only logic, so nothing needed to
  know an article isn't a paper.
- `golden/wikipedia.json`: 5 `single`, 2 `global`, 3 `abstain` (one
  cross-scoped: a question the *other* Wikipedia article would answer, asked
  with `scope` limited to the one that doesn't cover it). Evidence matching
  needed no code change - `resolve_papers()` already matches on any substring
  of a document's `origin`, and a Wikipedia URL's slug (e.g.
  `Retrieval-augmented_generation`) satisfies that the same way an arXiv ID
  does.
- Found and fixed on the way: `uv run` re-syncs the environment against the
  lock file, and since torch's CUDA build is not something `uv`'s resolver
  can express (rag.md 3), that silently downgrades it back to the CPU-only
  wheel - `Torch not compiled with CUDA enabled` on the first embedding call.
  The fix (`uv pip install torch --index-url .../cu126
  --reinstall-package torch`) was already documented in rag.md 3; this phase
  just ran into it again and switched to calling `.venv/Scripts/abstractrag`
  directly to stop triggering the re-sync.
- Found and fixed: `EvaluationReport.evidence_recall` used the same
  empty-list-means-0.0 bug that faithfulness had (phase 5) - a golden set
  with no `comparison`/`multi_hop` questions (this one) printed `evidence
  recall: 0.00`, reading as a real score instead of "not applicable". Now
  `None` and hidden from the summary line, the same fix pattern as
  faithfulness (`_mean_or_none`).

Measured (`abstractrag eval --entry ask`, 10 questions, verification and
both judges on):

| kind | n | abstain accuracy | accuracy | supported claims | secs/question |
|---|---|---|---|---|---|
| single | 5 | - | 0.80 | 0.87 | 38.3 |
| global | 2 | - | 1.00 | 1.00 | 13.2 |
| abstain | 3 | 1.00 | 1.00 | - | 30.9 |

Overall: recall@5 1.00, MRR 1.00, abstain accuracy 1.00, faithfulness 1.00,
accuracy 0.90, supported claims 0.90.

Findings:
- **The adapter works end to end with no changes.** Retrieval, RAPTOR,
  routing, citation and verification all behaved on Wikipedia prose the same
  way they do on paper text - recall@5 and MRR are both 1.00 at this size,
  and the router correctly sent both "Summarize this article." questions to
  `summarize()`.
- **The one `single` miss is a judge-strictness case, not a wrong answer.**
  "What is RAG poisoning?" retrieved the right section but this run's
  multi-hop follow-up pulled a chunk built around one specific named example
  rather than the general definition; the answer was accurate but framed
  differently from the hand-written `expected_answer`, and the correctness
  judge scored it wrong. A manual re-ask of the same question (single-search,
  no follow-up) answered from the general definition and verified 1/2
  claims supported.
- **Dense, single-paragraph text under-cites.** Asking a broad "what is X"
  question that hits the article's lead paragraph got a nearly
  verbatim-length answer with exactly one trailing `[1]`, instead of a
  marker after each claim - 1/14 claims verified, the rest `uncited`.
  Section-scoped questions (the golden set's actual `single` questions) did
  not show this: 1/1 to 4/6 claims supported, cited per sentence. Likely
  cause: the lead paragraph is one dense block of short factual sentences
  with no internal section markers, unlike a paper's more discursive
  prose - the model appears to treat reproducing the whole block as one
  citable unit rather than several. Not reproduced in the golden set itself,
  so not scored, but a real corpus-specific weakness worth knowing about
  before writing lead-paragraph questions.
- The multi-hop agent still forces one follow-up search on every question
  (phase 4's design, rag.md 7.9), including plain single-fact Wikipedia
  questions that did not need one - visible in `searches=2` on every
  answered row above. Cost, not correctness: the follow-up search's own
  grading step keeps or drops it on its own merits.

Limits, honestly:
- 10 questions, one domain pairing (RAG-adjacent Wikipedia articles chosen
  to overlap the existing corpus's topic). A less technical article's prose
  might expose different weaknesses.
- No `comparison`/`multi_hop` questions: two unrelated Wikipedia articles
  don't form a real evidence chain (rag.md 7.9.2), so those metrics are not
  covered for this domain.

Commit not made - the user commits everything themselves.

## Phase 7 status (figure captioning)

Design: `rag.md` 7.9.3. Finding that shaped it: Docling already exports
tables as structured markdown and picture captions as their own text block
(`pdf_source.py`), so the phase 5-era worry about "tables and figures" (see
the old phase 7 row) was already half solved by the parser - the real gap is
a figure's *visual* content (a diagram, a plot), which was unconditionally
dropped.

Done:
- `BlockType.FIGURE`, a new block type for a figure's description.
- `ingestion.picture_description` settings (`enabled: false`, `model:
  "moondream"`, `prompt`, `timeout_seconds`).
- `PdfSource` configures its `DocumentConverter` with Docling's
  `do_picture_description` + `PictureDescriptionApiOptions` when enabled,
  pointed at the same Ollama `/v1` endpoint the main LLM already uses, just
  a second model tag. A picture's annotation text becomes a `FIGURE` block
  exactly like any other block - chunking, embedding, retrieval, citation
  and verification needed no changes.
- Disabled (the default), behaviour is unchanged: confirmed by re-ingesting
  RAG (2005.11401) and getting the same chunk count as before this phase.

**VRAM and a first quality pass, tested live (2026-09-25):** `ollama pull
moondream` (1.7 GB), `enabled: true`, re-ingested Lost in the Middle
(2307.03172, chosen for being figure-heavy, rag.md 11).

- **A real bug on the first run:** Docling refused with `OperationNotAllowed:
  Connections to remote services is only allowed when set explicitly` -
  `do_picture_description` needs `PdfPipelineOptions(enable_remote_services=
  True)` even for a `localhost` endpoint; Docling's flag means "call out to
  any API-based model," not "the model is actually remote." Fixed by setting
  it whenever picture description is enabled.
- **VRAM fits, no crash.** During parsing (before embedding starts) only
  `moondream` was resident; Ollama swapped it out for `qwen3:4b-instruct-8k`
  once RAPTOR tree-building needed the main model - `ollama ps` after the run
  showed only the 4B model loaded (4.2 GB), 1.9 GB still free on the 6 GB
  card. The two models were never forced to coexist at their combined size;
  Ollama's own LRU eviction handled the handoff.
- **Quality is mixed, on one paper's figures.** 41 chunks (up from 40), 10 of
  them carrying a figure description. One is roughly on-topic but garbled
  ("xtremely detailed graph of tokens on y axis and positions with x axes for
  word retrieval for 4 document types" - for a figure that is, in fact, a
  position-vs-accuracy graph). Another is a clean hallucination, unrelated to
  the paper entirely: a figure near "2.2 Models" was described as "a title
  page of a research paper and online survey by conrad rontgen about nobel
  laureates in medicine." `moondream` is a ~1.7 GB captioning model with no
  particular training on scientific figures, and it shows.
**A second model, same day: `qwen2.5vl:3b` (3.2 GB) - chosen because it is
specifically strong at charts, layouts and OCR, unlike `moondream`'s
natural-image captioning. It does not fit this hardware:**

- First few calls failed outright: Ollama returned `model requires more
  system memory (8.7-8.9 GiB) than is available (8.7-8.9 GiB)` - when a
  vision model does not fit the 6 GB card fully, Ollama falls back toward
  system RAM, and this machine's 16 GB (with the ingest process itself
  already holding a few GB for torch/bge-m3/Docling/RapidOCR) came up just
  short.
- Every call after that timed out at 60s, retried, timed out again, for
  about ten minutes straight (`ReadTimeoutError`, one every ~60-70s from
  20:14 to 20:25) - the model was never responding, not just slow.
- The whole ingest still finished (Docling drops a figure it can never
  describe, same as a disabled picture-description step would), but took
  17m24s wall-clock for one paper, and produced zero figure blocks - back to
  40 chunks, identical to `enabled: false`. All cost, no benefit.

**Root cause found (2026-09-27), correcting the read above: this was never a
hardware-capacity problem, it was Ollama.** `ollama ps` while the "requires
more system memory" errors were happening showed `qwen2.5vl:3b` loaded at
**10 GB, 100% CPU** - never touching the GPU at all, on a card that runs the
4B main LLM at 4.2 GB with no trouble. A direct API call with a tiny 512x384
synthetic test image (3.5 KB) reproduced the same 90s timeout, ruling out
"the page image Docling sends is too large" as the cause. This matches a
known, open Ollama bug
([ollama/ollama#13687](https://github.com/ollama/ollama/issues/13687)):
since Ollama 0.13.4, a change in compute-graph memory estimation for the
qwen2.5vl family inflates the pre-offload memory requirement from ~1.8 GB to
~6.7 GB, so Ollama abandons GPU placement entirely - reported on an 8 GB
card, worse on our 6 GB one. Our installed Ollama (0.17.1) is affected, and
no fix was found as of this writing. Because the bug is in Ollama's own
memory estimation, not the weights file, fetching the same model as a raw
GGUF and loading it via a custom Modelfile (`FROM /path/to/file.gguf`,
exactly how the main LLM is loaded from `llm/models/`) would not have helped -
same Ollama runtime, same bug.

**A third model, chosen to dodge that bug by being a different architecture
entirely: `granite3.2-vision:2b` (2.4 GB) - IBM's small vision-language
model, built specifically for "tables, charts, infographics, plots,
diagrams."** It is also Docling's own default engine for a separate,
dedicated chart-extraction feature (`ChartExtractionVlmEngineOptions`,
untried here - a different pipeline stage from picture-description, local
HF inference rather than an Ollama call), which was the tell that this
family is meant for exactly this job.

- Loaded at **100% GPU, 3.8 GB** - confirmed with `ollama ps` immediately
  after a direct test call (4.9 s for one small synthetic image).
- Real run on Lost in the Middle: 41 chunks (12 carrying a figure
  description), 2m16s total - much faster than `qwen2.5vl:3b`'s failed 17m,
  a bit slower than `moondream`'s under a minute.
- **No hallucinations** on any of the 12 figures - a real improvement over
  `moondream`. But the descriptions are generic and repeat a template
  ("In this image I can see a number of graphs on it, I can see something is
  written on few axes" / "This image consists of some text on a white color
  surface. This looks like a text box.") - it identifies *that* something is
  a chart or a text box, not *what* the chart or text says. A more specific
  prompt (state the axis labels, legend, and the concrete trend) produced the
  same genericness on a second run - this looks like a real capability
  ceiling for a 2 GB model on this hardware, not a prompting problem.
- Reverted after both tests: `enabled: false`, corpus re-ingested once more
  to restore the clean 40-chunk state.

**Default candidate model changed from `moondream` to `granite3.2-vision:2b`**
in both `config.yaml` and the settings' code default - if this is ever turned
on, grounded-but-vague is a safer failure mode than confident hallucination.
`enabled` itself stays `false`: neither model tested is good enough to be
worth the ~2 min/paper cost by default yet.

Limits, honestly: one paper, three vision models, no golden questions written
against figure content - this is "does it run, and how good, roughly," not a
measured accuracy number. VRAM headroom was checked with `nvidia-smi`/
`ollama ps` snapshots, not continuous profiling.

Next, in order of effort: (1) try Docling's dedicated
`ChartExtractionVlmEngineOptions` (Granite Vision, local HF inference, not
Ollama) - a purpose-built feature for exactly this, untried so far; (2) look
for a well-known chart-specific model between `granite3.2-vision`'s 2 GB and
whatever the next size up costs; (3) accept the current ceiling and decide
whether a generic-but-honest caption is worth including at all, versus a
citation-only fallback ("see Figure 3", no attempted description); (4)
revisit `qwen2.5vl` if Ollama ships a fix for #13687. The mechanics (Docling
-> vision model -> FIGURE block -> normal citation/verification) are proven
end to end either way - only the model is still an open question.

Sources:
- [ollama/ollama#13687 - qwen2.5vl:3b no longer runs on 8GB GPUs since Ollama 0.13.4](https://github.com/ollama/ollama/issues/13687)
- [granite3.2-vision on Ollama](https://ollama.com/library/granite3.2-vision)
- [qwen2.5vl:3b on Ollama](https://ollama.com/library/qwen2.5vl:3b)

## Phase 7 continued: query-time figure vision

The three attempts above all shared one assumption: pick a good enough
model, write one fixed description per figure at ingest time. Design
(`rag.md` 7.9.4): that assumption is the actual limit, not the model - one
canned caption cannot anticipate every question a user might ask about a
figure ("what does it show" and "what's the trend" need different answers).
Fix: describe a figure with the user's real question, at answer time, not a
generic prompt at ingest time.

Done:
- Ingest now only *saves* a figure's cropped image (`ingestion.figures`,
  independent of `ingestion.picture_description`) - no vision model call, no
  extra ingest cost beyond writing a PNG. `DocumentBlock.image_path` ->
  `ChunkMetadata.image_paths` carries it through chunking; a figure with no
  caption text is no longer dropped as long as its image was saved.
- `RagEngine._augment_figures()`, called right after context selection in
  both `answer()` and `stream_answer()`: for every selected chunk carrying an
  image, `VisionDescriber` sends that image and the actual question (not
  "describe this figure") to Ollama, and the response replaces the chunk's
  text for that one query only - nothing is written back to Qdrant. Because
  only `.text` changes, `prompts.build_context`, citation extraction and
  verification needed zero changes.
- Both `ingestion.figures.enabled` and `vision.enabled` stay off by default.

**Live test (2026-09-27), Lost in the Middle, `granite3.2-vision:2b`:**
ingest with only image-saving on took about the same time as a plain ingest
and produced no new chunks (a figure with no caption text and no added
description text does not push any section over the chunking threshold) -
15 images saved, 12 chunks carrying an `image_paths` entry.

- **The core premise held.** The same image, asked two different real
  questions ("what does this figure show about modulating input context
  length" vs. "how many lines are plotted, and what do the axes represent"),
  produced two different, question-specific responses about line ordering -
  not a repeated fixed caption. This is what the design set out to prove.
- **Quality is still uneven, the same way it was at ingest time.** On one
  figure the model gave a genuinely relevant (if slightly garbled)
  description of line trends; on another, asked a similarly specific
  question, it produced a complete non sequitur ("Ramón y Cajal won the
  Nobel Prize in Physics..." - unrelated to the actual figure). Asking a
  sharper question does not reliably fix `granite3.2-vision:2b`'s
  hallucination risk, only sometimes sharpens the answer when it does stay
  on topic.
- **The generator's own safety net caught the bad case.** With one
  hallucinated and one genuinely relevant description in context, a real
  `abstractrag ask` call about that figure abstained rather than answering
  confidently from the garbled mix - the grounded-generation design (abstain
  sentinel, verification) did its job; this is the system behaving safely
  under a noisy vision result, not a new bug.

Limits, honestly: one paper, one model, a handful of manually-asked
questions - this shows the mechanism works and roughly how good the model
is, not a measured accuracy number. No golden questions target figure
content yet.

Next: the same open question as before (rag.md 7.9.4, section above) - a
better vision model would help both this and the old ingest-time path,
since the model is the shared variable now, not the architecture. The
mechanism itself (image -> question -> chunk text -> existing pipeline) is
considered done.

## Phase 8 status (generator fine-tuning data)

Design: `rag.md` section 12, tier 2 ("Generator SFT/QLoRA"). Confirmed with
the user before building: no cross-lingual goal - the project and the
fine-tuned model are English-only, Turkish is only the chat language with
the user; the generator is never trained on raw papers (rag.md 12's own
principle - fine-tuning is not how this project adds knowledge, retrieval
is), only on the *behaviour* of answering solely from given context, citing
it, and abstaining when that context does not answer the question; base
model is the same `Qwen/Qwen3-4B-Instruct-2507` already served locally, so
no serving changes are needed after training and reconversion to GGUF.

Done:
- `backend/scripts/generate_training_data.py`: for each real ingested chunk,
  a synthetic question is generated by the local LLM, answered through the
  exact same `prompts.build_context`/`build_messages` calls the production
  engine uses, and kept only if every cited claim verifies as supported by
  that chunk (`ClaimVerifier.verify`, the same check verification runs live)
  - a strict self-distillation filter, not a heuristic one. The same
  question paired with an unrelated chunk from a different document becomes
  an abstain example targeting the literal `NOT_IN_SOURCE` sentinel, never a
  paraphrase of it. Output is one `{"messages": [system, user, assistant]}`
  JSON-lines file, the standard SFT chat format.
- `training/finetune_generator.py`, `training/requirements.txt`,
  `training/README.md`: a standalone LoRA/QLoRA script (`transformers` +
  `peft` + `trl` + `bitsandbytes`, 4-bit by default) meant for Kaggle, not
  `backend/.venv` - it never touches the backend's dependencies or its
  CUDA-sensitive `torch` install. `--merge` produces a full HF model ready
  for `convert_hf_to_gguf.py` + `llama-quantize`, to go back into the same
  Ollama Modelfile pattern the current model already uses.

**First run, 8 documents (2026-09-27):** 522 questions attempted, 812
examples kept - a 78% survival rate, well above this plan's own "roughly
half" expectation. That number turned out to be too good: `is_fully_supported`
only checked the *cited* subset of an answer's sentences, so a multi-sentence
answer with one cited sentence and several uncited ones could still pass in
full. Measured on a Wikipedia article specifically (asked by the user, who
suspected prose-style text might behave differently from paper text): a real
kept example read as five sentences with a single trailing `[1]` - four
uncited assertions the filter never looked at. Production verification calls
that exact case UNCITED and fails it (`verifier.py`'s `_resolve()`); the
training-data filter was looser than the thing it was supposed to imitate.

**Fixed:** `is_fully_supported` now rejects an answer if *any* claim lacks a
marker, before ever calling the verifier - matching production's own bar
instead of a laxer one. Re-running the same Wikipedia article through the
fixed filter dropped its yield from 12/13 to 5/13, and every one of the 5
that remained was fully cited, sentence for sentence.

**Corpus expanded, then the full corpus regenerated with the fixed filter
(2026-09-27):** at the user's request, LoRA (2106.09685) and QLoRA
(2305.14314) were ingested too - both already earmarked for this phase in
rag.md section 11 - bringing the corpus to 10 documents, 369 chunks. Full
run: 738 questions attempted, **544 examples kept (272 positive, 272 paired
abstain)** - a 37% survival rate under the corrected filter, and a spot
check across every kept positive found zero with an uncited sentence.
Yield varies a lot by source: LoRA (49 kept) and the RAG paper (36) sit at
the high end; the Wikipedia RAG article (8) and Question Answering article
(13) sit at the low end - a second, corpus-wide confirmation of the same
prose-vs-paper effect the one-article check already showed.

**A second opinion, checked point by point (2026-09-27):** the user brought
an external review of the dataset. Rather than accept it, every claim was
checked against the actual file:

| Claim | Verdict |
|---|---|
| No multi-citation `[1][3]` ever demonstrated | Confirmed - worse than claimed: 100% of examples were single-block, not "87%" |
| Negative examples all use an easy, cross-document topic mismatch | Confirmed - every abstain distractor came from a different document by construction |
| Negative answer format is clean (exactly `NOT_IN_SOURCE`) | Confirmed - true by construction, no issue |
| ~20% duplicate rows | Roughly confirmed - 91 exact duplicates measured (17%), not 109/20% |
| Numbers get paraphrased instead of quoted exactly | **Not confirmed** - checked every positive answer containing a number against its context; zero mismatches |
| Question-type distribution is narrow (~62% "how", ~11% yes/no) | Confirmed - measured 63.2% and 13.2%, closely matching |

**Fixed, without another Ollama run - by directly authoring examples from
real chunk text instead:**
- **Deduplication:** 91 exact-duplicate lines removed outright (a `temperature: 0.1` artifact - the same chunk asked twice tends to produce nearly the same question).
- **181 hard-negative examples:** each pairs a real, already-verified question with a *same-document, different-section* chunk as context (versus the original easy negatives' cross-document distractors) - constructed programmatically from the corpus already in Qdrant, no LLM call needed since the target is always the fixed `NOT_IN_SOURCE` sentinel. A random sample of 6 was read by hand against the real passages: 5 were clean rejections, 1 was borderline (a generic question that could loosely fit several sections) - an accepted noise level for a genuinely *harder* negative.
- **14 multi-citation examples:** hand-written by directly reading real chunk pairs from the corpus - five cross-document comparisons (e.g. "RAG's retriever is DPR" - the exact bridge fact phase 5 found the multi-hop agent missing; LoRA vs QLoRA's memory savings; DPR's dual-encoder vs ColBERT's late interaction) and nine same-document syntheses (e.g. RAPTOR's problem statement plus its tree-building mechanism). Every sentence in every answer was checked by hand against the source passages before being kept; all 14 use both citation markers, one combines them as `[1][2]` on a single claim.
- The original 272 easy (cross-document) negatives were kept, but sub-sampled to 90 rather than dropped entirely - an easy, obvious-mismatch case is still a real failure mode worth some representation, just not 100% of the negative examples.

**Final composition: 465 examples - 195 positive (181 single-citation + 14
multi-citation) and 270 abstain (181 hard + 89 easy)**, roughly a 42:58
positive:negative split (versus the original 50:50 with no hard negatives
and no multi-citation coverage at all).

**Not done, on purpose:** no training has run yet - that is the user's next
step, on Kaggle, following `training/README.md`. Embedding/reranker
fine-tuning (rag.md 12's tier 1) is deferred; the user prioritized generator
behaviour first, since "answering RAG context correctly" was the concrete
goal, not retrieval quality (already measured extensively in phases 2-7).

Limits, honestly: the quality filter is only as strict as the verifier
already in production (rag.md's own honest limits on that judge apply here
too, and did not run at all on the 195 hand-authored examples - those were
checked by direct reading instead, a different and less systematic kind of
check than the LLM judge gives the rest of the dataset); 465 examples is a
modest dataset for SFT - enough for a first LoRA attempt, not necessarily
enough to reach a ceiling; the hard negatives' "different section" heuristic
is not a guarantee of no overlap, only a reasonable one (matching the spot
check above). No held-out eval split of *this* synthetic data exists - the
project's real golden sets (with their own `train`/`eval` split) remain the
actual measure of whether fine-tuning helped, once a trained model exists to
`abstractrag eval`.

Next: the user trains on Kaggle, downloads and reconverts to GGUF, then
`abstractrag eval` with the new model against the same golden sets already
used for every other phase - a real, comparable before/after number.

## Phase 8 continued: training completed, before/after comparison (2026-09-28)

Done:
- **Training ran on Kaggle** (`kaggle_train_and_convert.py`, T4 x2, ~8h15m):
  465 examples, 5 epochs, LoRA r=16/alpha=32 on all-linear, 4-bit QLoRA base -
  adapter saved successfully. Three real environment failures hit and fixed
  along the way, each the kind of thing that only shows up running the actual
  script rather than reading it:
  - `pip install -U ... peft` pulled a version whose LoRA merge path
    version-checks `torchao` and raises instead of skipping when it finds one
    older than it expects; Kaggle's base image ships one that's too old.
    Fixed by pinning `torchao>=0.16.0` in the install line - but only for the
    *next* run, so it hit for real after the 8-hour training run had already
    finished and only the (separately re-runnable) merge step failed.
    `merge_and_convert_from_adapter.py` exists because of this: pick up an
    already-trained adapter and redo just the merge/GGUF/quantize steps
    (minutes, not hours) without repeating training.
  - The recovery run then hit its own new failure with no GPU attached
    (notebook `Settings > Accelerator` left on `None` - `torch` resolves its
    CPU-only build silently in that case, no error until `device_map`
    references a GPU that isn't there). Both Kaggle scripts now check
    `torch.cuda.is_available()` and fail immediately with an actionable
    message instead of ~2 minutes into a confusing `transformers` traceback.
  - GGUF conversion then hit a *third* one: `AttributeError: 'list' object
    has no attribute 'keys'` inside `transformers`, tracked to another `-U`
    pull regressing on the tokenizer's `extra_special_tokens` field (saved as
    a list, a newer `transformers` expects a dict). Both scripts now sanitize
    this field right after saving the merged tokenizer, before GGUF
    conversion touches it.
- **Merged and converted**: `llm/models/qwen3-4b-finetuned-q4_k_m.gguf` (2.5 GB)
  plus a ready Ollama Modelfile, same template and generation params as the
  base model's.

**Live before/after comparison, run 2026-09-28→29**: full `abstractrag eval`
(all golden sets, 70 questions, judges on) against the base model, then
against the fine-tuned one, same infrastructure both times - the
`--profile gpu` container (verified Ollama-free earlier this session),
swapping only `LLM_MODEL_FILE`. ~1h20m per run (embedding/reranker on the
GPU alongside the LLM, see the `docker-compose.gpu.yml` note below).

| Metric | Base | Fine-tuned |
|---|---|---|
| recall@5 | 0.93 | 0.93 |
| MRR | 0.86 | 0.88 |
| abstain accuracy | 0.84 | **0.74** |
| accuracy (LLM judge) | 0.79 | **0.64** |
| faithfulness | 0.89 | 0.92 |
| supported claims | 0.73 | **0.57** |
| chunks/answer | 3.3 | 2.2 |
| secs/question | 64.2 | 70.7 |

By kind (accuracy / abstain accuracy, base → fine-tuned):

| Kind | n | accuracy | abstain accuracy | supported claims |
|---|---|---|---|---|
| single | 29 | 1.00 → 0.83 | 1.00 → 1.00 | 0.77 → 0.65 |
| comparison | 6 | 0.50 → **0.17** | 0.83 → **0.17** | 0.73 → 1.00 |
| multi_hop | 6 | 0.17 → **0.00** | 0.33 → **0.00** | 1.00 → n/a (abstained on all 6) |
| abstain | 15 | 0.60 → 0.53 | 0.60 → 0.53 | 0.67 → **0.07** |
| global | 14 | 0.93 → 0.86 | 1.00 → 1.00 | 0.63 → 0.61 |

**Finding: fine-tuning made the model worse, not better - recorded honestly
rather than spun.** recall@5 and evidence recall are identical (expected:
fine-tuning never touches retrieval), but everywhere generation behaviour is
measured, the fine-tuned model regressed. It abstains far more readily -
0/6 multi-hop questions answered at all (down from a weak but nonzero
1/6), comparison abstain accuracy fell from 0.83 to 0.17 - and on the rare
case where it *does* answer a question that should have abstained, its
citations are ungrounded far more often (`abstain` kind's supported claims:
0.67 → 0.07). The likely cause is sitting in the dataset composition
already recorded above: **270 of 465 training examples (58%) were abstain
examples** (181 hard negatives + 89 easy), against 195 positive ones - a
generator trained on a majority-abstain distribution over only 5 epochs on
a 465-example set learned "say NOT_IN_SOURCE" as the dominant pattern more
than it learned "answer correctly when the context supports it." This isn't
a measurement artifact: the swing is large, consistent across question
kinds, and in the direction the data imbalance predicts.

**Decision: `config.yaml`'s default stays the base model.** The fine-tuned
GGUF is kept in `llm/models/` and documented, not adopted - a negative result
that cost real time (8h15m training + ~2h40m comparing) is still a result,
and the project's own standard throughout this roadmap is to record what
was tried and measured, not just what worked.

### Aside: model size and where quantization actually pays off

A note worth keeping next to these numbers, not because it explains the
regression above (it doesn't - that's the data imbalance, not model size)
but because it's the reasoning behind fine-tuning a 4B model at all, and it
belongs in the same write-up. Qwen3-4B-Instruct is roughly 8 GB in its
native bf16/fp16 form; Q4_K_M quantization brings that down to the 2.5 GB
GGUF this project actually serves - a real reduction, but a 4B model already
leaves headroom on a 6 GB card even before quantization gets involved.
Compare an 8B model at the same quantization: roughly 15-16 GB unquantized,
which does not fit a 6 GB card at all in any form, against roughly 5 GB
quantized - there, Q4_K_M isn't an optimization, it's the only way the model
runs locally at all. The smaller the base model, the less of the story
quantization is telling; the project's 6 GB VRAM constraint (`rag.md`,
README's Goals) is exactly why 4B was the model chosen to fine-tune in the
first place, and exactly why a hypothetical 8B fine-tune would show a much
more dramatic before/after memory picture than this one did - at the cost of
likely not leaving enough VRAM for the embedder and reranker to share the
card the way `docker-compose.gpu.yml` now does (measured this session: the
4B setup alone already sits at ~5.5/6 GB during generation).

**Follow-up (2026-09-29): `training/` and `backend/scripts/generate_training_data.py`
removed.** Neither was ever committed (deliberately - the raw methodology
scripts were kept local-only, only the already-generated JSONL they produced
was meant to leave the machine, and even that stayed local in the end). With
the fine-tuning decision made and recorded above, there was nothing left for
the code to do that this write-up doesn't already cover: the dataset
generation approach, the verification-based quality filter, the LoRA/QLoRA
config, training time, and the before/after numbers are all still here in
prose. This section is now the record of what was done, not the code that
did it - if the methodology needs to be re-run, it starts from this
description, not from files in this repo.

## Phase 9 status (web interface, accounts, chat history)

Started as "build the real web UI" (a ChatGPT/Gemini-style layout: sidebar
of past conversations, streaming chat panel) and grew during design into a
bigger scope once the user set a concrete requirement: a conversation
started on the web must be continuable from a future Flutter mobile app.
That needs server-side persistence and a notion of whose conversation it
is - real accounts - pulled forward from phase 10 rather than left for
later, since mobile continuity needs them now. Two more decisions shaped
the design: documents (ingested papers/articles) stay one shared pool
across users, not per-user, so a paper ingested once by anyone is reused,
never re-ingested; and a conversation's document scope is fixed once, at
creation, never changed mid-conversation or per-message.

Backend, done:
- `abstractrag/accounts/models.py`: `User`, `Conversation`, `Message`
  (SQLModel - SQLAlchemy with pydantic-native models, matching the rest of
  the codebase's style), backed by a small SQLite file
  (`abstractrag/core/db.py`, `database.path` in config, resolved the same
  way `ingestion.storage_dir` already is). No Alembic yet - three tables, no
  rows in production, `SQLModel.metadata.create_all()` at startup
  (`main.py`'s `lifespan`) is enough until an actual migration is needed.
- `abstractrag/core/security.py`: bcrypt password hashing, PyJWT bearer
  tokens (`POST /auth/register`, `POST /auth/login`) - a single long-lived
  token (30 days), no refresh flow, since this is realistically single-user
  local use (rag.md already frames the project as LAN-only) even though the
  whole thing is deliberately built as if multi-user, for a clean portfolio
  story later.
- `abstractrag/api/conversations.py`: CRUD scoped to the current user
  (`CurrentUserDep`, a new `SessionDep` alongside the existing `EngineDep`
  pattern in `dependencies.py`) - create (optionally against a document, or
  `null` for the existing "all documents" cross-paper agent mode), list,
  get-with-messages, rename, delete.
- `/chat` and `/chat/stream` (`api/chat.py`, rewritten) now require a
  `conversation_id` instead of a per-request `document_id` - the document
  comes from the conversation itself. Both persist the user's question
  immediately and the assistant's answer once it's ready; the streaming
  endpoint's final write uses a fresh DB session inside the SSE generator
  rather than the request-scoped one, since a yield-dependency's lifetime
  across a `StreamingResponse`'s background iteration isn't something worth
  relying on. Verified live against a real server (not just `TestClient`,
  which can mask exactly this class of bug): a two-turn conversation over
  Ollama + Qdrant streamed correctly and both turns persisted.
- Lightweight multi-turn context: `RagEngine.ask()`/`stream_answer()` and
  `prompts.build_messages()` gained an optional `history` parameter - the
  last 3 prior turns (6 messages) are spliced in as their own chat messages
  ahead of the current question, not re-retrieved and not wrapped in the
  `Context:`/`Question:` template (only the current turn gets fresh
  retrieved context). No query rewriting, no re-retrieval - deliberately
  the smallest thing that lets a follow-up like "so how does that work?"
  resolve against what was just discussed. **Verified live**: asked "What is
  retrieval-augmented generation?", then "So how does that work step by
  step?" with no mention of RAG in the second question - the answer opened
  with "Retrieval-augmented generation works step by step as follows: ...",
  correctly resolving the reference purely from injected history.
- First HTTP-layer (`TestClient`) tests in this repo (`test_auth.py`,
  `test_conversations.py`) - 304 tests pass, ruff clean across the whole
  backend.

Frontend (`web/`), done - the skeleton single-screen app is fully replaced:
- `src/auth/`: JWT stored in `localStorage` (not a cookie - the backend's
  CORS has no `allow_credentials`, and a bearer header works identically
  for a future Flutter client, which has no browser-cookie concept at all),
  route guard redirecting to `/login` when there's no token.
- `src/api/sse.ts`: a hand-rolled SSE-over-`fetch` client - the browser's
  native `EventSource` is GET-only with no custom headers, and
  `/chat/stream` is a POST with a JSON body and an `Authorization` header.
- `src/components/Sidebar.tsx` + `ConversationListItem.tsx`: New Chat
  button, a flat Recents list (no date grouping - kept deliberately simple),
  a per-item menu for rename/delete.
- `src/components/NewChatDialog.tsx`: the moment a conversation's document
  scope gets fixed - pick an already-ingested document, ingest a new
  arXiv/Wikipedia source, or choose "all documents." A blocking modal with a
  spinner during ingest (which can take 1-2 minutes for RAPTOR tree
  building) - a background-ingest flow would be more polished but needs a
  job/status concept not warranted for a first pass.
- `src/components/ChatPanel.tsx` + `MessageBubble.tsx`: streams tokens into
  a growing assistant bubble, replaces it with the final answer (citations
  included) on the `done` event.
- CSS Modules throughout, no CSS framework, no state-management library
  (plain React Context for auth) - the user asked not to over-build the
  history mechanism, and that preference extended to the frontend's
  dependency footprint too.

Not done, on purpose (separate from this phase): the Flutter mobile UI
itself (the backend API is already client-agnostic bearer-token auth, ready
for it), Dockerizing Ollama (an unrelated tangent raised mid-design,
explicitly left as-is - it already runs as an always-on host server;
containerizing it would only add GPU-passthrough setup cost with no benefit
for a single local machine), and any retrieval-aware conversational query
rewriting (the "lightweight only" history injection above is deliberately
not that).

## Phase 9 continued (Flutter, Docker packaging, and four bugs real usage found)

- **Flutter (`mobile/`) done**: same backend, accounts and design tokens as
  `web/`. Runs as a Windows desktop app for development (`flutter run -d
  windows`), builds to a real Android debug APK (`flutter build apk`), both
  actually built and verified. Real bug found and fixed: the repo path
  contains a non-ASCII character (`OneDrive\Masaüstü`), which crashed both
  `flutter analyze` and the Android Gradle build - fixed with
  `android.overridePathCheck=true` in `android/gradle.properties` (not
  committed - `android/` is regenerated locally, so this needs re-adding if
  it ever is).
- **Docker packaging done**: `web/Dockerfile` (Node build + nginx, proxying
  `/api/` to the backend container instead of baking in a URL - no CORS
  needed either), a `web` service added to `docker-compose.yml`, the
  backend's stale `ACR_LLM__BASE_URL` (an old raw-llama.cpp port that
  predates the project's move to Ollama) fixed to point at Ollama's actual
  port, CPU device overrides added for the GPU-less containers. Real bug
  found and fixed: nginx's default 60s proxy timeout was cutting off
  multi-hop generation (measured 150-250s) with an empty response -
  `proxy_read_timeout`/`proxy_send_timeout` raised to 600s. `docker compose
  up` (qdrant+backend+web) verified live end to end through the browser-facing
  port only, same as a real deployment would be reached.
- **Four independent bugs found in real usage, all fixed**:
  1. `/chat/stream` (what both web and mobile actually use) never applied
     `/chat`'s global-question routing, so "summarize this paper" streamed a
     weak/abstained retrieval answer instead of a real summary - fixed to
     route the same way, sending the summary as one `token` event since
     map-reduce/RAPTOR has nothing incremental to stream.
  2. Fixing that exposed a second bug: streaming persistence was never
     testable in isolation, since it called the DB engine directly instead
     of through an overridable dependency - added `DbEngineDep` to fix that.
  3. The router's fixed vocabulary didn't recognise "research" ("what is
     this research about" on a Wikipedia article - not a "paper") -
     extended, plus a real safety net: `answer()`/`stream_answer()` now
     retry via `summarize()` before finalising an abstain, whenever nothing
     has been shown to the user yet and there's one document to fall back
     to - closing the gap for whatever phrasing the fixed vocabulary still
     misses, at the cost of a slower genuine abstain.
  4. `/chat` requiring `conversation_id` + auth broke the Gradio dev console
     (`ui/app.py`), unnoticed until now - fixed with a fixed dev account
     that logs itself in and a throwaway conversation per question.
- Backend suite: 317 tests, ruff clean.
- **`ui/` removed entirely, same day.** Once the user saw it running, it
  turned out to just be a source of confusion ("bildiğimiz Python
  uygulaması sanmıştım" - expected a real desktop app, not a Gradio web
  form) with no real job left now that `web/` and `mobile/` both exist -
  its whole reason for existing (Phase 1: a form to poke the API from a
  browser before any real client did) is gone. Deleted the directory, the
  `ui` Docker service, and its Docker image. Layout/README references
  updated; the Phase 1/3/4 status entries above describing it are left
  as-is - they're an accurate record of what was true when they were
  written, not a description of the current state.

## Phase 9 continued: the GPU overlay was never actually verified, real CPU-vs-GPU numbers (2026-09-29)

Asked to make `docker-compose.gpu.yml` (the overlay that moves the backend's
embedder/reranker onto the GPU) actually earn its "verified" comment instead
of taking it on faith. It didn't hold up:

- **Real bug found**: `backend/Dockerfile` always installed the CPU-only
  torch wheel (`--index-url .../whl/cpu`), unconditionally, regardless of
  the overlay's `ACR_EMBEDDING__DEVICE=cuda`. Inside the actual container,
  `torch.cuda.is_available()` was `False` and the first embed call crashed
  with `AssertionError: Torch not compiled with CUDA enabled` - GPU
  reservation or not. The overlay's prior "verified" note had only ever run
  a host Python process (real CUDA torch, installed outside Docker), never
  the container itself, and missed this entirely.
- **Fixed**: `backend/Dockerfile` gained `ARG TORCH_VARIANT=cpu`, which skips
  the CPU-only index (and so resolves PyPI's default CUDA build instead)
  when built with `--build-arg TORCH_VARIANT=cuda`; `docker-compose.gpu.yml`
  now passes that build arg. Confirmed correct as far as install goes: the
  CUDA torch + torchvision + the rest of the dependency set (docling,
  FlagEmbedding, etc.) installed and cached successfully across four
  separate build attempts.
- **Could not verify live, in-container**: all four attempts - two via
  `docker compose build`, one via plain `docker build` (bypassing compose's
  bake driver), one after a full Docker Desktop restart - hung indefinitely
  at the final "exporting to image" step, zero progress each time. Not a
  disk-space issue: freed ~24GB (deleting unrelated large images from
  another project) made no difference, and the two post-restart attempts
  already had 23GB+ free and still hung. This looks like a Docker
  Desktop/WSL2-specific issue on this dev machine, unrelated to the code
  fix - left as a known, unresolved gap. `docker-compose.gpu.yml`'s own
  comment reflects this now instead of overclaiming.
- **Measured for real instead, on the equivalent host path**: `backend/.venv`
  already has CUDA torch (`2.14.0+cu126`, `torch.cuda.is_available()=True`)
  and `config.yaml` already defaults both `embedding.device` and
  `reranker.device` to `"cuda"` - `abstractrag serve` (no Docker) needed no
  changes at all to run on GPU. Timed the real `_search()` pipeline (hybrid
  retrieve + cross-encoder rerank, 50 candidates, all 8 ingested papers)
  both ways:

  | | retrieve() | rerank() | total |
  |---|---|---|---|
  | CPU (Docker's forced default) | 16.59s | 157.90s | 174.49s |
  | GPU (host venv) | 20.19s | 10.32s | 30.51s |

  Reranking - the documented bottleneck - is **15.3x faster** on GPU;
  retrieve() didn't improve (dominated by fixed overhead, not compute) and
  overall wall time is 5.7x faster. End-to-end sanity check: a real
  multi-hop question through the CLI (two full agent hops, each with its
  own embed+rerank cycle plus LLM planning/verification calls) completed in
  80s on GPU - the same shape of query would cost minutes per hop on CPU.
- **Practical upshot**: `docker compose up` (the all-in-one path) forces
  CPU by design (`docker-compose.yml`'s `backend.environment`) and is
  measurably the slow path. The host-venv path (`abstractrag serve` +
  `cd web && npm run dev`, both already documented in the README's Quick
  start) is GPU by default and needs nothing further - this is the path
  actually running now. The Docker GPU overlay remains a real, if currently
  unverifiable-on-this-machine, option for a deploy with no host Python
  setup at all.

**Follow-up, same day**: two compose files for one CPU/GPU toggle was
confusing enough on its own to raise ("gpu yml'yi normal docker compose
ymlye aktaralım... 2 kafa karıştırıyor") - folded `docker-compose.gpu.yml`
into `docker-compose.yml` and deleted it. `backend` is now GPU-by-default
unconditionally (the `TORCH_VARIANT: cuda` build arg, `cuda` device env
vars, and the GPU reservation all live directly on that one service); there
is no CPU profile/overlay left. `docker compose up` on a machine with no
GPU/no NVIDIA Container Toolkit will now fail outright rather than
silently running slow - intentional, given a single-user project doesn't
need to optimize for that machine, and the host-venv path above is the
documented fallback for it. The still-unresolved export hang from the
paragraph above is unaffected by this - it was never about which file the
config lived in.

**Further follow-up, same day**: restructured so every service builds from
its own top-level folder (`backend/`, `web/` already did; added `llm/`
Dockerfile + `llm/models/` moved from root `models/`, and `qdrant/`
Dockerfile - both `FROM <the same public image as before>`, no custom code
yet, just a real build context instead of a bare `image:` reference).
Useful accidental clue about the export hang while verifying this: `docker
compose build qdrant` and `docker compose build llm` - both single-`FROM`
wrappers with zero new layers to write - exported instantly. Only
`backend`'s build (10 new layers, several GB of actually-new content: torch,
docling, FlagEmbedding, etc.) hangs at export. Narrows the earlier "Docker
Desktop/WSL2 issue" guess to something about exporting a large *number of
new layers/bytes* specifically, not Docker's export path in general - still
unresolved, but a real lead for whoever picks this up next.

**Corrected, same day**: `llm/Dockerfile` and `qdrant/Dockerfile` (and the
whole `qdrant/` folder) were reverted right back out - both were `FROM
<vendor image>` and nothing else, no real content, added for consistency
with `backend/`/`web/` rather than because either service actually needed a
build context. That's the wrong test for whether a folder should exist: it
should hold something real, not just look uniform with its neighbors - a
`qdrant/` folder with one boilerplate line invites a reader to go looking
for Qdrant-specific code in it and find nothing (the real Qdrant client,
`backend/abstractrag/database/qdrant_store.py`, runs inside the backend
container, not this one - orchestration code lives where it executes, not
next to the service it talks to). Both services are back to a plain
`image:` reference in docker-compose.yml. `llm/` itself stays, because
`llm/models/` (the actual GGUFs and Modelfiles) is real content that earns
the folder on its own.

**Follow-up, same day - the export hang, run to ground (as far as this machine allows):** deleted every Docker image and pruned the full build cache for a genuinely clean slate (35GB freed) and retried. Two findings:

- **A real, separate bug surfaced once the cache was clean enough to stop masking it**: `backend/Dockerfile` still had `COPY scripts ./scripts`, left over from before `backend/scripts/` was deleted earlier this session. Every prior attempt had this layer cached from before the deletion; a fully pruned cache hit it fresh and failed fast (`"/scripts": not found`) instead of hanging. Fixed by removing the line - `scripts` was never in `pyproject.toml`'s packages list, so nothing depended on it.
- **The export hang itself survived the clean slate**: same indefinite stall at "exporting layers", confirmed via `docker system df` showing zero byte movement across multiple 5+ minute windows. Ruled out stale cache and disk space as causes, for good.
- **One more variant, to isolate build-time computation from Docker Desktop's image store**: switched to the `docker-container` buildx driver with an isolated builder (`docker buildx create --driver docker-container`) instead of the default `docker` driver, which builds inside its own container and needs an explicit `--load` to hand the result back. For the first time in this entire investigation, the export step **actually completed** - `exporting layers` finished in 153.5s, confirmed as real work via sustained 100-250% CPU and several GB of growing block I/O in `docker stats` the whole time, not a frozen process. It then moved to a new, later step - `sending tarball`, the final hand-off of the built image into Docker Desktop's own local image store - and stalled there instead, just as completely: 0% CPU, zero I/O growth, for 4+ confirmed minutes.
- **Conclusion**: the root cause is specifically Docker Desktop's local image-store ingestion on this Windows machine, not the build/export computation - that part is now proven capable of finishing. This is very unlikely to reproduce on a real Linux Docker host, which is this project's actual deployment target anyway (`rag.md` 6 always assumed a real Ubuntu server, not this Windows dev machine). Recommendation, not yet acted on: stop spending more time on this here, and validate `docker compose up --build` directly on a real Ubuntu box or a cheap cloud VM instead.
