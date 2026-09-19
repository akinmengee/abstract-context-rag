# Roadmap

Each phase adds one capability and states how its effect is measured. Phase order
can change; the folder layout never encodes it.

Status: ✅ done · 🚧 in progress · ⬜ not started

| # | Phase | What it adds | How it is measured | Status |
|---|---|---|---|---|
| 1 | Baseline | Docling parsing, section-aware chunking, bge-m3, Qdrant, grounded answers from llama.cpp | Parse QC report, first golden set answers | 🚧 |
| 2 | Retrieval depth | Hybrid search, RRF, cross-encoder reranking, chunking variants | Ablation table: BM25 vs dense vs hybrid vs hybrid+rerank (recall@k, MRR, nDCG) | 🚧 |
| 3 | Query and verification | Query routing, map-reduce summaries, citation verification | Faithfulness before/after verification; abstain accuracy | ⬜ |
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

Next:
- Run the parser over the target paper set and review the QC reports
- Build the golden dataset (LLM draft, verified by hand) and the evaluation harness
- Produce the first ablation table
