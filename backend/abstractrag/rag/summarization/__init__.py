"""Summarising a whole document, which top-k retrieval structurally cannot do.

  sections.py    groups chunks into top-level sections, one per LLM call (no LLM)
  prompts.py     the map and reduce prompts
  map_reduce.py  summarise each section, then write the answer from those summaries

A citation marker here means a section, not a chunk, and verification resolves it
to that section's source text - see rag.md section 8.1. With summarization.method
raptor, the map stage is replaced by the document's RAPTOR tree (rag/raptor/),
built once at ingestion, and only the reduce prompt runs per question.
"""
