"""Query routing: deciding whether a question needs retrieval or the whole document.

  router.py   is_global_question() - keyword-based classifier, no LLM call

Query rewriting, multi-query, HyDE, step-back and decomposition (rag.md 7.4)
remain unimplemented - only the routing decision between the specific-question
pipeline and the global-summary pipeline is built so far.
"""
