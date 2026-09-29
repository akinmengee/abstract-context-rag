"""Agents that decide what context an answer is built from.

  base.py        ContextSelection, and the search/release tools the engine lends
  prompts.py     grade, rewrite and plan prompts, with parsers that can say "unreadable"
  corrective.py  grade retrieved chunks; rewrite and retry the query when none answer it
  multi_hop.py   plan follow-up searches one at a time, each one graded (self-ask)

Selected with `agent.mode` (off | corrective | multi_hop); the engine still writes
the answer, so citations, abstain and verification work the same in every mode.
Plain Python rather than LangGraph: a bounded loop of at most `max_searches`
steps does not need a graph framework (rag.md 7.9).
"""
