"""RAPTOR: a summary tree over each document, built once at indexing time (roadmap phase 5).

  clustering.py  groups embeddings into clusters of about cluster_size (no LLM)
  tree.py        summarises each cluster, re-embeds the summaries, repeats up the tree

Nodes live in the same Qdrant collection as the chunks, marked with a level and
the leaf chunks they cover (rag.md 7.9.1). A claim citing a node is verified
against those leaves, never the node's own summary.
"""
