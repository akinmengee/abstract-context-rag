"""Citation verification - not implemented yet (roadmap phase 3).

Planned: split an answer into atomic claims, check each claim against the chunk it
cites with an NLI model or an LLM judge, and flag the ones that are not supported.
Today's guards are the grounding prompt and the reranker score threshold in the engine.
"""
