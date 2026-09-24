"""Evaluation harness: does the pipeline actually work, and did a change help?

Metrics are implemented here rather than pulled from RAGAS/DeepEval - recall@k
and MRR are short arithmetic, and a framework would add a heavy dependency
(langchain, cloud-API defaults) around code the project exists to understand.

  models.py   GoldenQuestion, QuestionResult, EvaluationReport
  metrics.py  recall@k, MRR - pure functions over ranked section names
  judge.py    LLM-as-judge faithfulness, the one metric arithmetic can't give
  runner.py   runs a golden set through the engine, aggregates a report
  golden/     the question sets themselves (see golden/README.md)

Run it with `abstractrag eval`; compare retrieval modes by running it once per
mode with ACR_RETRIEVAL__MODE set.
"""
