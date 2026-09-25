# Golden sets

Questions whose correct behaviour is known in advance. Internal test data - the
product never generates questions, it only answers them.

`abstractrag eval` runs every file here by default; `--golden <file>` (repeatable)
runs a subset. The phase 2 ablation table was measured on `rag_paper.json` alone.

Each entry:

| Field | Meaning |
|---|---|
| `question` | What gets asked |
| `expected_answer` | Reference answer, for human review and for the correctness judge |
| `kind` | `single`, `comparison`, `multi_hop` or `abstain` - the report scores each kind separately |
| `scope` | arXiv ID to limit retrieval to, or `null` to search the whole corpus. Any question that says "this paper" needs one |
| `evidence` | Every `{"paper": <arXiv ID>, "section": <section name>}` the answer needs. `single`: exactly 1, `comparison`/`multi_hop`: 2 or more, `abstain`: none |

Every paper named in `scope` or `evidence` must be ingested first; the run stops
before the first question otherwise.

`section` values must be copied from real parsed section names, not invented - a
wrong section silently makes recall@k, MRR and evidence recall meaningless. Read
the real ones out of the store before adding questions:

```python
from abstractrag.core.container import get_engine
engine = get_engine()
for document in engine.store.list_documents():
    print(document["origin"], sorted({c.metadata.section for c in engine.store.list_chunks(document["document_id"])}))
```
