# Golden sets

Questions whose correct behaviour is known in advance. Internal test data - the
product never generates questions, it only answers them.

One JSON file per paper. Each entry:

| Field | Meaning |
|---|---|
| `question` | What gets asked |
| `expected_answer` | Reference answer, for human review and as judge context |
| `expected_section` | Section the answer lives in — the retrieval ground truth. `null` for abstain questions |
| `is_abstain` | `true` when the paper does not contain the answer and the engine must say so |

`expected_section` values must be copied from real parsed section names, not
invented — a wrong section silently makes recall@k and MRR meaningless. Read the
real ones out of the store before adding questions:

```python
points, _ = store.client.scroll(collection_name=store.collection, limit=500, with_payload=True)
{p.payload["chunk"]["metadata"]["section"] for p in points}
```
