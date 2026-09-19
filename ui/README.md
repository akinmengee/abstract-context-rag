# Dev console (Gradio)

Internal debugging panel, not the product UI. It shows the retrieved chunks and
their scores next to each answer, which is what makes retrieval problems visible.

```bash
pip install -r requirements.txt
ACR_API_URL=http://localhost:8000 python app.py   # http://localhost:7860
```

Talks to the backend over HTTP only — it has no access to the engine.
