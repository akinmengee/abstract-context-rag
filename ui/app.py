"""Gradio dev console.

Not the product UI - this is the panel used while building: ingest a source, ask a
question, and see exactly which chunks were retrieved and how they scored. It only
speaks HTTP to the backend, so it can never drift from what the API actually offers.
"""

import os

import gradio as gr
import httpx

API_URL = os.getenv("ACR_API_URL", "http://localhost:8000")
TIMEOUT = httpx.Timeout(600.0)  # parsing a paper on CPU is slow


def _post(path: str, **kwargs) -> dict:
    response = httpx.post(f"{API_URL}{path}", timeout=TIMEOUT, **kwargs)
    if response.status_code >= 400:
        raise gr.Error(response.json().get("detail", response.text))
    return response.json()


def ingest(arxiv_id: str, wikipedia: str, pdf_path: str) -> str:
    if pdf_path:
        with open(pdf_path, "rb") as handle:
            result = _post("/api/v1/ingest/upload", files={"file": (os.path.basename(pdf_path), handle)})
    elif arxiv_id:
        result = _post("/api/v1/ingest", json={"arxiv_id": arxiv_id})
    elif wikipedia:
        result = _post("/api/v1/ingest", json={"wikipedia": wikipedia})
    else:
        raise gr.Error("Give an arXiv ID, a Wikipedia title, or a PDF.")

    return f"{result['title']}\n{result['chunk_count']} chunks\nid: {result['document_id']}"


def ask(question: str, document_id: str) -> tuple[str, list[list]]:
    payload = {"question": question, "document_id": document_id or None}
    answer = _post("/api/v1/chat", json=payload)

    sources = "\n".join(
        f"[{citation['marker']}] {citation['title']} — {citation.get('section') or ''} "
        f"{citation.get('page') or ''} — {citation['origin']}"
        for citation in answer["citations"]
    )
    text = answer["text"] + (f"\n\nSources:\n{sources}" if sources else "")

    # The retrieval table is the point of this panel: it shows why the answer looks
    # the way it does, including when the engine abstained.
    rows = [
        [
            round(chunk.get("rerank_score") or chunk["score"], 4),
            chunk["chunk"]["metadata"].get("section") or "",
            chunk["chunk"]["metadata"].get("page") or "",
            chunk["chunk"]["text"][:300],
        ]
        for chunk in answer["used_chunks"]
    ]
    return text, rows


def list_documents() -> list[list]:
    response = httpx.get(f"{API_URL}/api/v1/documents", timeout=TIMEOUT)
    response.raise_for_status()
    return [
        [document["title"], document["source_type"], document["chunk_count"], document["document_id"]]
        for document in response.json()
    ]


with gr.Blocks(title="abstract-context-rag dev console") as demo:
    gr.Markdown("# abstract-context-rag — dev console")

    with gr.Tab("Ingest"):
        arxiv_input = gr.Textbox(label="arXiv ID", placeholder="2005.11401")
        wiki_input = gr.Textbox(label="Wikipedia title or URL")
        pdf_input = gr.File(label="PDF", type="filepath")
        ingest_output = gr.Textbox(label="Result", lines=3)
        gr.Button("Ingest", variant="primary").click(
            ingest, [arxiv_input, wiki_input, pdf_input], ingest_output
        )

    with gr.Tab("Ask"):
        question_input = gr.Textbox(label="Question", lines=2)
        document_input = gr.Textbox(label="Document ID (optional)")
        answer_output = gr.Textbox(label="Answer", lines=8)
        chunks_output = gr.Dataframe(
            headers=["score", "section", "page", "text"], label="Retrieved context", wrap=True
        )
        gr.Button("Ask", variant="primary").click(
            ask, [question_input, document_input], [answer_output, chunks_output]
        )

    with gr.Tab("Documents"):
        documents_output = gr.Dataframe(
            headers=["title", "source", "chunks", "document_id"], label="Ingested"
        )
        gr.Button("Refresh").click(list_documents, None, documents_output)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
