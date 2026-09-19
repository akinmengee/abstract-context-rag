// Skeleton: proves the API contract end to end (ingest -> ask -> cited answer).
// The real interface is designed later, once the backend has settled.

import { useState } from "react";

import { ask, ingestArxiv, type Answer } from "./api/client";

export function App() {
  const [arxivId, setArxivId] = useState("2005.11401");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [status, setStatus] = useState("");

  async function handleIngest() {
    setStatus("ingesting…");
    try {
      const result = await ingestArxiv(arxivId);
      setStatus(`${result.title} — ${result.chunk_count} chunks`);
    } catch (error) {
      setStatus((error as Error).message);
    }
  }

  async function handleAsk() {
    setStatus("thinking…");
    try {
      setAnswer(await ask(question));
      setStatus("");
    } catch (error) {
      setStatus((error as Error).message);
    }
  }

  return (
    <main style={{ fontFamily: "system-ui", maxWidth: 720, margin: "2rem auto", padding: "0 1rem" }}>
      <h1>abstract-context-rag</h1>

      <section>
        <h2>Ingest</h2>
        <input value={arxivId} onChange={(event) => setArxivId(event.target.value)} />
        <button onClick={handleIngest}>Add arXiv paper</button>
      </section>

      <section>
        <h2>Ask</h2>
        <input
          style={{ width: "100%" }}
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="What datasets does the paper evaluate on?"
        />
        <button onClick={handleAsk}>Ask</button>
      </section>

      {status && <p>{status}</p>}

      {answer && (
        <section>
          <p>{answer.text}</p>
          <ul>
            {answer.citations.map((citation) => (
              <li key={citation.chunk_id}>
                [{citation.marker}] {citation.title} {citation.section ?? ""}{" "}
                {citation.page ? `p.${citation.page}` : ""}
              </li>
            ))}
          </ul>
        </section>
      )}
    </main>
  );
}
