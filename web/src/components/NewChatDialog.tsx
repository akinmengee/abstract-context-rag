import { useEffect, useState } from "react";

import {
  createConversation,
  ingestArxiv,
  ingestWikipedia,
  listDocuments,
  type ConversationSummary,
  type DocumentSummary,
} from "../api/client";
import styles from "./NewChatDialog.module.css";

export function NewChatDialog({
  onClose,
  onCreated,
}: {
  onClose(): void;
  onCreated(conversation: ConversationSummary): void;
}) {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [arxivId, setArxivId] = useState("");
  const [wikiTitle, setWikiTitle] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    listDocuments().then(setDocuments).catch(() => setDocuments([]));
  }, []);

  async function pickExisting(document: DocumentSummary) {
    setError("");
    setBusy(true);
    try {
      onCreated(await createConversation(document.document_id, document.title));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function pickAllDocuments() {
    setError("");
    setBusy(true);
    try {
      onCreated(await createConversation(null, "All documents"));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function ingestAndCreate(kind: "arxiv" | "wikipedia") {
    setError("");
    setBusy(true);
    try {
      const result =
        kind === "arxiv" ? await ingestArxiv(arxivId.trim()) : await ingestWikipedia(wikiTitle.trim());
      onCreated(await createConversation(result.document_id, result.title));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.dialog} onClick={(event) => event.stopPropagation()}>
        <h2 className={styles.title}>New Chat</h2>

        <div className={styles.section}>
          <div className={styles.sectionLabel}>Existing document</div>
          <div className={styles.docList}>
            {documents.length === 0 && <span>No documents ingested yet.</span>}
            {documents.map((document) => (
              <button
                key={document.document_id}
                className={styles.docButton}
                disabled={busy}
                onClick={() => pickExisting(document)}
              >
                {document.title}
              </button>
            ))}
          </div>
        </div>

        <div className={styles.section}>
          <div className={styles.sectionLabel}>Ingest a new arXiv paper</div>
          <div className={styles.row}>
            <input
              placeholder="e.g. 2005.11401"
              value={arxivId}
              onChange={(event) => setArxivId(event.target.value)}
            />
            <button disabled={busy || !arxivId.trim()} onClick={() => ingestAndCreate("arxiv")}>
              {busy ? "Working…" : "Ingest"}
            </button>
          </div>
        </div>

        <div className={styles.section}>
          <div className={styles.sectionLabel}>Ingest a new Wikipedia article</div>
          <div className={styles.row}>
            <input
              placeholder="Paste a wikipedia.org link, or type the exact title"
              value={wikiTitle}
              onChange={(event) => setWikiTitle(event.target.value)}
            />
            <button disabled={busy || !wikiTitle.trim()} onClick={() => ingestAndCreate("wikipedia")}>
              {busy ? "Working…" : "Ingest"}
            </button>
          </div>
        </div>

        <div className={styles.section}>
          <div className={styles.sectionLabel}>Or</div>
          <button className={styles.allDocsButton} disabled={busy} onClick={pickAllDocuments}>
            All documents (cross-paper)
          </button>
        </div>

        {error && <p className={styles.error}>{error}</p>}

        <div className={styles.footer}>
          <button className={styles.cancel} onClick={onClose}>
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}
