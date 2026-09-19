// Thin client for the backend API.
// Types are hand-written for now; run `npm run generate:api` once
// shared/openapi.json exists to replace them with generated ones.

const BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export interface Citation {
  marker: number;
  chunk_id: string;
  title: string;
  section: string | null;
  page: number | null;
  origin: string;
}

export interface Answer {
  text: string;
  citations: Citation[];
  abstained: boolean;
}

export interface IngestResult {
  document_id: string;
  title: string;
  source_type: string;
  origin: string;
  chunk_count: number;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(detail.detail ?? "request failed");
  }
  return response.json() as Promise<T>;
}

export function ingestArxiv(arxivId: string): Promise<IngestResult> {
  return post<IngestResult>("/api/v1/ingest", { arxiv_id: arxivId });
}

export function ingestWikipedia(article: string): Promise<IngestResult> {
  return post<IngestResult>("/api/v1/ingest", { wikipedia: article });
}

export function ask(question: string, documentId?: string): Promise<Answer> {
  return post<Answer>("/api/v1/chat", { question, document_id: documentId ?? null });
}
