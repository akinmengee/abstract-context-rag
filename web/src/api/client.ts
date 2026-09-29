// Thin client for the backend API. Types are hand-written.

// Empty by default: every call below uses a path starting with "/api/v1",
// so a relative BASE_URL resolves against whatever origin served this page -
// Vite's dev proxy locally, nginx's /api/ proxy in the Docker image. Set
// VITE_API_URL only to point dev at a backend that isn't localhost:8000.
export const BASE_URL = import.meta.env.VITE_API_URL ?? "";

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

export interface DocumentSummary {
  document_id: string;
  title: string;
  source_type: string;
  origin: string;
  chunk_count: number;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  user_id: string;
  email: string;
}

export interface ConversationSummary {
  id: string;
  title: string;
  document_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface MessageOut {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[] | null;
  created_at: string;
}

export interface ConversationDetail extends ConversationSummary {
  messages: MessageOut[];
}

// Exported so api/sse.ts (which streams the response body itself instead of
// going through request()) can attach the same auth header.
export function authHeaders(): Record<string, string> {
  const token = localStorage.getItem("acr_token");
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...authHeaders(), ...(init.headers ?? {}) },
  });
  if (response.status === 401) {
    localStorage.removeItem("acr_token");
    localStorage.removeItem("acr_email");
    window.location.assign("/login");
    throw new Error("session expired");
  }
  if (!response.ok) {
    const detail = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(detail.detail ?? "request failed");
  }
  return response.status === 204 ? (undefined as T) : (response.json() as Promise<T>);
}

function get<T>(path: string): Promise<T> {
  return request<T>(path);
}

function post<T>(path: string, body?: unknown): Promise<T> {
  return request<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });
}

function patch<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, { method: "PATCH", body: JSON.stringify(body) });
}

function del<T>(path: string): Promise<T> {
  return request<T>(path, { method: "DELETE" });
}

// --- Auth -------------------------------------------------------------

export function register(email: string, password: string): Promise<TokenResponse> {
  return post<TokenResponse>("/api/v1/auth/register", { email, password });
}

export function login(email: string, password: string): Promise<TokenResponse> {
  return post<TokenResponse>("/api/v1/auth/login", { email, password });
}

// --- Documents ----------------------------------------------------------

export function listDocuments(): Promise<DocumentSummary[]> {
  return get<DocumentSummary[]>("/api/v1/documents");
}

export function ingestArxiv(arxivId: string): Promise<IngestResult> {
  return post<IngestResult>("/api/v1/ingest", { arxiv_id: arxivId });
}

export function ingestWikipedia(article: string): Promise<IngestResult> {
  return post<IngestResult>("/api/v1/ingest", { wikipedia: article });
}

// --- Conversations --------------------------------------------------------

export function listConversations(): Promise<ConversationSummary[]> {
  return get<ConversationSummary[]>("/api/v1/conversations");
}

export function createConversation(
  documentId: string | null,
  title?: string,
): Promise<ConversationSummary> {
  return post<ConversationSummary>("/api/v1/conversations", { document_id: documentId, title });
}

export function getConversation(id: string): Promise<ConversationDetail> {
  return get<ConversationDetail>(`/api/v1/conversations/${id}`);
}

export function renameConversation(id: string, title: string): Promise<ConversationSummary> {
  return patch<ConversationSummary>(`/api/v1/conversations/${id}`, { title });
}

export function deleteConversation(id: string): Promise<void> {
  return del<void>(`/api/v1/conversations/${id}`);
}

// --- Chat -----------------------------------------------------------------
// Non-streaming; kept for parity/scripted use. The live chat UI streams
// exclusively via api/sse.ts's streamChat().
export function ask(conversationId: string, question: string): Promise<Answer> {
  return post<Answer>("/api/v1/chat", { conversation_id: conversationId, question });
}
