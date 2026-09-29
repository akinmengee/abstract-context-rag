// Hand-rolled SSE-over-fetch client. The browser's native EventSource is
// GET-only and cannot send a JSON body or a custom Authorization header,
// both of which /chat/stream needs - so this reads the response body's
// ReadableStream directly and parses "event: ...\ndata: ...\n\n" frames.

import { authHeaders, BASE_URL, type Answer, type Citation } from "./client";

export interface StreamEvent {
  event: "citations" | "token" | "done";
  citations?: Citation[];
  token?: string;
  answer?: Answer;
}

export async function* streamChat(
  conversationId: string,
  question: string,
): AsyncGenerator<StreamEvent> {
  const response = await fetch(`${BASE_URL}/api/v1/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ conversation_id: conversationId, question }),
  });

  if (response.status === 401) {
    localStorage.removeItem("acr_token");
    localStorage.removeItem("acr_email");
    window.location.assign("/login");
    throw new Error("session expired");
  }
  if (!response.ok || !response.body) {
    const detail = await response
      .json()
      .catch(() => ({ detail: response.statusText }));
    throw new Error(detail.detail ?? "request failed");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) return;
    buffer += decoder.decode(value, { stream: true });
    let boundary: number;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      if (frame.trim()) yield parseFrame(frame);
    }
  }
}

function parseFrame(frame: string): StreamEvent {
  let event = "message";
  let data = "";
  for (const line of frame.split("\n")) {
    if (line.startsWith("event: ")) event = line.slice(7);
    if (line.startsWith("data: ")) data = line.slice(6);
  }
  return { event: event as StreamEvent["event"], ...JSON.parse(data) };
}
