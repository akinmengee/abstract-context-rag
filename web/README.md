# Web (React + Vite + TypeScript)

A ChatGPT/Gemini-style client: email+password accounts (JWT bearer tokens,
`src/auth/`), a sidebar of past conversations (`src/components/Sidebar.tsx`),
and a streaming chat panel (`src/components/ChatPanel.tsx`, reading
`/api/v1/chat/stream`'s Server-Sent Events by hand via `src/api/sse.ts`,
since that endpoint is a POST with a body and an auth header - both
incompatible with the browser's native `EventSource`). A conversation's
document scope (one paper/article, or "all documents" for the existing
cross-paper agent mode) is fixed once, when it's created, in
`src/components/NewChatDialog.tsx`. Routing (`react-router-dom`) covers
`/login`, `/register` and `/chat/:conversationId` (`src/pages/`);
`src/components/ThemeToggle.tsx` flips a `data-theme` attribute for manual
light/dark, on top of the `prefers-color-scheme` default in `src/index.css`.

**Prerequisites:** Node 18+.

```bash
npm install
npm run dev             # http://localhost:5173
npm run build           # production build, tsc -b && vite build -> dist/
```

Point it at a backend other than localhost with `VITE_API_URL`. The backend
must be running with accounts enabled (it always is - `abstractrag serve`
creates `data/app.db` on first startup, no extra flag needed).

Types in `src/api/client.ts` are hand-written against the backend's actual
Pydantic response models - see
[`backend/abstractrag/api/README.md`](../backend/abstractrag/api/README.md)
for the endpoints themselves. Production build/serve is via
`web/Dockerfile` + `web/nginx.conf` (see the root
[README](../README.md#docker)); `npm run build` alone is enough for a plain
static-file deploy.
