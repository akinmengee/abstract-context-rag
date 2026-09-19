# Web (React + Vite + TypeScript)

Skeleton only. It exercises the API contract end to end — ingest a paper, ask a
question, render the cited answer — so the backend can be validated from a browser.
The real interface is designed once the backend has settled (roadmap phase 9).

```bash
npm install
npm run dev            # http://localhost:5173
```

Point it at a backend other than localhost with `VITE_API_URL`.

Types in `src/api/client.ts` are hand-written for now. Once the backend has written
`shared/openapi.json` (`abstractrag export-openapi`), replace them:

```bash
npm run generate:api
```
