# Shared API contract

`openapi.json` is generated from the FastAPI app and is the single source of truth
for every client, so the React and Flutter apps cannot drift from the backend.

```bash
cd backend && abstractrag export-openapi --output ../shared/openapi.json
```

Then generate typed clients:

```bash
cd web && npm run generate:api                # TypeScript
# Flutter: openapi-generator with the dart-dio generator
```

Generated clients land in `shared/generated/` (gitignored) or inside each app.
