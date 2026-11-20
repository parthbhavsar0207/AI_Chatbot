# LPU Academic Assistant frontend

React, TypeScript, Vite, and Tailwind CSS client for the LPU CSE-AIML Student Assistance chatbot.

## Run locally

```powershell
cd frontend
pnpm install
Copy-Item .env.example .env.local
pnpm dev
```

Set `VITE_API_BASE_URL` in `.env.local` to the FastAPI origin (default: `http://localhost:8000`). Start the backend separately and ensure its CORS allowlist includes the Vite origin (`http://localhost:5173`).

## Validate

```powershell
pnpm test
pnpm build
```
