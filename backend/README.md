# LPU CSE-AIML RAG Chatbot — Backend Service

FastAPI-based backend foundation for the Lovely Professional University (LPU) CSE-AIML Student Assistance RAG Chatbot.

---

## Directory Structure

```
backend/
├── app/
│   ├── api/             # API route controllers and endpoints
│   │   └── endpoints/   # Individual endpoint modules (e.g. health.py)
│   ├── core/            # App configuration and environment settings (config.py)
│   ├── rag/             # Modular RAG package placeholder
│   ├── schemas/         # Pydantic request and response schemas
│   ├── services/        # Business logic and service layers
│   ├── utils/           # Shared helper functions
│   └── main.py          # FastAPI application entrypoint
├── tests/               # Pytest automated test suite
├── .env.example         # Environment variable template
├── .gitignore           # Backend-specific ignore rules
├── requirements.txt     # Essential Python dependency specifications
└── README.md            # Setup and execution guide
```

---

## Prerequisites

- **Python**: version 3.10 or newer (tested on 3.10–3.15).

---

## Setup Instructions

### 1. Create and Activate a Virtual Environment

Open a terminal and navigate to the `backend/` directory:

```bash
cd backend
```

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(If you encounter execution policy restrictions in PowerShell, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

**On Windows (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 2. Install Dependencies

Install the essential foundation dependencies:

```bash
python -m pip install -r requirements.txt
```

---

### 3. Configure Environment Variables

Create your local `.env` configuration file from the template:

**On Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
```

**On Linux / macOS / Bash:**
```bash
cp .env.example .env
```

---

## Running the Backend Server

Start the development server with live reload enabled:

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The server will start at:
- **Base URL:** `http://localhost:8000`
- **Interactive Swagger Docs:** `http://localhost:8000/docs`
- **Alternative ReDoc:** `http://localhost:8000/redoc`

---

## Testing the Health Endpoint

### Option A: Using Web Browser
Open your browser and navigate to:
```
http://127.0.0.1:8000/health
```

### Option B: Using PowerShell
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get | ConvertTo-Json
```

### Option C: Using cURL
```bash
curl -X GET http://127.0.0.1:8000/health
```

**Expected JSON Response:**
```json
{
  "status": "healthy",
  "service": "LPU CSE-AIML RAG Chatbot Backend",
  "version": "0.1.0",
  "environment": "development",
  "timestamp": "2026-11-16T10:40:00.000000+00:00"
}
```

---

## Document Ingestion

To ingest the raw university documents (`data/raw/`) into structured internal documents:

```bash
# Run from repository root
python scripts/ingest.py
```

This will:
- Process syllabus PDFs (page-by-page tracking).
- Extract regulation & notice DOCX files (preserving headings and Markdown tables).
- Ingest student FAQ JSON records (canonical source of truth, skipping redundant FAQ DOCX).
- Clean and normalize text without dropping academic codes or dates.
- Output the normalized documents to `data/processed/ingested_documents.json`.

---

## Running Automated Tests

Run the test suite using `pytest`:

```bash
python -m pytest
```
