# P08 — AI Student Assistance Chatbot

## 1. Purpose and scope

Build a university-specific chatbot that answers student academic and institutional questions using a trusted corpus of official university documents. The MVP covers document ingestion, retrieval-augmented answer generation with citations, a basic chat interface, and evaluation against labelled questions.

## 2. Functional requirements

1. **Document corpus:** Ingest syllabus, university regulation, notice, FAQ, and academic policy documents. Each document must retain basic source metadata (title or filename and document type).
2. **Processing:** Extract text from supported document formats, clean it, split it into searchable chunks, and associate each chunk with its source metadata and page or section when available.
3. **Retrieval:** For each student question, retrieve the most relevant chunks from the processed corpus and use them as context for answer generation.
4. **Grounded answers and citations:** Generate concise answers only from the retrieved context. Show citations that identify the supporting document and, where available, its page or section.
5. **Unknown-answer fallback:** If evidence is missing, irrelevant, or insufficient, clearly say that the answer could not be found in the available documents; do not guess.
6. **Basic chat UI:** Provide a simple interface to submit a question and view the answer and its source citations.
7. **Evaluation set:** Maintain a labelled set of student questions with expected answers and/or supporting sources. Use it to assess retrieval and answer faithfulness during development.

## 3. Essential non-functional requirements

- **Correctness and faithfulness:** Answers should be supported by the retrieved official content; unsupported claims should be avoided.
- **Source traceability:** Every substantive answer must expose its supporting source(s), with page or section where available.
- **Basic security:** Keep API keys and other secrets out of source control and the user interface. Validate uploaded files and restrict accepted file types and sizes.
- **Maintainability:** Keep ingestion, retrieval, generation, and UI responsibilities separable; document setup and configuration.
- **Acceptable response time:** A typical question should return an answer within 10 seconds under local or small-demo use, excluding unusually slow external provider outages.

## 4. Minimum architecture and stack assumptions

- Python backend with a small HTTP API (FastAPI assumed).
- Basic web chat UI (React assumed; a simpler equivalent is acceptable).
- Document text extraction and chunking pipeline.
- Embedding model plus a local vector index (FAISS or Chroma assumed).
- LLM accessed through a configurable provider/API; credentials stored outside the codebase.
- Local corpus and index are sufficient for the MVP; no production-scale deployment is assumed.

## 5. Acceptance criteria

- The system can ingest representative documents from each required corpus category and preserve source metadata.
- A student can submit a question through the UI and receive a response based on retrieved corpus content.
- Answers include citations identifying their supporting documents and page/section when available.
- Questions not supported by the corpus receive the explicit unknown-answer fallback rather than a guessed answer.
- The labelled evaluation set can be run to inspect retrieval relevance and answer faithfulness.
- In a small local/demo run, typical responses meet the 10-second target.
- Secrets are not committed in the project, and unsupported upload types or oversized files are rejected.

## 6. Out of scope for the MVP

Authentication and roles, admin dashboards, analytics, conversation memory, multilingual support, streaming responses, notifications, integrations with student information systems, automated live website crawling, and production-scale deployment or availability guarantees.
