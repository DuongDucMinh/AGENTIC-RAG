# Vietnamese Tax Legal Agentic RAG

FastAPI backend for a Vietnamese legal RAG assistant focused on tax, fees, charges, and financial obligations. It uses the Hugging Face dataset `th1nhng0/vietnamese-legal-documents`, Qdrant hybrid retrieval, reranking, LangGraph orchestration, and Groq for answer generation.

## Checkpoints

For detailed module-by-module commands, Colab artifact preparation, BM25/RRF notes, LangSmith tracing, RAGAS-lite evaluation, and Mermaid diagrams, see [HUONG_DAN_CHAY_TUNG_PHAN.md](HUONG_DAN_CHAY_TUNG_PHAN.md).

### Recommended Data Workflow

Do not preprocess the full Hugging Face dataset on a weak local machine. Keep the preprocessing code in this repo, but run the heavy job on Colab:

```bash
python scripts/prepare_artifact.py --max-documents 50 --output-dir artifacts/legal_tax_v1_50
```

Download the artifact zip, extract it locally, then import it:

```powershell
python scripts/06_import_artifact.py --artifact-dir artifacts/legal_tax_v1_50 --reset
```

This imports parent chunks, indexes child chunks into Qdrant, and builds the PyVi-tokenized BM25 sidecar used with RRF fusion.

### A. API + logging

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload
```

Open:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`

### B. Docker + Qdrant

```powershell
docker compose up --build
```

Qdrant runs at `http://localhost:6333`.

### C. Preview selected legal documents

```powershell
curl -X POST "http://127.0.0.1:8000/indexing/preview?limit=5"
```

### D-F. Index a small subset

Start small:

```powershell
curl -X POST "http://127.0.0.1:8000/indexing/run?max_documents=20&reset_collection=true"
```

Then inspect status:

```powershell
curl "http://127.0.0.1:8000/indexing/status"
```

### G. Test retrieval

```powershell
curl -X POST "http://127.0.0.1:8000/retrieval/search" ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"mức thu lệ phí trước bạ được quy định như thế nào?\",\"top_k\":5}"
```

### H. Chat

Set `GROQ_API_KEY` in `.env`, then:

```powershell
curl -X POST "http://127.0.0.1:8000/chat" ^
  -H "Content-Type: application/json" ^
  -d "{\"session_id\":\"demo-session\",\"question\":\"Lệ phí trước bạ được quy định như thế nào?\",\"debug\":true}"
```

## Design

- Source dataset: `th1nhng0/vietnamese-legal-documents`
- Domain filter: tax, fees, charges, and tax policy
- Primary document types: `Luật`, `Nghị định`, `Thông tư`, `Thông tư liên tịch`
- Status: `Còn hiệu lực`, `Hết hiệu lực một phần`
- Retrieval: hybrid dense+sparse search over child chunks
- Context: parent chunks by legal article
- Reranking: cross-encoder, configurable
- Agent: LangGraph query rewrite, retrieval, answer generation

## Notes

- Do not index the whole dataset first. Start with `max_documents=20`, then `100`, then `1000`.
- The app starts without `GROQ_API_KEY`; `/chat` returns a clear configuration error until the key is set.
- Logs are designed to identify failures in config, dataset loading, HTML cleaning, parsing, indexing, Qdrant, retrieval, reranking, and LLM calls.
