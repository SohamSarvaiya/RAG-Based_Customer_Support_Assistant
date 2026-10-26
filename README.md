# RAG-Based Customer Support Assistant

Upload company PDFs (policies, FAQs, manuals) and ask questions. The assistant retrieves the relevant passages,
answers **only from them**, and shows **page-level source citations**. If the documents don't contain the answer,
it says so instead of guessing.

## How the RAG pipeline works

```
PDF -> extract text per page -> sentence-aware chunks (800 chars, 150 overlap) -> embeddings -> Chroma (cosine)

Question -> embed -> top-k semantic search -> drop chunks below MIN_SCORE
         -> no chunks?  REFUSE (LLM is never called)
         -> else LLM with strict prompt (context only, cite [n], reply NOT_FOUND if unsure)
         -> answer + only the sources it actually cited
```

### How wrong answers are reduced
| Guardrail | Where |
|---|---|
| Similarity threshold (`MIN_SCORE`) - weak matches are discarded, and with none left the bot refuses without calling the LLM | `pipeline.py` |
| Strict system prompt: context-only, no outside knowledge, `NOT_FOUND` when unsure, temperature 0 | `generator.py` |
| Mandatory `[n]` citations; only cited passages are returned as sources (file + page + score + snippet) | `pipeline.py` |
| Evaluation set with out-of-scope questions measuring refusal accuracy | `eval/` |

## Features
PDF upload - chunking - embeddings (sentence-transformers) - semantic search (Chroma) - context-based answers (Claude) -
source citations - evaluation dataset - error handling (non-PDF, fake/corrupt/scanned/oversized/empty files, LLM outage) -
API-key auth (`X-API-Key`) - Docker - pytest (22 tests).

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env          # set API_KEYS and ANTHROPIC_API_KEY
uvicorn app.main:app --reload # demo UI: http://localhost:8000  |  API docs: /docs
```
**LLM options** - put ONE key in `.env` (provider is auto-detected, or force it with `LLM_PROVIDER`):
- `ANTHROPIC_API_KEY` - Claude
- `OPENAI_API_KEY` - ChatGPT models (`OPENAI_MODEL`, default `gpt-4o-mini`)
- `GEMINI_API_KEY` - Google Gemini (`GEMINI_MODEL`, default `gemini-2.5-flash`)
- `LLM_PROVIDER=ollama` - free local model (`ollama pull llama3.2:3b`)
- no key - retrieval-only: returns the best passage with its citation

## Docker
```bash
cp .env.example .env && docker compose up --build
```
The vector DB persists in `./data`.

## API
```bash
H="X-API-Key: change-me-123"
curl -H "$H" -F file=@sample_docs/AcmeCo_Support_Handbook.pdf localhost:8000/documents
curl -H "$H" -H "Content-Type: application/json" -d '{"question":"What is your refund policy?"}' localhost:8000/ask
curl -H "$H" localhost:8000/documents
curl -H "$H" -X DELETE localhost:8000/documents/<doc_id>
```
Example response:
```json
{"answer":"Full refund within 30 days if unused and in original packaging [1].","grounded":true,
 "sources":[{"id":1,"filename":"AcmeCo_Support_Handbook.pdf","page":1,"score":0.71,"snippet":"..."}],"latency_ms":820}
```

## Tests & evaluation
```bash
pytest -q
python -m eval.run_eval      # retrieval hit-rate, page match, refusal accuracy (+ answer accuracy if LLM key set)
```
Tune `MIN_SCORE` with the eval set: raise it if the bot answers out-of-scope questions, lower it if it refuses valid ones.

## Roadmap
OCR for scanned PDFs - hybrid (BM25 + vector) search with re-ranking - per-tenant collections - streaming answers.
