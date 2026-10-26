import hashlib
import re
import time

from app.config import Settings
from app.errors import DocumentError
from app.rag.chunker import chunk_pages
from app.rag.generator import ABSTAIN, NOT_FOUND, Generator
from app.rag.loader import extract_pages
from app.rag.vectorstore import VectorStore


class RAGPipeline:
    def __init__(self, settings: Settings, embedder, store: VectorStore, generator: Generator):
        self.s, self.embedder, self.store, self.generator = settings, embedder, store, generator

    # ---------- ingestion: PDF -> pages -> chunks -> embeddings -> vector DB ----------
    def ingest_pdf(self, filename: str, data: bytes) -> dict:
        doc_id = hashlib.sha256(data).hexdigest()[:16]
        if self.store.has_document(doc_id):
            return {"doc_id": doc_id, "filename": filename, "status": "already_indexed", "chunks": 0}
        pages = extract_pages(data)
        chunks = chunk_pages(pages, self.s.chunk_size, self.s.chunk_overlap)
        if not chunks:
            raise DocumentError("Document produced no text chunks.", 422)
        embeddings = self.embedder.embed([c for _, c in chunks])
        self.store.add(doc_id, filename, chunks, embeddings)
        return {"doc_id": doc_id, "filename": filename, "status": "indexed",
                "pages": len(pages), "chunks": len(chunks)}

    # ---------- retrieval ----------
    def retrieve(self, question: str, top_k: int | None = None):
        hits = self.store.search(self.embedder.embed([question])[0], top_k or self.s.top_k)
        return [h for h in hits if h.score >= self.s.min_score]  # drop weak matches

    # ---------- answer ----------
    def ask(self, question: str, top_k: int | None = None) -> dict:
        t0 = time.perf_counter()
        hits = self.retrieve(question, top_k)
        if not hits:  # guardrail 1: nothing relevant -> refuse WITHOUT calling the LLM
            return self._result(ABSTAIN, False, [], t0)
        answer = self.generator.generate(question, hits)
        if answer.strip().upper().startswith(NOT_FOUND):  # guardrail 2: the LLM itself says context is insufficient
            return self._result(ABSTAIN, False, [], t0)
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
        used = [(i, h) for i, h in enumerate(hits, 1) if i in cited] or list(enumerate(hits, 1))
        sources = [{"id": i, "filename": h.filename, "page": h.page, "score": h.score,
                    "snippet": h.text[:300]} for i, h in used]
        return self._result(answer, True, sources, t0)

    @staticmethod
    def _result(answer, grounded, sources, t0):
        return {"answer": answer, "grounded": grounded, "sources": sources,
                "latency_ms": int((time.perf_counter() - t0) * 1000)}
