"""Basic evaluation of the RAG pipeline.

Metrics
  * Retrieval hit-rate : for answerable questions, do the retrieved chunks contain the expected facts
                         (and come from the expected page)?
  * Abstention accuracy: for out-of-scope questions, does the bot refuse instead of hallucinating?
  * Answer check       : (only if an LLM is configured) does the final answer contain the facts?

Run:  python -m eval.run_eval
"""
import json
import sys
import tempfile
from pathlib import Path

from app.config import Settings
from app.rag.embeddings import build_embedder
from app.rag.generator import Generator
from app.rag.pipeline import RAGPipeline
from app.rag.vectorstore import VectorStore

ROOT = Path(__file__).parent.parent


def main() -> int:
    s = Settings(chroma_path=tempfile.mkdtemp())
    pipe = RAGPipeline(s, build_embedder(s.embedding_backend, s.embedding_model),
                       VectorStore(s.chroma_path, "eval"), Generator.from_settings(s))
    for pdf in (ROOT / "sample_docs").glob("*.pdf"):
        pipe.ingest_pdf(pdf.name, pdf.read_bytes())

    data = json.loads((ROOT / "eval" / "dataset.json").read_text())
    use_llm = pipe.generator.provider != "none"
    rows, hit, page_ok, abst, ans_ok = [], 0, 0, 0, 0
    n_ans = sum(d["answerable"] for d in data)
    n_unans = len(data) - n_ans

    for d in data:
        hits = pipe.retrieve(d["question"])
        if d["answerable"]:
            text = " ".join(h.text for h in hits).lower()
            ok = all(k.lower() in text for k in d["keywords"])
            pg = any(h.page == d["page"] for h in hits)
            hit += ok
            page_ok += pg
            status = "PASS" if ok else "FAIL"
            if use_llm:
                a = pipe.ask(d["question"])
                a_ok = a["grounded"] and any(k.lower() in a["answer"].lower() for k in d["keywords"])
                ans_ok += a_ok
                status += f" | answer {'PASS' if a_ok else 'FAIL'}"
        else:
            refused = not hits if not use_llm else not pipe.ask(d["question"])["grounded"]
            abst += refused
            status = "PASS (refused)" if refused else "FAIL (hallucination risk)"
        rows.append((status, d["question"]))

    for status, q in rows:
        print(f"{status:<28} {q}")
    summary = {
        "embedding_backend": s.embedding_backend, "min_score": s.min_score,
        "retrieval_hit_rate": round(hit / n_ans, 2), "page_match_rate": round(page_ok / n_ans, 2),
        "abstention_accuracy": round(abst / n_unans, 2),
    }
    if use_llm:
        summary["answer_accuracy"] = round(ans_ok / n_ans, 2)
    print("\n" + json.dumps(summary, indent=2))
    (ROOT / "eval" / "results.json").write_text(json.dumps(summary, indent=2))
    return 0 if summary["retrieval_hit_rate"] >= 0.8 and summary["abstention_accuracy"] >= 0.75 else 1


if __name__ == "__main__":
    sys.exit(main())
