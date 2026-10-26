from dataclasses import dataclass

import chromadb


@dataclass
class Hit:
    text: str
    filename: str
    page: int
    doc_id: str
    chunk_index: int
    score: float  # cosine similarity, 0..1 (higher = more relevant)


class VectorStore:
    def __init__(self, path: str, name: str):
        self.client = chromadb.PersistentClient(path=path)
        self.col = self.client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})

    def has_document(self, doc_id: str) -> bool:
        return bool(self.col.get(where={"doc_id": doc_id}, limit=1)["ids"])

    def add(self, doc_id: str, filename: str, chunks: list[tuple[int, str]], embeddings: list[list[float]]) -> None:
        self.col.add(
            ids=[f"{doc_id}:{i}" for i in range(len(chunks))],
            documents=[c for _, c in chunks],
            embeddings=embeddings,
            metadatas=[{"doc_id": doc_id, "filename": filename, "page": p, "chunk_index": i}
                       for i, (p, _) in enumerate(chunks)],
        )

    def search(self, embedding: list[float], k: int) -> list[Hit]:
        n = self.col.count()
        if n == 0:
            return []
        res = self.col.query(query_embeddings=[embedding], n_results=min(k, n))
        hits = []
        for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            hits.append(Hit(text, meta["filename"], meta["page"], meta["doc_id"], meta["chunk_index"],
                            round(1 - dist, 4)))
        return hits

    def list_documents(self) -> list[dict]:
        docs: dict[str, dict] = {}
        for m in self.col.get(include=["metadatas"])["metadatas"]:
            d = docs.setdefault(m["doc_id"], {"doc_id": m["doc_id"], "filename": m["filename"], "chunks": 0})
            d["chunks"] += 1
        return list(docs.values())

    def delete_document(self, doc_id: str) -> int:
        ids = self.col.get(where={"doc_id": doc_id})["ids"]
        if ids:
            self.col.delete(ids=ids)
        return len(ids)
