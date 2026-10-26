import hashlib
import math
import re

STOPWORDS = {"the", "a", "an", "is", "are", "of", "to", "and", "in", "for", "on", "what", "how", "do", "does",
             "your", "you", "i", "my", "be", "it", "this", "that", "or", "with", "can", "we", "our", "at", "by"}


class HashEmbedder:
    """Deterministic bag-of-words embedder. For tests/offline eval only - no model download."""

    dim = 4096

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                if w in STOPWORDS:
                    continue
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()


def build_embedder(backend: str, model_name: str):
    if backend == "hash":
        return HashEmbedder()
    return SentenceTransformerEmbedder(model_name)
