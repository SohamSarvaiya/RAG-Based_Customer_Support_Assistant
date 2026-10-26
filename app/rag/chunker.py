import re


def _tail(text: str, n: int) -> str:
    """Last ~n chars of text, cut at a word boundary (used as chunk overlap)."""
    if len(text) <= n:
        return text
    tail = text[-n:]
    i = tail.find(" ")
    return tail[i + 1:] if i != -1 else tail


def chunk_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    """Sentence-aware chunking with overlap so answers split across a boundary aren't lost."""
    if size <= overlap:
        raise ValueError("chunk size must be larger than overlap")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    units: list[str] = []
    for s in re.split(r"(?<=[.!?])\s+", text):
        units.extend([s[i:i + size] for i in range(0, len(s), size)] if len(s) > size else [s])

    chunks: list[str] = []
    cur = ""
    for u in units:
        if cur and len(cur) + 1 + len(u) > size:
            chunks.append(cur)
            cur = _tail(cur, overlap)
            if len(cur) + 1 + len(u) > size:
                cur = ""
        cur = f"{cur} {u}".strip()
    if cur:
        chunks.append(cur)
    return chunks


def chunk_pages(pages: list[tuple[int, str]], size: int, overlap: int) -> list[tuple[int, str]]:
    """Chunk each page separately so every chunk keeps its page number (for citations)."""
    return [(n, c) for n, t in pages for c in chunk_text(t, size, overlap)]
