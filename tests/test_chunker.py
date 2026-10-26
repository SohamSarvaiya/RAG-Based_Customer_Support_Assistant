import pytest

from app.rag.chunker import chunk_pages, chunk_text


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"Sentence number {i} is here." for i in range(60))
    chunks = chunk_text(text, size=200, overlap=40)
    assert len(chunks) > 3
    assert all(len(c) <= 200 for c in chunks)
    assert chunks[0].split()[-2:] == chunks[1].split()[:len(chunks[0].split()[-2:])] or any(
        w in chunks[1] for w in chunks[0].split()[-3:])


def test_empty_text_returns_nothing():
    assert chunk_text("   \n ") == []


def test_very_long_sentence_is_split():
    chunks = chunk_text("x" * 1000, size=300, overlap=50)
    assert len(chunks) == 4 and all(len(c) <= 300 for c in chunks)


def test_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text("hello", size=50, overlap=50)


def test_page_numbers_preserved():
    out = chunk_pages([(1, "Alpha beta."), (3, "Gamma delta.")], 100, 10)
    assert [p for p, _ in out] == [1, 3]
