import io

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from app.errors import DocumentError


def extract_pages(data: bytes) -> list[tuple[int, str]]:
    """Return [(page_number, text)] for pages that contain text (1-indexed)."""
    if not data.startswith(b"%PDF"):
        raise DocumentError("File is not a valid PDF.", 415)
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise DocumentError("PDF is password-protected.", 422)
        pages = [(i + 1, (p.extract_text() or "").strip()) for i, p in enumerate(reader.pages)]
    except DocumentError:
        raise
    except (PyPdfError, ValueError, KeyError) as exc:
        raise DocumentError(f"Corrupted or unreadable PDF: {exc}", 422) from exc
    pages = [(n, t) for n, t in pages if t]
    if not pages:
        raise DocumentError("No extractable text found (scanned PDF? OCR is not supported yet).", 422)
    return pages
