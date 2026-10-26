import io

import pytest
from fastapi.testclient import TestClient
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from app.config import Settings
from app.main import create_app

KEY = {"X-API-Key": "test-key"}


def make_pdf(pages: list[str]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    for text in pages:
        y = 800
        for line in text.split("\n"):
            c.drawString(40, y, line)
            y -= 16
        c.showPage()
    c.save()
    return buf.getvalue()


@pytest.fixture
def settings(tmp_path):
    return Settings(api_keys="test-key", embedding_backend="hash", chroma_path=str(tmp_path / "chroma"),
                    anthropic_api_key="", llm_provider="none", min_score=0.15, chunk_size=400, chunk_overlap=60)


@pytest.fixture
def client(settings):
    return TestClient(create_app(settings))


@pytest.fixture
def handbook_pdf():
    return make_pdf([
        "Refund policy: customers can get a full refund within 30 days of purchase.",
        "Warranty: every product includes a 2-year limited warranty against manufacturing defects.",
    ])
