import httpx
import pytest

from app.errors import LLMError
from app.rag.embeddings import HashEmbedder
from app.rag.generator import NOT_FOUND, Generator, build_prompt
from app.rag.pipeline import RAGPipeline
from app.rag.vectorstore import VectorStore
from tests.conftest import make_pdf


class FakeLLM(Generator):
    def __init__(self, reply):
        self.reply = reply

    def generate(self, question, hits):
        return self.reply


def make_pipe(settings, reply):
    return RAGPipeline(settings, HashEmbedder(), VectorStore(settings.chroma_path, "test_col"), FakeLLM(reply))


def test_llm_not_found_triggers_abstention(settings):
    p = make_pipe(settings, NOT_FOUND)
    p.ingest_pdf("a.pdf", make_pdf(["Refund within 30 days of purchase."]))
    r = p.ask("refund policy?")
    assert r["grounded"] is False and r["sources"] == []


def test_only_cited_sources_returned(settings):
    p = make_pipe(settings, "Refunds are possible within 30 days [1].")
    p.ingest_pdf("a.pdf", make_pdf(["Refund within 30 days of purchase."]))
    r = p.ask("refund policy?")
    assert r["grounded"] and [s["id"] for s in r["sources"]] == [1]


def test_prompt_contains_numbered_sources(settings):
    p = make_pipe(settings, "x")
    p.ingest_pdf("a.pdf", make_pdf(["Refund within 30 days of purchase."]))
    prompt = build_prompt("q?", p.retrieve("refund"))
    assert "[1] (source: a.pdf, page 1)" in prompt


def test_ollama_provider_calls_local_server(monkeypatch, settings):
    import httpx

    class R:
        def raise_for_status(self): pass
        def json(self): return {"message": {"content": "30 days [1]"}}

    seen = {}
    monkeypatch.setattr(httpx, "post", lambda url, **kw: seen.update(url=url, **kw) or R())
    g = Generator(provider="ollama", ollama_model="llama3.2:3b")
    p = RAGPipeline(settings, HashEmbedder(), VectorStore(settings.chroma_path, "test_col"), g)
    p.ingest_pdf("a.pdf", make_pdf(["Refund within 30 days of purchase."]))
    r = p.ask("refund policy?")
    assert r["grounded"] and seen["url"].endswith("/api/chat") and seen["json"]["model"] == "llama3.2:3b"


class _Resp:
    def __init__(self, status=200):
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=httpx.Request("POST", "http://x"),
                                        response=httpx.Response(self.status_code))

    def json(self):
        return {"choices": [{"message": {"content": "30 days [1]"}}]}


def _hits(settings):
    p = make_pipe(settings, "x")
    p.ingest_pdf("a.pdf", make_pdf(["Refund within 30 days of purchase."]))
    return p.retrieve("refund")


def test_openai_provider(monkeypatch, settings):
    seen = {}
    monkeypatch.setattr(httpx, "post", lambda url, **kw: seen.update(url=url, **kw) or _Resp())
    g = Generator("openai", openai_key="sk-test", openai_model="gpt-4o-mini")
    assert g.generate("refund?", _hits(settings)) == "30 days [1]"
    assert seen["url"] == "https://api.openai.com/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer sk-test"


def test_gemini_provider_uses_openai_compatible_endpoint(monkeypatch, settings):
    seen = {}
    monkeypatch.setattr(httpx, "post", lambda url, **kw: seen.update(url=url, **kw) or _Resp())
    Generator("gemini", gemini_key="g-key").generate("refund?", _hits(settings))
    assert "generativelanguage.googleapis.com" in seen["url"] and seen["json"]["model"] == "gemini-2.5-flash"


def test_auto_picks_provider_from_available_key():
    assert Generator("auto").provider == "none"
    assert Generator("auto", openai_key="k").provider == "openai"
    assert Generator("auto", gemini_key="k").provider == "gemini"


def test_provider_without_key_fails_fast():
    with pytest.raises(ValueError):
        Generator("openai")


def test_retries_without_temperature_on_400(monkeypatch, settings):
    calls = []
    monkeypatch.setattr(httpx, "post", lambda url, **kw: calls.append(dict(kw["json"])) or _Resp(400 if len(calls) == 1 else 200))
    Generator("openai", openai_key="k").generate("q", _hits(settings))
    assert "temperature" in calls[0] and "temperature" not in calls[1]


def test_bad_key_gives_friendly_error(monkeypatch, settings):
    monkeypatch.setattr(httpx, "post", lambda url, **kw: _Resp(401))
    with pytest.raises(LLMError, match="rejected the API key"):
        Generator("gemini", gemini_key="bad").generate("q", _hits(settings))
