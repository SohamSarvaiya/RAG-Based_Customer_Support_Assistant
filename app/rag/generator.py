import logging

from app.errors import LLMError
from app.rag.vectorstore import Hit

log = logging.getLogger("rag.generator")

OPENAI_URL = "https://api.openai.com/v1"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai"  # Gemini's OpenAI-compatible API

ABSTAIN = ("I couldn't find this in the uploaded documents. "
           "Please rephrase your question or contact a human support agent.")
NOT_FOUND = "NOT_FOUND"

SYSTEM_PROMPT = f"""You are a customer support assistant. Answer ONLY using the numbered context passages provided.
Rules:
1. Never use outside knowledge. Do not guess or invent policies, numbers, dates or steps.
2. If the context does not contain the answer, reply with exactly: {NOT_FOUND}
3. After every claim, cite the passage(s) it came from like [1] or [2][3].
4. Be concise and use the customer's language."""


def build_prompt(question: str, hits: list[Hit]) -> str:
    ctx = "\n\n".join(f"[{i}] (source: {h.filename}, page {h.page})\n{h.text}" for i, h in enumerate(hits, 1))
    return f"Context passages:\n\n{ctx}\n\nQuestion: {question}"


class Generator:
    """LLM providers: anthropic | openai | gemini | ollama (free, local) | none (retrieval-only).
    provider="auto" picks the first provider that has an API key (anthropic, openai, gemini), else none."""

    def __init__(self, provider: str = "none", *, anthropic_key: str = "", anthropic_model: str = "claude-sonnet-5-5",
                 openai_key: str = "", openai_model: str = "gpt-4o-mini",
                 gemini_key: str = "", gemini_model: str = "gemini-2.5-flash",
                 ollama_url: str = "http://localhost:11434", ollama_model: str = "llama3.2:3b"):
        if provider == "auto":
            provider = ("anthropic" if anthropic_key else "openai" if openai_key
                        else "gemini" if gemini_key else "none")
        keys = {"anthropic": anthropic_key, "openai": openai_key, "gemini": gemini_key}
        if provider in keys and not keys[provider]:
            raise ValueError(f"LLM_PROVIDER={provider} requires its API key in .env")
        if provider not in {"anthropic", "openai", "gemini", "ollama", "none"}:
            raise ValueError(f"Unknown LLM_PROVIDER: {provider}")
        self.provider = provider
        self.anthropic_model, self.openai_key, self.openai_model = anthropic_model, openai_key, openai_model
        self.gemini_key, self.gemini_model = gemini_key, gemini_model
        self.ollama_url, self.ollama_model = ollama_url.rstrip("/"), ollama_model
        self.client = None
        if provider == "anthropic":
            import anthropic

            self.client = anthropic.Anthropic(api_key=anthropic_key)

    @classmethod
    def from_settings(cls, s) -> "Generator":
        return cls(s.llm_provider, anthropic_key=s.anthropic_api_key, anthropic_model=s.llm_model,
                   openai_key=s.openai_api_key, openai_model=s.openai_model,
                   gemini_key=s.gemini_api_key, gemini_model=s.gemini_model,
                   ollama_url=s.ollama_url, ollama_model=s.ollama_model)

    def generate(self, question: str, hits: list[Hit]) -> str:
        if self.provider == "none":  # no LLM: return the best passage with its citation
            return f"{hits[0].text} [1]"
        try:
            if self.provider == "anthropic":
                msg = self.client.messages.create(
                    model=self.anthropic_model, max_tokens=600, temperature=0, system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": build_prompt(question, hits)}])
                return "".join(b.text for b in msg.content if b.type == "text").strip()
            if self.provider == "openai":
                return self._chat_completions(OPENAI_URL, self.openai_key, self.openai_model, question, hits)
            if self.provider == "gemini":
                return self._chat_completions(GEMINI_URL, self.gemini_key, self.gemini_model, question, hits)
            return self._ollama(question, hits)
        except Exception as exc:
            log.exception("LLM call failed (%s)", self.provider)
            raise LLMError(self._friendly(exc)) from exc

    def _friendly(self, exc: Exception) -> str:
        code = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
        name = self.provider
        if code in (401, 403):
            return f"{name} rejected the API key. Check the key in .env."
        if code == 404:
            return f"{name}: model not found. Check the model name in .env."
        if code == 429:
            return f"{name}: rate limit or quota exceeded (check billing / free-tier limits)."
        if name == "ollama":
            return "Could not reach Ollama. Is it running (`ollama serve`) and the model pulled?"
        return f"The {name} language model is unavailable. Try again."

    def _chat_completions(self, base_url: str, key: str, model: str, question: str, hits: list[Hit]) -> str:
        """OpenAI-compatible /chat/completions (works for OpenAI and Gemini)."""
        import httpx

        payload = {"model": model, "temperature": 0,
                   "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": build_prompt(question, hits)}]}
        headers = {"Authorization": f"Bearer {key}"}
        r = httpx.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=60)
        if r.status_code == 400:  # some newer models only accept the default temperature
            payload.pop("temperature")
            r = httpx.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    def _ollama(self, question: str, hits: list[Hit]) -> str:
        import httpx

        r = httpx.post(f"{self.ollama_url}/api/chat", timeout=180, json={
            "model": self.ollama_model, "stream": False, "options": {"temperature": 0},
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": build_prompt(question, hits)}]})
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
