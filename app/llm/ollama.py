"""Thin wrapper over the local Ollama server. One model, one request at a time."""

import threading

import httpx
import ollama

from app.config import Settings


class LLMError(Exception):
    pass


class OllamaLLM:
    def __init__(self, settings: Settings):
        self.model = settings.ollama_model
        self._client = ollama.Client(host=settings.ollama_host, timeout=settings.ollama_timeout_s)
        self._num_ctx = settings.ollama_num_ctx
        self._keep_alive = settings.ollama_keep_alive
        # CPU-only inference: never run two generations concurrently.
        self._lock = threading.Lock()

    def check(self) -> None:
        """Raise LLMError unless the server is up and the model is pulled."""
        try:
            names = {m.model for m in self._client.list().models}
        except (httpx.HTTPError, ollama.ResponseError, ConnectionError) as exc:
            raise LLMError(f"Ollama is not reachable: {exc}") from None
        if self.model not in names:
            raise LLMError(f"Model {self.model!r} is not pulled. Run: ollama pull {self.model}")

    def complete(self, prompt: str, max_tokens: int = 512) -> str:
        with self._lock:
            try:
                response = self._client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0, "num_ctx": self._num_ctx, "num_predict": max_tokens},
                    keep_alive=self._keep_alive,
                )
            except (httpx.HTTPError, ollama.ResponseError, ConnectionError) as exc:
                raise LLMError(f"Ollama request failed: {exc}") from None
        return response.message.content or ""
