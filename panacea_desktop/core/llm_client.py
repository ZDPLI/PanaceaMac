from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

import requests


class LLMError(RuntimeError):
    pass


@dataclass
class ProviderConfig:
    name: str  # novita|openrouter|custom
    base_url: str
    api_key: str
    extra_headers: dict[str, str] | None = None


def _normalize_base_url(base_url: str) -> str:
    base_url = (base_url or "").strip()
    if not base_url:
        return ""
    return base_url.rstrip("/")


class OpenAICompatibleClient:
    """Minimal OpenAI-compatible /chat/completions client."""

    def __init__(self, cfg: ProviderConfig, timeout: int = 120):
        self.cfg = ProviderConfig(
            name=cfg.name,
            base_url=_normalize_base_url(cfg.base_url),
            api_key=(cfg.api_key or "").strip(),
            extra_headers=cfg.extra_headers or {},
        )
        self.timeout = timeout

    def chat_completions(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 1200,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        if not self.cfg.base_url:
            raise LLMError(f"Provider '{self.cfg.name}': base_url is empty")
        if not self.cfg.api_key:
            raise LLMError(f"Provider '{self.cfg.name}': api_key is empty")

        url = f"{self.cfg.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self.cfg.extra_headers or {})

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if extra_body:
            payload.update(extra_body)

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            resp.encoding = 'utf-8'
        except Exception as e:
            raise LLMError(f"Provider '{self.cfg.name}' request failed: {e}") from e

        if resp.status_code >= 400:
            try:
                err = resp.json()
            except Exception:
                err = resp.text
            raise LLMError(f"Provider '{self.cfg.name}' HTTP {resp.status_code}: {err}")

        try:
            data = resp.json()
        except Exception as e:
            raise LLMError(f"Provider '{self.cfg.name}' invalid JSON response: {e}\n{resp.text[:1000]}") from e

        try:
            return (data["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            raise LLMError(f"Provider '{self.cfg.name}' unexpected response schema: {json.dumps(data)[:1200]}")

    def chat_completions_stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 1200,
        extra_body: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """Yield incremental text deltas using OpenAI-compatible SSE streaming.

        Note: In many providers the endpoint is the same; `stream: true` enables SSE.
        """
        if not self.cfg.base_url:
            raise LLMError(f"Provider '{self.cfg.name}': base_url is empty")
        if not self.cfg.api_key:
            raise LLMError(f"Provider '{self.cfg.name}': api_key is empty")

        url = f"{self.cfg.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        headers.update(self.cfg.extra_headers or {})

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if extra_body:
            payload.update(extra_body)

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout, stream=True)
            resp.encoding = 'utf-8'
        except Exception as e:
            raise LLMError(f"Provider '{self.cfg.name}' request failed: {e}") from e

        if resp.status_code >= 400:
            try:
                err = resp.json()
            except Exception:
                err = resp.text
            raise LLMError(f"Provider '{self.cfg.name}' HTTP {resp.status_code}: {err}")

        def _iter_lines() -> Iterable[str]:
            for raw in resp.iter_lines(decode_unicode=False):
                if raw is None:
                    continue
                # iter_lines returns bytes when decode_unicode=False
                if isinstance(raw, bytes):
                    raw = raw.decode('utf-8', errors='replace')
                line = str(raw).strip()
                if not line:
                    continue
                yield line

        for line in _iter_lines():
            if not line.startswith("data:"):
                continue
            data_str = line[len("data:") :].strip()
            if data_str == "[DONE]":
                break
            try:
                evt = json.loads(data_str)
            except Exception:
                continue
            try:
                delta = evt["choices"][0].get("delta", {})
                piece = delta.get("content")
                if piece:
                    yield piece
            except Exception:
                continue

    def audio_transcriptions(
        self,
        *,
        model: str,
        file_path: str,
        language: str | None = None,
        prompt: str | None = None,
        response_format: str = "json",
        temperature: float | None = None,
    ) -> str:
        """OpenAI-compatible audio transcription.

        Many providers implement POST /v1/audio/transcriptions.
        We call {base_url}/audio/transcriptions assuming base_url already ends with /v1.
        """
        if not self.cfg.base_url:
            raise LLMError(f"Provider '{self.cfg.name}': base_url is empty")
        if not self.cfg.api_key:
            raise LLMError(f"Provider '{self.cfg.name}': api_key is empty")

        url = f"{self.cfg.base_url}/audio/transcriptions"
        headers = {"Authorization": f"Bearer {self.cfg.api_key}"}
        headers.update(self.cfg.extra_headers or {})

        data: dict[str, Any] = {"model": model, "response_format": response_format}
        if language:
            data["language"] = language
        if prompt:
            data["prompt"] = prompt
        if temperature is not None:
            data["temperature"] = temperature

        try:
            with open(file_path, "rb") as f:
                files = {"file": (file_path.split("/")[-1], f)}
                resp = requests.post(url, headers=headers, data=data, files=files, timeout=self.timeout)
                resp.encoding = "utf-8"
        except Exception as e:
            raise LLMError(f"Provider '{self.cfg.name}' STT request failed: {e}") from e

        if resp.status_code >= 400:
            try:
                err = resp.json()
            except Exception:
                err = resp.text
            raise LLMError(f"Provider '{self.cfg.name}' STT HTTP {resp.status_code}: {err}")

        try:
            j = resp.json()
        except Exception as e:
            # Some providers might return plain text.
            txt = (resp.text or "").strip()
            if txt:
                return txt
            raise LLMError(f"Provider '{self.cfg.name}' STT invalid response: {e}") from e

        # OpenAI schema: {"text": "..."}
        if isinstance(j, dict) and "text" in j:
            return (j.get("text") or "").strip()
        # Fallback for other schemas
        if isinstance(j, dict) and "data" in j and isinstance(j["data"], str):
            return j["data"].strip()
        return json.dumps(j, ensure_ascii=False)