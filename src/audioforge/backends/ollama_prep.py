"""Ollama-backed text preparation for audiobook narration."""

from __future__ import annotations

from typing import Any

import httpx

from audioforge.models import BuildOptions

SYSTEM_PROMPT = (
    "You clean fiction chapter text for audiobook narration. "
    "Do not change meaning, plot, or dialogue content. "
    "Remove site chrome, navigation, ads, author notes that are not part of "
    "the story, and HTML/markdown artifacts. "
    "Output plain text only — no commentary, no markdown fences, no preamble."
)


class OllamaPrepError(Exception):
    """Ollama request failed or returned an unusable response."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        body: str = "",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class OllamaTextPrep:
    """Call a local Ollama chat model to clean chapter text."""

    def __init__(
        self,
        base_url: str,
        *,
        client: httpx.Client | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client(timeout=timeout)

    def close(self) -> None:
        """Close the underlying client when this instance created it."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> OllamaTextPrep:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def prepare(self, text: str, *, options: BuildOptions) -> str:
        """Return speech-ready text via Ollama ``/api/chat``."""
        url = f"{self._base_url}/api/chat"
        payload: dict[str, Any] = {
            "model": options.prep_model,
            "stream": False,
            "options": {"temperature": 0.2},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        }
        try:
            response = self._client.post(url, json=payload)
        except httpx.HTTPError as exc:
            raise OllamaPrepError(
                f"Ollama request failed: {exc}",
            ) from exc

        if response.status_code >= 400:
            raise OllamaPrepError(
                f"Ollama HTTP {response.status_code}: {response.text[:500]}",
                status_code=response.status_code,
                body=response.text,
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise OllamaPrepError(
                "Ollama returned invalid JSON",
                status_code=response.status_code,
                body=response.text,
            ) from exc

        content = _extract_message_content(data)
        if content is None:
            raise OllamaPrepError(
                "Ollama response missing message content",
                status_code=response.status_code,
                body=response.text,
            )
        cleaned = content.strip()
        if cleaned:
            return cleaned + "\n"
        return ""


def _extract_message_content(data: object) -> str | None:
    if not isinstance(data, dict):
        return None
    message = data.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    # Fallback: /api/generate style
    response = data.get("response")
    if isinstance(response, str):
        return response
    return None
