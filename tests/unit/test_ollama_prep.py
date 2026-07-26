"""Tests for OllamaTextPrep with httpx MockTransport."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from audioforge.backends.ollama_prep import (
    SYSTEM_PROMPT,
    OllamaPrepError,
    OllamaTextPrep,
)
from audioforge.models import BuildOptions


def _options(model: str = "llama3.2:3b") -> BuildOptions:
    return BuildOptions(source=".", prep_model=model)


def _chat_handler(
    *,
    status: int = 200,
    body: dict[str, Any] | list[Any] | str | None = None,
    raise_connect: bool = False,
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if raise_connect:
            raise httpx.ConnectError("connection refused", request=request)
        if body is None:
            content = {
                "message": {"role": "assistant", "content": "Cleaned chapter.\n"},
                "done": True,
            }
            return httpx.Response(status, json=content)
        if isinstance(body, str):
            return httpx.Response(status, text=body)
        return httpx.Response(status, json=body)

    return httpx.MockTransport(handler)


def test_prepare_success_uses_chat_api_and_model() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["payload"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "  Narration ready.  "}},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    prep = OllamaTextPrep("http://127.0.0.1:11434/", client=client)
    result = prep.prepare("raw chapter", options=_options("my-model"))

    assert result == "Narration ready.\n"
    assert seen["url"] == "http://127.0.0.1:11434/api/chat"
    assert seen["payload"]["model"] == "my-model"
    assert seen["payload"]["stream"] is False
    assert seen["payload"]["messages"][0]["role"] == "system"
    assert SYSTEM_PROMPT in seen["payload"]["messages"][0]["content"]
    assert seen["payload"]["messages"][1]["content"] == "raw chapter"
    assert seen["payload"]["options"]["temperature"] == 0.2
    client.close()


def test_prepare_http_error() -> None:
    client = httpx.Client(
        transport=_chat_handler(status=500, body="internal error"),
    )
    prep = OllamaTextPrep("http://localhost:11434", client=client)
    with pytest.raises(OllamaPrepError, match="HTTP 500") as exc_info:
        prep.prepare("x", options=_options())
    assert exc_info.value.status_code == 500
    assert "internal error" in exc_info.value.body
    client.close()


def test_prepare_invalid_json() -> None:
    client = httpx.Client(transport=_chat_handler(status=200, body="not-json{"))
    prep = OllamaTextPrep("http://localhost:11434", client=client)
    with pytest.raises(OllamaPrepError, match="invalid JSON"):
        prep.prepare("x", options=_options())
    client.close()


def test_prepare_missing_content() -> None:
    client = httpx.Client(transport=_chat_handler(status=200, body={"done": True}))
    prep = OllamaTextPrep("http://localhost:11434", client=client)
    with pytest.raises(OllamaPrepError, match="missing message content"):
        prep.prepare("x", options=_options())
    client.close()


def test_prepare_connect_error() -> None:
    client = httpx.Client(transport=_chat_handler(raise_connect=True))
    prep = OllamaTextPrep("http://localhost:11434", client=client)
    with pytest.raises(OllamaPrepError, match="request failed"):
        prep.prepare("x", options=_options())
    client.close()


def test_prepare_generate_style_response_fallback() -> None:
    client = httpx.Client(
        transport=_chat_handler(status=200, body={"response": "from generate"}),
    )
    prep = OllamaTextPrep("http://localhost:11434", client=client)
    assert prep.prepare("x", options=_options()) == "from generate\n"
    client.close()


def test_prepare_empty_content() -> None:
    client = httpx.Client(
        transport=_chat_handler(
            status=200,
            body={"message": {"role": "assistant", "content": "   "}},
        ),
    )
    prep = OllamaTextPrep("http://localhost:11434", client=client)
    assert prep.prepare("x", options=_options()) == ""
    client.close()


def test_prepare_non_dict_json() -> None:
    client = httpx.Client(transport=_chat_handler(status=200, body=["list"]))
    prep = OllamaTextPrep("http://localhost:11434", client=client)
    with pytest.raises(OllamaPrepError, match="missing message content"):
        prep.prepare("x", options=_options())
    client.close()


def test_owns_client_context_manager() -> None:
    transport = _chat_handler()
    client = httpx.Client(transport=transport)
    with OllamaTextPrep("http://localhost:11434", client=client) as prep:
        assert prep.prepare("hi", options=_options()) == "Cleaned chapter.\n"
    client.close()


def test_default_client_created_and_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    real_client = httpx.Client
    created: list[httpx.Client] = []

    def fake_client(**kwargs: object) -> httpx.Client:
        del kwargs
        client = real_client(transport=_chat_handler())
        created.append(client)
        return client

    monkeypatch.setattr("audioforge.backends.ollama_prep.httpx.Client", fake_client)
    prep = OllamaTextPrep("http://localhost:11434")
    assert len(created) == 1
    assert prep.prepare("hi", options=_options()) == "Cleaned chapter.\n"
    with prep:
        pass  # __exit__ closes owned client


def test_injected_client_not_closed_by_close() -> None:
    client = httpx.Client(transport=_chat_handler())
    prep = OllamaTextPrep("http://localhost:11434", client=client)
    prep.close()
    # Injected client still usable
    assert prep.prepare("x", options=_options()) == "Cleaned chapter.\n"
    client.close()


def test_ollama_prep_error_defaults() -> None:
    err = OllamaPrepError("x")
    assert err.status_code is None
    assert err.body == ""


def test_message_content_non_string() -> None:
    client = httpx.Client(
        transport=_chat_handler(
            status=200,
            body={"message": {"role": "assistant", "content": 123}},
        ),
    )
    prep = OllamaTextPrep("http://localhost:11434", client=client)
    with pytest.raises(OllamaPrepError, match="missing message content"):
        prep.prepare("x", options=_options())
    client.close()
