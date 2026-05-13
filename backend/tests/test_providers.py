from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.llm.providers import chat_completion


def test_chat_completion_vllm_posts_openai_shape_without_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_API_BASE", "http://vllm:8000/v1")
    monkeypatch.delenv("VLLM_API_KEY", raising=False)

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "  hello  "}}],
    }
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = False
    mock_client.post.return_value = mock_resp

    with patch("backend.llm.providers.httpx.Client", return_value=mock_client):
        out = chat_completion(
            provider="vllm",
            model="m1",
            messages=[{"role": "user", "content": "x"}],
            temperature=0.5,
            top_p=0.9,
        )

    assert out == "hello"
    mock_client.post.assert_called_once()
    call_kw = mock_client.post.call_args
    url = call_kw[0][0]
    assert url == "http://vllm:8000/v1/chat/completions"
    body = call_kw[1]["json"]
    assert body["model"] == "m1"
    assert body["messages"] == [{"role": "user", "content": "x"}]
    assert body["temperature"] == 0.5
    assert body["top_p"] == 0.9
    headers = call_kw[1]["headers"]
    assert "Authorization" not in headers


def test_chat_completion_vllm_sends_bearer_when_api_key_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_API_BASE", "http://localhost:8000/v1")
    monkeypatch.setenv("VLLM_API_KEY", "secret")

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = False
    mock_client.post.return_value = mock_resp

    with patch("backend.llm.providers.httpx.Client", return_value=mock_client):
        chat_completion(provider="vllm", model="m", messages=[{"role": "user", "content": "y"}])

    headers = mock_client.post.call_args[1]["headers"]
    assert headers["Authorization"] == "Bearer secret"


def test_chat_completion_unsupported_provider_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        chat_completion(provider="unknown", model="m", messages=[{"role": "user", "content": "z"}])
