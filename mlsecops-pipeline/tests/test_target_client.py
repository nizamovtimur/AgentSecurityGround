from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx

from boart.target_client import HttpTargetClient, extract_langflow_run_message, http_target_timeout_from_env


def test_extract_langflow_run_message_string() -> None:
    data = {
        "outputs": [
            {
                "outputs": [
                    {
                        "messages": [
                            {"message": "Привет, чем помочь?"},
                        ]
                    }
                ]
            }
        ]
    }
    assert extract_langflow_run_message(data) == "Привет, чем помочь?"


def test_extract_langflow_run_message_dict_text() -> None:
    data = {
        "outputs": [
            {
                "outputs": [
                    {
                        "messages": [
                            {"message": {"text": "nested reply", "sender": "Machine"}},
                        ]
                    }
                ]
            }
        ]
    }
    assert extract_langflow_run_message(data) == "nested reply"


def test_http_target_langflow_uses_run_payload() -> None:
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "outputs": [{"outputs": [{"messages": [{"message": "m"}]}]}],
    }
    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        c = HttpTargetClient(
            endpoint="http://localhost:7860/api/v1/run/flow-uuid",
            api_key="k",
            timeout_seconds=30.0,
        )
        out = c.send("hello", [])
        assert out == "m"
        _url, kwargs = mock_client.post.call_args
        assert kwargs["json"]["input_value"] == "hello"
        assert kwargs["json"]["output_type"] == "chat"
        assert kwargs["json"]["input_type"] == "chat"
        assert kwargs["headers"]["x-api-key"] == "k"
        assert mock_client_cls.call_args[1]["timeout"] == httpx.Timeout(30.0)


def test_http_target_generic_json_body() -> None:
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"response": "plain"}
    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        c = HttpTargetClient(endpoint="http://example.com/api/chat", timeout_seconds=30.0)
        out = c.send("x", [{"role": "user", "content": "a"}])
        assert out == "plain"
        _url, kwargs = mock_client.post.call_args
        assert kwargs["json"] == {"message": "x", "history": [{"role": "user", "content": "a"}]}
        assert mock_client_cls.call_args[1]["timeout"] == httpx.Timeout(30.0)


def test_http_target_timeout_env_precedence(monkeypatch) -> None:
    monkeypatch.setenv("MLSECOPS_TARGET_TIMEOUT", "99")
    monkeypatch.delenv("LANGFLOW_RUN_TIMEOUT", raising=False)
    monkeypatch.setenv("OPENAI_TIMEOUT", "1")
    assert http_target_timeout_from_env() == 99.0


def test_http_target_langflow_new_session_per_send() -> None:
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "outputs": [{"outputs": [{"messages": [{"message": "ok"}]}]}],
    }
    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        c = HttpTargetClient(endpoint="http://localhost:7860/api/v1/run/flow-uuid", timeout_seconds=30.0)
        c.send("hello", [])
        first = mock_client.post.call_args_list[0][1]["json"]["session_id"]
        c.send("again", [{"role": "user", "content": "hello"}])
        second = mock_client.post.call_args_list[1][1]["json"]["session_id"]

        assert first != second
