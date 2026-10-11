"""Unit tests for invoke_bedrock function (task 2.2)."""

import json
import io
from unittest.mock import patch, MagicMock

import pytest
from botocore.exceptions import ClientError


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    """Set required environment variables before importing handler."""
    monkeypatch.setenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
    monkeypatch.setenv("BEDROCK_REGION", "eu-north-1")


@pytest.fixture
def mock_bedrock_client():
    """Mock the bedrock_runtime client."""
    with patch("src.handler.bedrock_runtime") as mock_client:
        yield mock_client


class TestInvokeBedrockSuccess:
    """Tests for successful Bedrock invocations."""

    def test_returns_generated_text(self, mock_bedrock_client):
        """invoke_bedrock returns text from Bedrock response content."""
        from src.handler import invoke_bedrock

        response_body = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Tesla is an electric vehicle company."}],
            "stop_reason": "end_turn"
        }
        mock_response = {
            "body": io.BytesIO(json.dumps(response_body).encode())
        }
        mock_bedrock_client.invoke_model.return_value = mock_response

        messages = [{"role": "user", "content": [{"type": "text", "text": "Tell me about Tesla"}]}]
        result = invoke_bedrock(messages, "You are a helpful assistant.")

        assert result == "Tesla is an electric vehicle company."

    def test_calls_invoke_model_with_correct_payload(self, mock_bedrock_client):
        """invoke_bedrock sends correct Claude Messages API payload."""
        from src.handler import invoke_bedrock

        response_body = {
            "content": [{"type": "text", "text": "Response"}]
        }
        mock_response = {
            "body": io.BytesIO(json.dumps(response_body).encode())
        }
        mock_bedrock_client.invoke_model.return_value = mock_response

        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
        system_prompt = "You are helpful."
        invoke_bedrock(messages, system_prompt)

        call_args = mock_bedrock_client.invoke_model.call_args
        body = json.loads(call_args.kwargs["body"])

        assert body["anthropic_version"] == "bedrock-2023-05-31"
        assert body["max_tokens"] == 1024
        assert body["system"] == "You are helpful."
        assert body["messages"] == messages
        assert call_args.kwargs["contentType"] == "application/json"
        assert call_args.kwargs["accept"] == "application/json"


class TestInvokeBedrockErrors:
    """Tests for error handling in invoke_bedrock."""

    def test_client_error_raises(self, mock_bedrock_client):
        """invoke_bedrock raises ClientError so retry wrapper can handle it."""
        from src.handler import invoke_bedrock

        error_response = {
            "Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}
        }
        mock_bedrock_client.invoke_model.side_effect = ClientError(
            error_response, "InvokeModel"
        )

        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]

        with pytest.raises(ClientError):
            invoke_bedrock(messages, "System prompt")

    def test_empty_content_returns_fallback_message(self, mock_bedrock_client):
        """invoke_bedrock returns fallback when content is empty list."""
        from src.handler import invoke_bedrock

        response_body = {"content": []}
        mock_response = {
            "body": io.BytesIO(json.dumps(response_body).encode())
        }
        mock_bedrock_client.invoke_model.return_value = mock_response

        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
        result = invoke_bedrock(messages, "System prompt")

        assert result == "I'm sorry, I didn't get a valid response. Please try again."

    def test_missing_content_key_returns_fallback_message(self, mock_bedrock_client):
        """invoke_bedrock returns fallback when content key is missing."""
        from src.handler import invoke_bedrock

        response_body = {"id": "msg_123", "type": "message"}
        mock_response = {
            "body": io.BytesIO(json.dumps(response_body).encode())
        }
        mock_bedrock_client.invoke_model.return_value = mock_response

        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
        result = invoke_bedrock(messages, "System prompt")

        assert result == "I'm sorry, I didn't get a valid response. Please try again."

    def test_non_retryable_error_handled_by_retry_wrapper(self, mock_bedrock_client):
        """invoke_bedrock_with_retry returns error message for non-retryable ClientError."""
        from src.handler import invoke_bedrock_with_retry

        error_response = {
            "Error": {"Code": "AccessDeniedException", "Message": "Not authorized"}
        }
        mock_bedrock_client.invoke_model.side_effect = ClientError(
            error_response, "InvokeModel"
        )

        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]

        with patch("src.handler.structured_logger") as mock_logger:
            result = invoke_bedrock_with_retry(messages, "System prompt")
            assert result == "I'm sorry, I couldn't process your request. Please try again."
            mock_logger.error.assert_called_once()
            call_kwargs = mock_logger.error.call_args
            assert call_kwargs.kwargs["error_type"] == "AccessDeniedException"
            assert call_kwargs.kwargs["error_message"] == "Not authorized"
