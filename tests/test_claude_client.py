"""Tests for Claude client."""

import unittest.mock as mock

import anthropic
import pytest

from src.claude_client import ClaudeClient


def test_claude_client_success():
    """Test successful Claude API call."""
    with mock.patch("anthropic.Anthropic") as mock_anthropic:
        mock_client = mock.Mock()
        mock_anthropic.return_value = mock_client

        mock_response = mock.Mock()
        mock_content = mock.Mock()
        mock_content.text = "Paris is the capital of France."
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response

        client = ClaudeClient(api_key="test-key")
        result = client.chat("What is the capital of France?")

        assert result == "Paris is the capital of France."
        mock_client.messages.create.assert_called_once_with(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "What is the capital of France?"}],
        )


def test_claude_client_rate_limit_error():
    """Test Claude client raises RateLimitError on 429."""
    with mock.patch("anthropic.Anthropic") as mock_anthropic:
        mock_client = mock.Mock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.side_effect = anthropic.RateLimitError(
            "Rate limit exceeded", response=mock.Mock(), body=None
        )

        client = ClaudeClient(api_key="test-key")

        with pytest.raises(anthropic.RateLimitError):
            client.chat("test message")


def test_claude_client_api_error():
    """Test Claude client raises APIError on other errors."""
    with mock.patch("anthropic.Anthropic") as mock_anthropic:
        mock_client = mock.Mock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.side_effect = anthropic.APIError(
            "API error", request=mock.Mock(), body=None
        )

        client = ClaudeClient(api_key="test-key")

        with pytest.raises(anthropic.APIError):
            client.chat("test message")
