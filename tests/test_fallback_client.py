"""Tests for fallback client."""

import unittest.mock as mock

import anthropic
import pytest

from src.fallback_client import FallbackChatClient


def test_fallback_client_primary_success():
    """Test fallback client uses primary (Claude) when successful."""
    with mock.patch("src.fallback_client.ClaudeClient") as mock_claude:
        with mock.patch("src.fallback_client.GeminiClient") as mock_gemini:
            mock_claude_instance = mock.Mock()
            mock_claude.return_value = mock_claude_instance
            mock_claude_instance.chat.return_value = "Claude response"

            client = FallbackChatClient(anthropic_api_key="claude-key", google_api_key="gemini-key")
            result = client.chat("test message")

            assert result == "Claude response"
            mock_claude_instance.chat.assert_called_once_with("test message", max_tokens=1024)
            mock_gemini.return_value.chat.assert_not_called()


def test_fallback_client_rate_limit_fallback():
    """Test fallback to Gemini on Claude rate limit (429)."""
    with mock.patch("src.fallback_client.ClaudeClient") as mock_claude:
        with mock.patch("src.fallback_client.GeminiClient") as mock_gemini:
            mock_claude_instance = mock.Mock()
            mock_claude.return_value = mock_claude_instance
            mock_claude_instance.chat.side_effect = anthropic.RateLimitError(
                "Rate limit", response=mock.Mock(), body=None
            )

            mock_gemini_instance = mock.Mock()
            mock_gemini.return_value = mock_gemini_instance
            mock_gemini_instance.chat.return_value = "Gemini response"

            client = FallbackChatClient(anthropic_api_key="claude-key", google_api_key="gemini-key")
            result = client.chat("test message")

            assert result == "Gemini response"
            mock_claude_instance.chat.assert_called_once()
            mock_gemini_instance.chat.assert_called_once_with("test message")


def test_fallback_client_api_error_fallback():
    """Test fallback to Gemini on Claude API error."""
    with mock.patch("src.fallback_client.ClaudeClient") as mock_claude:
        with mock.patch("src.fallback_client.GeminiClient") as mock_gemini:
            mock_claude_instance = mock.Mock()
            mock_claude.return_value = mock_claude_instance
            mock_claude_instance.chat.side_effect = anthropic.APIError(
                "API error", request=mock.Mock(), body=None
            )

            mock_gemini_instance = mock.Mock()
            mock_gemini.return_value = mock_gemini_instance
            mock_gemini_instance.chat.return_value = "Gemini fallback"

            client = FallbackChatClient(anthropic_api_key="claude-key", google_api_key="gemini-key")
            result = client.chat("test message")

            assert result == "Gemini fallback"
            mock_gemini_instance.chat.assert_called_once_with("test message")


def test_fallback_client_both_fail():
    """Test exception raised when both providers fail."""
    with mock.patch("src.fallback_client.ClaudeClient") as mock_claude:
        with mock.patch("src.fallback_client.GeminiClient") as mock_gemini:
            mock_claude_instance = mock.Mock()
            mock_claude.return_value = mock_claude_instance
            mock_claude_instance.chat.side_effect = anthropic.APIError(
                "Claude error", request=mock.Mock(), body=None
            )

            mock_gemini_instance = mock.Mock()
            mock_gemini.return_value = mock_gemini_instance
            mock_gemini_instance.chat.side_effect = Exception("Gemini error")

            client = FallbackChatClient(anthropic_api_key="claude-key", google_api_key="gemini-key")

            with pytest.raises(Exception, match="Gemini error"):
                client.chat("test message")


def test_fallback_client_generic_error_fallback():
    """Test fallback to Gemini on generic Claude exception."""
    with mock.patch("src.fallback_client.ClaudeClient") as mock_claude:
        with mock.patch("src.fallback_client.GeminiClient") as mock_gemini:
            mock_claude_instance = mock.Mock()
            mock_claude.return_value = mock_claude_instance
            mock_claude_instance.chat.side_effect = ValueError("Unexpected error")

            mock_gemini_instance = mock.Mock()
            mock_gemini.return_value = mock_gemini_instance
            mock_gemini_instance.chat.return_value = "Gemini rescue"

            client = FallbackChatClient(anthropic_api_key="claude-key", google_api_key="gemini-key")
            result = client.chat("test message")

            assert result == "Gemini rescue"
            mock_gemini_instance.chat.assert_called_once_with("test message")
