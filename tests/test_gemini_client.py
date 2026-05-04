"""Tests for Gemini client."""

import unittest.mock as mock

import pytest

from src.gemini_client import GeminiClient


def test_gemini_client_success():
    """Test successful Gemini API call."""
    with mock.patch("google.generativeai.configure") as mock_configure:
        with mock.patch("google.generativeai.GenerativeModel") as mock_model_class:
            mock_model = mock.Mock()
            mock_model_class.return_value = mock_model

            mock_response = mock.Mock()
            mock_response.text = "Tokyo is the capital of Japan."
            mock_model.generate_content.return_value = mock_response

            client = GeminiClient(api_key="test-key")
            result = client.chat("What is the capital of Japan?")

            assert result == "Tokyo is the capital of Japan."
            mock_configure.assert_called_once_with(api_key="test-key")
            mock_model.generate_content.assert_called_once_with("What is the capital of Japan?")


def test_gemini_client_api_error():
    """Test Gemini client raises exception on API error."""
    with mock.patch("google.generativeai.configure"):
        with mock.patch("google.generativeai.GenerativeModel") as mock_model_class:
            mock_model = mock.Mock()
            mock_model_class.return_value = mock_model
            mock_model.generate_content.side_effect = Exception("API error")

            client = GeminiClient(api_key="test-key")

            with pytest.raises(Exception, match="API error"):
                client.chat("test message")


def test_gemini_client_custom_model():
    """Test Gemini client with custom model name."""
    with mock.patch("google.generativeai.configure"):
        with mock.patch("google.generativeai.GenerativeModel") as mock_model_class:
            mock_model = mock.Mock()
            mock_model_class.return_value = mock_model

            mock_response = mock.Mock()
            mock_response.text = "Response"
            mock_model.generate_content.return_value = mock_response

            client = GeminiClient(api_key="test-key", model="gemini-pro")

            assert client.model_name == "gemini-pro"
            mock_model_class.assert_called_once_with("gemini-pro")
