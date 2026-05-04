"""Multi-LLM fallback client with Claude primary and Gemini secondary."""

import logging

import anthropic

from src.claude_client import ClaudeClient
from src.gemini_client import GeminiClient

logger = logging.getLogger(__name__)


class FallbackChatClient:
    """Chat client with automatic fallback from Claude to Gemini."""

    def __init__(
        self,
        anthropic_api_key: str,
        google_api_key: str,
        claude_model: str = "claude-3-5-sonnet-20241022",
        gemini_model: str = "gemini-1.5-flash",
    ) -> None:
        """Initialize fallback client with both providers.

        Args:
            anthropic_api_key: API key for Claude
            google_api_key: API key for Gemini
            claude_model: Claude model name (default: claude-3-5-sonnet-20241022)
            gemini_model: Gemini model name (default: gemini-1.5-flash)
        """
        self.primary = ClaudeClient(api_key=anthropic_api_key, model=claude_model)
        self.fallback = GeminiClient(api_key=google_api_key, model=gemini_model)

    def chat(self, message: str, max_tokens: int = 1024) -> str:
        """Send a chat message with automatic fallback.

        Tries Claude first. If rate-limited (429) or other errors occur,
        automatically falls back to Gemini.

        Args:
            message: User message to send
            max_tokens: Maximum tokens in response (Claude only)

        Returns:
            Response text from either Claude or Gemini

        Raises:
            Exception: If both providers fail
        """
        try:
            logger.info("Attempting primary provider (Claude)")
            return self.primary.chat(message, max_tokens=max_tokens)
        except anthropic.RateLimitError as e:
            logger.warning(f"Claude rate limit exceeded: {e}. Falling back to Gemini.")
            return self._fallback_chat(message)
        except anthropic.APIError as e:
            logger.warning(f"Claude API error: {e}. Falling back to Gemini.")
            return self._fallback_chat(message)
        except Exception as e:
            logger.warning(f"Unexpected error with Claude: {e}. Falling back to Gemini.")
            return self._fallback_chat(message)

    def _fallback_chat(self, message: str) -> str:
        """Internal method to call fallback provider.

        Args:
            message: User message to send

        Returns:
            Response text from Gemini

        Raises:
            Exception: If fallback provider also fails
        """
        logger.info("Using fallback provider (Gemini)")
        return self.fallback.chat(message)
