"""Claude (Anthropic) API client wrapper."""

import anthropic


class ClaudeClient:
    """Claude API client for chat completions."""

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022") -> None:
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def chat(self, message: str, max_tokens: int = 1024) -> str:
        """Send a chat message and return the response text.

        Args:
            message: User message to send
            max_tokens: Maximum tokens in response

        Returns:
            Response text from Claude

        Raises:
            anthropic.RateLimitError: When rate limit is exceeded (429)
            anthropic.APIError: For other API errors
        """
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": message}],
        )
        return response.content[0].text
