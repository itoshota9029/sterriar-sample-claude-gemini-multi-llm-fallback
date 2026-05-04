"""Gemini (Google Generative AI) client wrapper."""

import google.generativeai as genai


class GeminiClient:
    """Gemini API client for chat completions."""

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash") -> None:
        genai.configure(api_key=api_key)
        self.model_name = model
        self.model = genai.GenerativeModel(model)

    def chat(self, message: str) -> str:
        """Send a chat message and return the response text.

        Args:
            message: User message to send

        Returns:
            Response text from Gemini

        Raises:
            Exception: For any API errors
        """
        response = self.model.generate_content(message)
        return response.text
