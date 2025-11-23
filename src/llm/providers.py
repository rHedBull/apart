"""LLM provider implementations for various services."""

import os
from typing import Optional
import google.generativeai as genai
from llm.llm_provider import LLMProvider


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider implementation."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-1.5-flash"):
        """
        Initialize Gemini provider.

        Args:
            api_key: Google API key. If None, will try to get from GEMINI_API_KEY env var
            model_name: Gemini model to use (default: gemini-1.5-flash - free tier)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self._model = None

        if self.api_key:
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(self.model_name)

    def is_available(self) -> bool:
        """Check if Gemini is properly configured."""
        return self.api_key is not None and self._model is not None

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate a response using Google Gemini.

        Args:
            prompt: The user prompt/message
            system_prompt: Optional system instruction (Gemini supports this natively)

        Returns:
            The generated response text

        Raises:
            ValueError: If the provider is not properly configured
            Exception: If the API request fails
        """
        if not self.is_available():
            raise ValueError(
                "Gemini provider not configured. Please set GEMINI_API_KEY environment variable."
            )

        try:
            # Combine system prompt with user prompt if provided
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            response = self._model.generate_content(full_prompt)
            return response.text

        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}") from e


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider implementation."""

    def __init__(
        self,
        model: str = "llama2",
        base_url: Optional[str] = None
    ):
        """
        Initialize Ollama provider.

        Args:
            model: Ollama model to use (e.g., llama2, mistral, codellama)
            base_url: Ollama server URL. If None, uses OLLAMA_BASE_URL env var or default localhost
        """
        self.model = model
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def is_available(self) -> bool:
        """Check if Ollama server is running and accessible."""
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate a response using Ollama.

        Args:
            prompt: The user prompt/message
            system_prompt: Optional system instruction

        Returns:
            The generated response text

        Raises:
            ValueError: If Ollama is not available
            Exception: If the API request fails
        """
        if not self.is_available():
            raise ValueError(
                f"Ollama server not available at {self.base_url}. "
                f"Make sure Ollama is running (ollama serve)."
            )

        try:
            import requests

            # Build messages for Ollama API
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Call Ollama API
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False
                },
                timeout=30
            )

            # Handle model not found error specifically
            if response.status_code == 404:
                raise Exception(
                    f"Model '{self.model}' not found in Ollama. "
                    f"Pull it first with: ollama pull {self.model}"
                )

            response.raise_for_status()

            result = response.json()
            return result["message"]["content"]

        except ImportError:
            raise Exception(
                "requests library not installed. Install with: pip install requests"
            )
        except Exception as e:
            # Re-raise if already formatted
            if "not found in Ollama" in str(e):
                raise
            raise Exception(f"Ollama API error: {str(e)}") from e


# Placeholder for future providers
# class OpenAIProvider(LLMProvider):
#     """OpenAI API provider implementation."""
#     pass
