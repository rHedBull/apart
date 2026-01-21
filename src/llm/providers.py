"""LLM provider implementations for various services."""

import os
from typing import Optional, Dict, Any
from llm.llm_provider import LLMProvider


class UnifiedLLMProvider(LLMProvider):
    """Unified LLM provider supporting OpenAI, Grok, Anthropic, Gemini, and Ollama."""

    SUPPORTED_PROVIDERS = ["openai", "grok", "anthropic", "gemini", "ollama"]

    # Environment variable mappings for API keys
    API_KEY_ENV_MAP = {
        "openai": "OPENAI_API_KEY",
        "grok": "XAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "ollama": None,  # No API key needed
    }

    # Package names for each provider
    PACKAGE_MAP = {
        "openai": "openai",
        "grok": "openai",
        "anthropic": "anthropic",
        "gemini": "google-generativeai",
        "ollama": "requests",
    }

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize unified LLM provider.

        Args:
            provider: Provider type (openai, grok, anthropic, gemini, ollama)
            model: Model identifier
            api_key: Optional API key (falls back to env vars)
            base_url: Optional base URL override
        """
        provider = provider.lower()
        if provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unknown provider: {provider}. "
                f"Supported providers: {', '.join(self.SUPPORTED_PROVIDERS)}"
            )

        self.provider_type = provider
        self.model = model
        self.base_url = base_url

        # Set API key from parameter or environment
        self.api_key = api_key or self._get_api_key_from_env()

        # Initialize provider-specific clients
        self._client = None
        self._model_instance = None
        self._initialize_provider()

    def _get_api_key_from_env(self) -> Optional[str]:
        """Get API key from environment variables."""
        env_var = self.API_KEY_ENV_MAP.get(self.provider_type)
        if env_var:
            return os.getenv(env_var)
        return None

    def _initialize_provider(self):
        """Initialize provider-specific client."""
        try:
            if self.provider_type == "openai":
                import openai
                base_url = self.base_url or "https://api.openai.com/v1"
                self._client = openai.OpenAI(api_key=self.api_key, base_url=base_url)

            elif self.provider_type == "grok":
                import openai
                base_url = self.base_url or "https://api.x.ai/v1"
                self._client = openai.OpenAI(api_key=self.api_key, base_url=base_url)

            elif self.provider_type == "anthropic":
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)

            elif self.provider_type == "gemini":
                import google.generativeai as genai
                if self.api_key:
                    genai.configure(api_key=self.api_key)
                    self._model_instance = genai.GenerativeModel(self.model)

            elif self.provider_type == "ollama":
                # Ollama doesn't need client initialization
                self.base_url = self.base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        except ImportError as e:
            raise ImportError(
                f"Required package not installed for {self.provider_type}. "
                f"Install with: uv pip install {self._get_package_name()}"
            ) from e

    def _get_package_name(self) -> str:
        """Get pip package name for provider."""
        return self.PACKAGE_MAP.get(self.provider_type, "unknown")

    def is_available(self) -> bool:
        """Check if provider is configured and available."""
        # Check provider-specific requirements
        if self.provider_type == "ollama":
            # Check if Ollama server is reachable
            try:
                import requests
                response = requests.get(f"{self.base_url}/api/tags", timeout=2)
                return response.status_code == 200
            except Exception:
                return False

        elif self.provider_type == "gemini":
            # Check if API key is set and model instance was created
            return self.api_key is not None and self._model_instance is not None

        else:
            # For OpenAI, Grok, Anthropic: check if API key is set and client initialized
            return self.api_key is not None and self._client is not None

    def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        force_json: bool = False
    ) -> str:
        """
        Generate response from LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            json_schema: Optional JSON schema (Gemini only)
            force_json: Force JSON output mode

        Returns:
            Generated response text
        """
        if not self.is_available():
            raise ValueError(self._get_unavailable_error_message())

        if self.provider_type in ["openai", "grok"]:
            return self._generate_openai_compatible(prompt, system_prompt, force_json)
        elif self.provider_type == "anthropic":
            return self._generate_anthropic(prompt, system_prompt, force_json)
        elif self.provider_type == "gemini":
            return self._generate_gemini(prompt, system_prompt, json_schema, force_json)
        elif self.provider_type == "ollama":
            return self._generate_ollama(prompt, system_prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider_type}")

    def _generate_openai_compatible(
        self,
        prompt: str,
        system_prompt: Optional[str],
        force_json: bool
    ) -> str:
        """Generate response using OpenAI-compatible API (OpenAI, Grok)."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.model,
            "messages": messages
        }

        if force_json:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self._client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"{self.provider_type} API error: {str(e)}") from e

    def _generate_anthropic(
        self,
        prompt: str,
        system_prompt: Optional[str],
        force_json: bool
    ) -> str:
        """Generate response using Anthropic API."""
        messages = [{"role": "user", "content": prompt}]

        kwargs = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": messages
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        # Note: Anthropic doesn't have native JSON mode
        # If force_json is True, we rely on prompt engineering
        if force_json and system_prompt:
            kwargs["system"] = f"{system_prompt}\n\nRespond only with valid JSON."
        elif force_json:
            kwargs["system"] = "Respond only with valid JSON."

        try:
            response = self._client.messages.create(**kwargs)
            return response.content[0].text
        except Exception as e:
            raise Exception(f"Anthropic API error: {str(e)}") from e

    def _generate_gemini(
        self,
        prompt: str,
        system_prompt: Optional[str],
        json_schema: Optional[Dict[str, Any]],
        force_json: bool
    ) -> str:
        """Generate response using Google Gemini API."""
        # Build full prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        # Determine generation config
        generation_config = None
        if force_json or json_schema:
            generation_config = {"response_mime_type": "application/json"}
            if json_schema:
                generation_config["response_schema"] = json_schema

        try:
            if generation_config:
                response = self._model_instance.generate_content(
                    full_prompt,
                    generation_config=generation_config
                )
            else:
                response = self._model_instance.generate_content(full_prompt)

            return response.text
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}") from e

    def _generate_ollama(self, prompt: str, system_prompt: Optional[str]) -> str:
        """Generate response using Ollama API."""
        import requests

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False
                },
                timeout=30
            )

            # Handle model not found error
            if response.status_code == 404:
                raise Exception(
                    f"Model '{self.model}' not found in Ollama. "
                    f"Pull it first with: ollama pull {self.model}"
                )

            response.raise_for_status()
            result = response.json()
            return result["message"]["content"]

        except Exception as e:
            if "not found in Ollama" in str(e):
                raise
            raise Exception(f"Ollama API error: {str(e)}") from e

    def _get_unavailable_error_message(self) -> str:
        """Get error message for unavailable provider."""
        if self.provider_type == "ollama":
            env_var = "OLLAMA_BASE_URL (default: http://localhost:11434)"
        else:
            env_var = self.API_KEY_ENV_MAP.get(self.provider_type, "unknown")
        return (
            f"{self.provider_type} provider not configured. "
            f"Set {env_var} environment variable."
        )
