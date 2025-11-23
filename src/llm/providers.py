"""LLM provider implementations for various services."""

import os
from typing import Optional, Dict, Any
from llm.llm_provider import LLMProvider


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider implementation supporting both API and Vertex AI."""

    # JSON schema for simulation engine responses
    # Note: Gemini's schema validation requires explicit type definitions
    # Using STRING type for nested objects to avoid complex schema definitions
    SIMULATION_ENGINE_SCHEMA = {
        "type": "OBJECT",
        "properties": {
            "state_updates": {
                "type": "OBJECT",
                "properties": {
                    "global_vars": {"type": "OBJECT"},
                    "agent_vars": {"type": "OBJECT"}
                },
                "required": ["global_vars", "agent_vars"]
            },
            "events": {
                "type": "ARRAY",
                "items": {"type": "OBJECT"}
            },
            "agent_messages": {"type": "OBJECT"},
            "reasoning": {"type": "STRING"}
        },
        "required": ["state_updates", "events", "agent_messages", "reasoning"]
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-1.5-flash",
        use_vertex: bool = False,
        project: Optional[str] = None,
        location: str = "us-central1"
    ):
        """
        Initialize Gemini provider.

        Args:
            api_key: Google API key. If None, will try GEMINI_API_KEY env var (only for direct API)
            model_name: Gemini model to use (default: gemini-1.5-flash)
            use_vertex: If True, use Vertex AI instead of direct API
            project: GCP project ID (required if use_vertex=True)
            location: GCP region (default: us-central1)
        """
        self.model_name = model_name
        self.use_vertex = use_vertex
        self._model = None

        if use_vertex:
            # Vertex AI initialization
            try:
                import vertexai
                from vertexai.generative_models import GenerativeModel

                project_id = project or os.getenv("GCP_PROJECT")
                if not project_id:
                    raise ValueError(
                        "Vertex AI requires project ID. Set GCP_PROJECT env var or pass project parameter."
                    )

                vertexai.init(project=project_id, location=location)
                self._model = GenerativeModel(self.model_name)
                self._is_vertex = True

            except ImportError:
                raise ImportError(
                    "Vertex AI SDK not installed. Install with: pip install google-cloud-aiplatform"
                )
        else:
            # Direct API initialization
            try:
                import google.generativeai as genai

                self.api_key = api_key or os.getenv("GEMINI_API_KEY")
                if self.api_key:
                    genai.configure(api_key=self.api_key)
                    self._model = genai.GenerativeModel(self.model_name)
                self._is_vertex = False

            except ImportError:
                raise ImportError(
                    "Google Generative AI SDK not installed. Install with: pip install google-generativeai"
                )

    def is_available(self) -> bool:
        """Check if Gemini is properly configured."""
        return self._model is not None

    def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        force_json: bool = False
    ) -> str:
        """
        Generate a response using Google Gemini with optional structured JSON output.

        Args:
            prompt: The user prompt/message
            system_prompt: Optional system instruction
            json_schema: Optional JSON schema to enforce structured output
            force_json: If True, use default simulation engine schema

        Returns:
            The generated response text (JSON string if schema provided)

        Raises:
            ValueError: If the provider is not properly configured
            Exception: If the API request fails
        """
        if not self.is_available():
            error_msg = (
                "Vertex AI provider not configured. Set GCP_PROJECT environment variable."
                if self.use_vertex
                else "Gemini provider not configured. Set GEMINI_API_KEY environment variable."
            )
            raise ValueError(error_msg)

        try:
            # Build full prompt
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            # Determine if we should use structured JSON output
            # Note: Using response_mime_type without schema for simulation engine
            # because the dynamic nature of the response doesn't fit Gemini's strict schema requirements
            if force_json:
                # Use JSON mode without strict schema validation
                generation_config = {"response_mime_type": "application/json"}
                response = self._model.generate_content(
                    full_prompt,
                    generation_config=generation_config
                )
            elif json_schema is not None:
                # Custom schema provided - use it
                generation_config = {
                    "response_mime_type": "application/json",
                    "response_schema": json_schema
                }
                response = self._model.generate_content(
                    full_prompt,
                    generation_config=generation_config
                )
            else:
                # No JSON mode
                response = self._model.generate_content(full_prompt)

            return response.text

        except Exception as e:
            provider_name = "Vertex AI" if self.use_vertex else "Gemini API"
            raise Exception(f"{provider_name} error: {str(e)}") from e


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
