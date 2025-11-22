"""Mock LLM provider for testing without API calls."""

from typing import Optional
from llm.llm_provider import LLMProvider


class MockOllamaProvider(LLMProvider):
    """Mock Ollama provider for testing without running Ollama server."""

    def __init__(self, model: str = "llama2", responses: Optional[list[str]] = None, available: bool = True):
        """
        Initialize mock Ollama provider.

        Args:
            model: Model name (for verification in tests)
            responses: List of responses to cycle through
            available: Whether the provider should report as available
        """
        self.model = model
        self.responses = responses or ["Mock Ollama response"]
        self.available = available
        self.call_count = 0
        self.last_prompt = None
        self.last_system_prompt = None

    def is_available(self) -> bool:
        """Check if mock Ollama is available."""
        return self.available

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate mock response."""
        if not self.is_available():
            raise ValueError("Mock Ollama provider not available")

        self.last_prompt = prompt
        self.last_system_prompt = system_prompt

        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response

    def reset(self):
        """Reset call count and stored prompts."""
        self.call_count = 0
        self.last_prompt = None
        self.last_system_prompt = None


class MockLLMProvider(LLMProvider):
    """Mock LLM provider that returns canned responses without API calls."""

    def __init__(self, responses: Optional[list[str]] = None, available: bool = True):
        """
        Initialize mock provider.

        Args:
            responses: List of responses to cycle through. If None, returns generic response.
            available: Whether the provider should report as available
        """
        self.responses = responses or ["Mock LLM response"]
        self.available = available
        self.call_count = 0
        self.last_prompt = None
        self.last_system_prompt = None

    def is_available(self) -> bool:
        """Check if mock provider is available."""
        return self.available

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate a mock response without any API calls.

        Args:
            prompt: The user prompt (stored for verification)
            system_prompt: Optional system prompt (stored for verification)

        Returns:
            Next response from the responses list (cycles through)
        """
        if not self.is_available():
            raise ValueError("Mock LLM provider not available")

        # Store for test verification
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt

        # Get response (cycle through list)
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1

        return response

    def reset(self):
        """Reset call count and stored prompts (useful between tests)."""
        self.call_count = 0
        self.last_prompt = None
        self.last_system_prompt = None
