from abc import ABC, abstractmethod
from typing import Optional


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: The user prompt/message to send to the LLM
            system_prompt: Optional system prompt to set context/behavior

        Returns:
            The generated response text

        Raises:
            Exception: If the LLM request fails
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the LLM provider is properly configured and available.

        Returns:
            True if the provider is ready to use, False otherwise
        """
        pass
