from typing import Optional
from llm.llm_provider import LLMProvider


class Agent:
    """Agent that can respond with fixed templates or using an LLM."""

    def __init__(
        self,
        name: str,
        response_template: Optional[str] = None,
        llm_provider: Optional[LLMProvider] = None,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize an agent.

        Args:
            name: Agent's name
            response_template: Template for fixed responses (if not using LLM)
            llm_provider: LLM provider for generating responses
            system_prompt: System prompt for LLM (sets agent behavior/role)
        """
        self.name = name
        self.response_template = response_template
        self.llm_provider = llm_provider
        self.base_system_prompt = system_prompt  # Store original system prompt
        self.system_prompt = system_prompt
        self.step_count = 0
        self.current_stats = {}  # Track current variable values

        # Validate configuration
        if not llm_provider and not response_template:
            raise ValueError(
                f"Agent '{name}' must have either response_template or llm_provider"
            )

    def update_stats(self, stats: dict) -> None:
        """
        Update agent's current stats and rebuild system prompt with stats.

        Args:
            stats: Dictionary of current variable values for this agent
        """
        self.current_stats = stats

        # Rebuild system prompt with current stats if LLM is used
        if self.llm_provider and self.base_system_prompt:
            if stats:
                stats_text = "\n\n=== YOUR CURRENT STATUS ===\n"
                for var_name, value in stats.items():
                    stats_text += f"{var_name}: {value}\n"
                self.system_prompt = self.base_system_prompt + stats_text
            else:
                self.system_prompt = self.base_system_prompt

    def respond(self, message: str) -> str:
        """Generate a response to the orchestrator's message."""
        self.step_count += 1

        # If LLM is configured, it must work - no fallback
        if self.llm_provider:
            if not self.llm_provider.is_available():
                raise ValueError(
                    f"Agent '{self.name}': LLM provider is configured but not available. "
                    f"Check API keys, network connection, and provider configuration."
                )
            # LLM must succeed - any exception bubbles up
            response = self.llm_provider.generate_response(
                prompt=message,
                system_prompt=self.system_prompt
            )
            return response

        # Only use template if NO LLM is configured
        if self.response_template:
            return f"{self.response_template} (step {self.step_count})"

        raise ValueError(f"Agent '{self.name}' has no available response method")
