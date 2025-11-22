class Agent:
    """Simple agent that responds with a fixed sentence."""

    def __init__(self, name: str, response_template: str):
        self.name = name
        self.response_template = response_template
        self.step_count = 0

    def respond(self, message: str) -> str:
        """Generate a response to the orchestrator's message."""
        self.step_count += 1
        return f"{self.response_template} (step {self.step_count})"
