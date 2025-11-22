import yaml
from pathlib import Path
from agent import Agent


class Orchestrator:
    """Orchestrator that manages multi-step simulation with agents."""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.max_steps = self.config.get("max_steps", 5)
        self.orchestrator_message = self.config.get("orchestrator_message", "Continue")
        self.agents = self._initialize_agents()

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _initialize_agents(self) -> list[Agent]:
        """Initialize agents from configuration."""
        agents = []
        for agent_config in self.config.get("agents", []):
            agent = Agent(
                name=agent_config["name"],
                response_template=agent_config["response_template"]
            )
            agents.append(agent)
        return agents

    def run(self):
        """Run the simulation loop."""
        print(f"Starting simulation with {len(self.agents)} agent(s) for {self.max_steps} steps\n")

        for step in range(1, self.max_steps + 1):
            print(f"=== Step {step}/{self.max_steps} ===")

            for agent in self.agents:
                print(f"Orchestrator -> {agent.name}: {self.orchestrator_message}")
                response = agent.respond(self.orchestrator_message)
                print(f"{agent.name} -> Orchestrator: {response}")

            print()

        print("Simulation completed.")
