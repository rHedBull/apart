import yaml
from pathlib import Path
from agent import Agent
from game_engine import GameEngine
from persistence import RunPersistence


class Orchestrator:
    """Orchestrator that manages multi-step simulation with agents."""

    def __init__(self, config_path: str, scenario_name: str, save_frequency: int):
        self.config = self._load_config(config_path)
        self.max_steps = self.config.get("max_steps", 5)
        self.agents = self._initialize_agents()
        self.game_engine = GameEngine(self.config)
        self.persistence = RunPersistence(scenario_name, save_frequency)

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
        print(f"Starting simulation with {len(self.agents)} agent(s) for {self.max_steps} steps")
        print(f"Results will be saved to: {self.persistence.run_dir}\n")

        for step in range(1, self.max_steps + 1):
            print(f"=== Step {step}/{self.max_steps} ===")

            # Collect messages for this step
            messages = []

            # Process each agent in turn
            for agent in self.agents:
                # Game engine generates the message based on current state
                message = self.game_engine.get_message_for_agent(agent.name)
                print(f"Orchestrator -> {agent.name}: {message}")

                messages.append({
                    "from": "Orchestrator",
                    "to": agent.name,
                    "content": message
                })

                # Agent responds
                response = agent.respond(message)
                print(f"{agent.name} -> Orchestrator: {response}")

                messages.append({
                    "from": agent.name,
                    "to": "Orchestrator",
                    "content": response
                })

                # Game engine processes the response and updates state
                self.game_engine.process_agent_response(agent.name, response)

            # Save snapshot if needed
            if self.persistence.should_save(step):
                snapshot = self.game_engine.get_state_snapshot()
                self.persistence.save_snapshot(
                    step,
                    snapshot["game_state"],
                    snapshot["global_vars"],
                    snapshot["agent_vars"],
                    messages
                )
                print(f"[Saved snapshot at step {step}]")

            # Advance to next round
            self.game_engine.advance_round()
            print()

        # Save final state
        snapshot = self.game_engine.get_state_snapshot()
        self.persistence.save_final(
            self.max_steps,
            snapshot["game_state"],
            snapshot["global_vars"],
            snapshot["agent_vars"],
            messages
        )

        print("Simulation completed.")
        print(f"Final game state: {self.game_engine.get_state()}")
        print(f"\nResults saved to: {self.persistence.run_dir}")
