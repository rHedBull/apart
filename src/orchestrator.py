import sys
import yaml
from pathlib import Path
from agent import Agent
from game_engine import GameEngine
from persistence import RunPersistence
from logging_config import MessageCode, PerformanceTimer


class Orchestrator:
    """Orchestrator that manages multi-step simulation with agents."""

    def __init__(self, config_path: str, scenario_name: str, save_frequency: int):
        self.config = self._load_config(config_path)
        self.max_steps = self.config.get("max_steps", 5)
        self.persistence = RunPersistence(scenario_name, save_frequency)
        self.logger = self.persistence.logger  # Use the same logger instance
        self.agents = self._initialize_agents()
        self.game_engine = GameEngine(self.config)

        self.logger.info(MessageCode.PER001, "Run directory created", run_id=self.persistence.run_id, run_dir=str(self.persistence.run_dir))

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load configuration: {e}")

    def _initialize_agents(self) -> list[Agent]:
        """Initialize agents from configuration."""
        agents = []
        for agent_config in self.config.get("agents", []):
            agent = Agent(
                name=agent_config["name"],
                response_template=agent_config["response_template"]
            )
            agents.append(agent)
            self.logger.info(MessageCode.AGT001, "Agent initialized", agent_name=agent.name)
        return agents

    def run(self):
        """Run the simulation loop."""
        self.logger.info(
            MessageCode.SIM001,
            "Simulation started",
            num_agents=len(self.agents),
            max_steps=self.max_steps
        )

        print(f"Starting simulation with {len(self.agents)} agent(s) for {self.max_steps} steps")
        print(f"Results will be saved to: {self.persistence.run_dir}\n")

        try:
            for step in range(1, self.max_steps + 1):
                with PerformanceTimer(self.logger, MessageCode.PRF001, f"Step {step}", step=step):
                    self.logger.info(MessageCode.SIM003, "Step started", step=step, max_steps=self.max_steps)
                    print(f"=== Step {step}/{self.max_steps} ===")

                    # Collect messages for this step
                    messages = []

                    # Process each agent in turn
                    for agent in self.agents:
                        try:
                            # Game engine generates the message based on current state
                            message = self.game_engine.get_message_for_agent(agent.name)
                            print(f"Orchestrator -> {agent.name}: {message}")

                            self.logger.info(
                                MessageCode.AGT002,
                                "Message sent to agent",
                                agent_name=agent.name,
                                step=step,
                                content=message
                            )

                            messages.append({
                                "from": "Orchestrator",
                                "to": agent.name,
                                "content": message
                            })

                        except Exception as e:
                            error_msg = f"Failed to generate message for agent: {e}"
                            self.logger.error(
                                MessageCode.AGT005,
                                error_msg,
                                agent_name=agent.name,
                                step=step,
                                error=str(e)
                            )
                            print(f"ERROR: {error_msg}", file=sys.stderr)
                            # Continue with next agent
                            continue

                        try:
                            # Agent responds
                            response = agent.respond(message)
                            print(f"{agent.name} -> Orchestrator: {response}")

                            self.logger.info(
                                MessageCode.AGT003,
                                "Response received from agent",
                                agent_name=agent.name,
                                step=step,
                                response=response
                            )

                            messages.append({
                                "from": agent.name,
                                "to": "Orchestrator",
                                "content": response
                            })

                        except Exception as e:
                            error_msg = f"Agent failed to respond: {e}"
                            self.logger.error(
                                MessageCode.AGT005,
                                error_msg,
                                agent_name=agent.name,
                                step=step,
                                error=str(e)
                            )
                            print(f"ERROR: {error_msg}", file=sys.stderr)
                            # Log error response
                            messages.append({
                                "from": agent.name,
                                "to": "Orchestrator",
                                "content": f"ERROR: {str(e)}"
                            })
                            # Continue with next agent
                            continue

                        try:
                            # Game engine processes the response and updates state
                            self.game_engine.process_agent_response(agent.name, response)
                            self.logger.debug(MessageCode.AGT004, "Agent state updated", agent_name=agent.name, step=step)

                        except Exception as e:
                            error_msg = f"Failed to process agent response: {e}"
                            self.logger.error(
                                MessageCode.GME004,
                                error_msg,
                                agent_name=agent.name,
                                step=step,
                                error=str(e)
                            )
                            print(f"ERROR: {error_msg}", file=sys.stderr)
                            # Continue with next agent

                    # Save snapshot if needed
                    if self.persistence.should_save(step):
                        try:
                            snapshot = self.game_engine.get_state_snapshot()
                            self.persistence.save_snapshot(
                                step,
                                snapshot["game_state"],
                                snapshot["global_vars"],
                                snapshot["agent_vars"],
                                messages
                            )
                            print(f"[Saved snapshot at step {step}]")
                        except Exception as e:
                            error_msg = f"Failed to save snapshot: {e}"
                            self.logger.error(MessageCode.PER004, error_msg, step=step, error=str(e))
                            print(f"WARNING: {error_msg}", file=sys.stderr)
                            # Continue simulation despite save failure

                    # Advance to next round
                    try:
                        self.game_engine.advance_round()
                        self.logger.debug(MessageCode.SIM005, "Round advanced", round=self.game_engine.state.round)
                    except Exception as e:
                        error_msg = f"Failed to advance round: {e}"
                        self.logger.error(MessageCode.GME004, error_msg, step=step, error=str(e))
                        print(f"WARNING: {error_msg}", file=sys.stderr)
                        # Continue anyway

                    self.logger.info(MessageCode.SIM004, "Step completed", step=step)
                    print()

            # Save final state
            try:
                snapshot = self.game_engine.get_state_snapshot()
                self.persistence.save_final(
                    self.max_steps,
                    snapshot["game_state"],
                    snapshot["global_vars"],
                    snapshot["agent_vars"],
                    messages
                )
            except Exception as e:
                error_msg = f"Failed to save final state: {e}"
                self.logger.critical(MessageCode.PER004, error_msg, error=str(e))
                print(f"ERROR: {error_msg}", file=sys.stderr)
                # Don't raise - we want to show simulation summary

            self.logger.info(MessageCode.SIM002, "Simulation completed", total_steps=self.max_steps)
            print("Simulation completed.")
            print(f"Final game state: {self.game_engine.get_state()}")
            print(f"\nResults saved to: {self.persistence.run_dir}")

        except KeyboardInterrupt:
            self.logger.warning(MessageCode.SIM002, "Simulation interrupted by user")
            print("\n\nSimulation interrupted by user")
            raise  # Re-raise to be caught by main()

        except Exception as e:
            self.logger.critical(MessageCode.SIM002, "Simulation failed with unhandled exception", error=str(e))
            raise  # Re-raise to be caught by main()

        finally:
            # Always close logger
            self.persistence.close()
