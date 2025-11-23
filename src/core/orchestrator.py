import sys
import yaml
from pathlib import Path
from dotenv import load_dotenv
from core.agent import Agent
from core.game_engine import GameEngine
from core.simulator_agent import SimulatorAgent, SimulationError
from utils.persistence import RunPersistence
from utils.logging_config import MessageCode, PerformanceTimer
from utils.config_parser import parse_scripted_events
from llm.providers import GeminiProvider, OllamaProvider, UnifiedLLMProvider


class Orchestrator:
    """Orchestrator that manages multi-step simulation with agents."""

    def __init__(self, config_path: str, scenario_name: str, save_frequency: int, engine_llm_provider=None):
        # Load environment variables from .env file
        load_dotenv()

        self.config = self._load_config(config_path)
        self.max_steps = self.config.get("max_steps", 5)
        self.persistence = RunPersistence(scenario_name, save_frequency)
        self.logger = self.persistence.logger  # Use the same logger instance
        self.agents = self._initialize_agents()
        self.game_engine = GameEngine(self.config)

        # Initialize SimulatorAgent
        engine_config = self.config.get("engine", {})

        # Use injected provider if provided (for testing), otherwise create one
        if engine_llm_provider is not None:
            simulator_llm = engine_llm_provider
        else:
            engine_llm_config = {
                "provider": engine_config.get("provider"),
                "model": engine_config.get("model")
            }
            simulator_llm = self._create_llm_provider_for_engine(engine_llm_config)

        self.simulator_agent = SimulatorAgent(
            llm_provider=simulator_llm,
            game_engine=self.game_engine,
            system_prompt=engine_config.get("system_prompt", ""),
            simulation_plan=engine_config.get("simulation_plan", ""),
            realism_guidelines=engine_config.get("realism_guidelines", ""),
            scripted_events=parse_scripted_events(engine_config.get("scripted_events")),
            context_window_size=engine_config.get("context_window_size", 5),
            logger=self.logger
        )

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
            # Check if agent uses LLM
            llm_config = agent_config.get("llm")
            llm_provider = None

            if llm_config:
                provider_type = llm_config.get("provider", "gemini").lower()

                # Use UnifiedLLMProvider for new providers, keep old ones for backwards compatibility
                if provider_type in ["openai", "grok", "anthropic"]:
                    model_name = llm_config.get("model")
                    llm_provider = UnifiedLLMProvider(
                        provider=provider_type,
                        model=model_name,
                        base_url=llm_config.get("base_url")
                    )
                    provider_display = f"{provider_type.title()} ({model_name})"

                    env_var_map = {
                        "openai": "OPENAI_API_KEY",
                        "grok": "XAI_API_KEY",
                        "anthropic": "ANTHROPIC_API_KEY"
                    }
                    setup_instructions = (
                        f"  1. Copy .env.example to .env\n"
                        f"  2. Add your API key: {env_var_map[provider_type]}=your_key_here\n"
                    )

                elif provider_type == "gemini":
                    model_name = llm_config.get("model", "gemini-1.5-flash")
                    # Can use UnifiedLLMProvider or keep GeminiProvider for backwards compatibility
                    llm_provider = UnifiedLLMProvider(provider="gemini", model=model_name)
                    provider_display = f"Google Gemini ({model_name})"
                    setup_instructions = (
                        "  1. Copy .env.example to .env\n"
                        "  2. Add your API key: GEMINI_API_KEY=your_key_here\n"
                        "  3. Get a free key at: https://makersuite.google.com/app/apikey\n"
                    )

                elif provider_type == "ollama":
                    model_name = llm_config.get("model", "llama2")
                    base_url = llm_config.get("base_url")
                    # Can use UnifiedLLMProvider or keep OllamaProvider for backwards compatibility
                    llm_provider = UnifiedLLMProvider(provider="ollama", model=model_name, base_url=base_url)
                    provider_display = f"Ollama ({model_name})"
                    setup_instructions = (
                        "  1. Install Ollama: https://ollama.ai\n"
                        "  2. Pull the model: ollama pull {model}\n"
                        "  3. Start Ollama server: ollama serve\n"
                    ).format(model=model_name)

                else:
                    raise ValueError(f"Unknown LLM provider: {provider_type}")

                # Check if provider is available - fail fast if not configured
                if llm_provider and not llm_provider.is_available():
                    error_msg = (
                        f"\n{'='*70}\n"
                        f"ERROR: LLM Provider Not Available\n"
                        f"{'='*70}\n"
                        f"Agent: {agent_config['name']}\n"
                        f"Provider: {provider_display}\n"
                        f"\nThe LLM provider is not available. Possible causes:\n"
                        f"  1. Provider not running or configured\n"
                        f"  2. Network issues\n"
                        f"  3. Invalid configuration\n"
                        f"\nTo fix ({provider_type}):\n"
                        f"{setup_instructions}"
                        f"\n{'='*70}\n"
                    )
                    print(error_msg, file=sys.stderr)
                    raise ValueError(
                        f"LLM provider not available for agent '{agent_config['name']}'."
                    )

            agent = Agent(
                name=agent_config["name"],
                response_template=agent_config.get("response_template"),
                llm_provider=llm_provider,
                system_prompt=agent_config.get("system_prompt")
            )
            agents.append(agent)

            agent_type = "LLM-powered" if llm_provider else "template-based"
            self.logger.info(
                MessageCode.AGT001,
                f"{agent_type} agent initialized",
                agent_name=agent.name
            )
        return agents

    def _create_llm_provider_for_engine(self, llm_config: dict):
        """Create LLM provider for SimulatorAgent (engine)."""
        provider_type = llm_config.get("provider", "gemini").lower()
        model_name = llm_config.get("model")
        base_url = llm_config.get("base_url")

        provider = UnifiedLLMProvider(
            provider=provider_type,
            model=model_name or "gemini-1.5-flash",
            base_url=base_url
        )

        # Engine LLM MUST be available
        if not provider.is_available():
            error_msg = (
                f"\n{'='*70}\n"
                f"ERROR: Engine LLM Provider Not Available\n"
                f"{'='*70}\n"
                f"Provider: {provider_type}\n"
                f"Model: {model_name}\n"
                f"\nThe simulation engine requires an LLM to run.\n"
                f"Please ensure the provider is configured and available.\n"
                f"{'='*70}\n"
            )
            print(error_msg, file=sys.stderr)
            raise ValueError("Engine LLM provider not available. Simulation cannot run.")

        return provider

    def run(self):
        """Run the simulation loop with SimulatorAgent."""
        self.logger.info(
            MessageCode.SIM001,
            "Simulation started",
            num_agents=len(self.agents),
            max_steps=self.max_steps
        )

        print(f"Starting simulation with {len(self.agents)} agent(s) for {self.max_steps} steps")
        print(f"Results will be saved to: {self.persistence.run_dir}\n")

        try:
            # Step 0: Initialize simulation
            print("=== Step 0: Initialization ===")
            agent_names = [agent.name for agent in self.agents]

            try:
                agent_messages = self.simulator_agent.initialize_simulation(agent_names)
                print("SimulatorAgent initialized simulation")
            except SimulationError as e:
                self.logger.critical(MessageCode.SIM002, "Initialization failed", error=str(e))
                print(f"\nERROR: Simulation initialization failed:\n{e}", file=sys.stderr)
                raise

            # Main simulation loop
            for step in range(1, self.max_steps + 1):
                with PerformanceTimer(self.logger, MessageCode.PRF001, f"Step {step}", step=step):
                    self.logger.info(MessageCode.SIM003, "Step started", step=step, max_steps=self.max_steps)
                    print(f"\n=== Step {step}/{self.max_steps} ===")

                    # Collect agent responses
                    agent_responses = {}
                    step_messages = []

                    for agent in self.agents:
                        try:
                            message = agent_messages[agent.name]
                            print(f"SimulatorAgent -> {agent.name}: {message}")

                            self.logger.info(
                                MessageCode.AGT002,
                                "Message sent to agent",
                                agent_name=agent.name,
                                step=step,
                                content=message
                            )

                            step_messages.append({
                                "from": "SimulatorAgent",
                                "to": agent.name,
                                "content": message
                            })

                            # Agent responds
                            response = agent.respond(message)
                            print(f"{agent.name} -> SimulatorAgent: {response}")

                            self.logger.info(
                                MessageCode.AGT003,
                                "Response received from agent",
                                agent_name=agent.name,
                                step=step,
                                response=response
                            )

                            step_messages.append({
                                "from": agent.name,
                                "to": "SimulatorAgent",
                                "content": response
                            })

                            agent_responses[agent.name] = response

                        except Exception as e:
                            error_msg = f"Agent {agent.name} failed: {e}"
                            self.logger.error(
                                MessageCode.AGT005,
                                error_msg,
                                agent_name=agent.name,
                                step=step,
                                error=str(e)
                            )
                            print(f"ERROR: {error_msg}", file=sys.stderr)
                            agent_responses[agent.name] = f"ERROR: {str(e)}"

                    # SimulatorAgent processes responses and generates next messages
                    try:
                        agent_messages = self.simulator_agent.process_step(step, agent_responses)
                        print(f"[SimulatorAgent processed step {step}]")
                    except SimulationError as e:
                        self.logger.critical(
                            MessageCode.SIM002,
                            "Simulation failed",
                            step=step,
                            error=str(e)
                        )
                        print(f"\nERROR: Simulation failed at step {step}:\n{e}", file=sys.stderr)
                        raise

                    # Save snapshot if needed
                    if self.persistence.should_save(step):
                        try:
                            snapshot = self.game_engine.get_state_snapshot()
                            self.persistence.save_snapshot(
                                step,
                                snapshot["game_state"],
                                snapshot["global_vars"],
                                snapshot["agent_vars"],
                                step_messages
                            )
                            print(f"[Saved snapshot at step {step}]")
                        except Exception as e:
                            error_msg = f"Failed to save snapshot: {e}"
                            self.logger.error(MessageCode.PER004, error_msg, step=step, error=str(e))
                            print(f"WARNING: {error_msg}", file=sys.stderr)

                    # Advance round
                    self.game_engine.advance_round()
                    self.logger.info(MessageCode.SIM004, "Step completed", step=step)

            # Save final state
            try:
                snapshot = self.game_engine.get_state_snapshot()
                self.persistence.save_final(
                    self.max_steps,
                    snapshot["game_state"],
                    snapshot["global_vars"],
                    snapshot["agent_vars"],
                    step_messages
                )
            except Exception as e:
                error_msg = f"Failed to save final state: {e}"
                self.logger.critical(MessageCode.PER004, error_msg, error=str(e))
                print(f"ERROR: {error_msg}", file=sys.stderr)

            self.logger.info(MessageCode.SIM002, "Simulation completed", total_steps=self.max_steps)
            print("\nSimulation completed.")
            print(f"Final game state: {self.game_engine.get_state()}")
            print(f"\nResults saved to: {self.persistence.run_dir}")

        except KeyboardInterrupt:
            self.logger.warning(MessageCode.SIM002, "Simulation interrupted by user")
            print("\n\nSimulation interrupted by user")
            raise

        except SimulationError:
            # Already logged, just re-raise
            raise

        except Exception as e:
            self.logger.critical(MessageCode.SIM002, "Simulation failed with unhandled exception", error=str(e))
            raise

        finally:
            self.persistence.close()
