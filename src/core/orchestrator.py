import os
import sys
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
from core.agent import Agent
from core.game_engine import GameEngine
from core.simulator_agent import SimulatorAgent, SimulationError
from core.event_emitter import emit, enable_event_emitter, disable_event_emitter, EventTypes
from utils.persistence import RunPersistence
from utils.logging_config import MessageCode, PerformanceTimer
from utils.config_parser import parse_scripted_events, parse_geography, parse_spatial_graph, parse_modules, merge_module_variables
from llm.providers import UnifiedLLMProvider


class Orchestrator:
    """Orchestrator that manages multi-step simulation with agents."""

    def __init__(self, config_path: str, scenario_name: str, save_frequency: int, engine_llm_provider=None, run_id: str | None = None):
        # Load environment variables from .env file
        load_dotenv()

        self.config = self._load_config(config_path)

        # Load and compose behavior modules if specified
        self.composed_modules = parse_modules(self.config)
        if self.composed_modules:
            self.config = merge_module_variables(self.config, self.composed_modules)

        self.max_steps = self.config.get("max_steps", 5)
        self.time_step_duration = self.config.get("time_step_duration", "1 turn")
        self.simulator_awareness = self.config.get("simulator_awareness", True)
        self.enable_compute_resources = self.config.get("enable_compute_resources", False)
        self.persistence = RunPersistence(scenario_name, save_frequency, run_id=run_id)
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

        # Parse geography if present
        geography = parse_geography(self.config.get("geography"))

        # Get spatial graph from modules or fall back to geography config
        if self.composed_modules and self.composed_modules.spatial_graph:
            self.spatial_graph = self.composed_modules.spatial_graph
            movement_config = self.composed_modules.movement_config
        else:
            # Parse spatial graph from geography config (legacy)
            self.spatial_graph, movement_config = parse_spatial_graph(self.config.get("geography"))

        self.simulator_agent = SimulatorAgent(
            llm_provider=simulator_llm,
            game_engine=self.game_engine,
            system_prompt=engine_config.get("system_prompt", ""),
            simulation_plan=engine_config.get("simulation_plan", ""),
            realism_guidelines=engine_config.get("realism_guidelines", ""),
            scripted_events=parse_scripted_events(engine_config.get("scripted_events")),
            context_window_size=engine_config.get("context_window_size", 5),
            time_step_duration=self.time_step_duration,
            simulator_awareness=self.simulator_awareness,
            enable_compute_resources=self.enable_compute_resources,
            geography=geography,
            spatial_graph=self.spatial_graph,
            movement_config=movement_config,
            composed_modules=self.composed_modules,
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

            # Build agent system prompt with simulator awareness instructions
            base_system_prompt = agent_config.get("system_prompt", "")

            if not self.simulator_awareness and llm_provider:
                # Add instructions for non-simulator-aware agents
                enhanced_system_prompt = f"""{base_system_prompt}

IMPORTANT - Response Format:
- You will receive messages describing events and situations happening to you
- Your responses should contain your ACTIONS and COMMUNICATIONS with the real world
- Include what you DO and what you SAY out loud
- Internal thoughts that you keep to yourself should NOT be in your response
- Only include actions/speech that others can observe or hear

Example of a GOOD response: "I walk to the market and ask the merchant: 'What is the price for wheat today?'"
Example of a BAD response: "I think about going to the market" (this is just internal thought)"""
            else:
                enhanced_system_prompt = base_system_prompt

            agent = Agent(
                name=agent_config["name"],
                response_template=agent_config.get("response_template"),
                llm_provider=llm_provider,
                system_prompt=enhanced_system_prompt
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

    def _initialize_simulation(self) -> dict:
        """Initialize simulation (step 0) and return initial agent messages."""
        print("=== Step 0: Initialization ===")
        agent_names = [agent.name for agent in self.agents]

        try:
            agent_messages = self.simulator_agent.initialize_simulation(agent_names)
            print("SimulatorAgent initialized simulation")
        except SimulationError as e:
            self.logger.critical(MessageCode.SIM002, "Initialization failed", error=str(e))
            print(f"\nERROR: Simulation initialization failed:\n{e}", file=sys.stderr)
            raise

        # Initialize agent stats after simulation setup
        initial_state = self.game_engine.get_state_snapshot()
        for agent in self.agents:
            agent_stats = initial_state["agent_vars"].get(agent.name, {})
            agent.update_stats(agent_stats)

        return agent_messages

    def _process_single_agent(self, agent: Agent, step: int, message: str, agent_stats: dict) -> dict:
        """
        Process a single agent's response. Designed to run in a thread pool.

        Returns:
            dict with keys: agent_name, message, response, error
        """
        result = {
            "agent_name": agent.name,
            "message": message,
            "response": None,
            "error": None
        }

        try:
            # Update agent's stats before they respond
            agent.update_stats(agent_stats)

            # Agent responds (this is the slow LLM call)
            response = agent.respond(message)
            result["response"] = response

        except Exception as e:
            result["error"] = str(e)

        return result

    def _collect_agent_responses(self, step: int, agent_messages: dict) -> tuple[dict, list]:
        """Collect responses from all agents for a step (parallel execution)."""
        agent_responses = {}
        step_messages = []

        # Check if parallel execution is enabled (default: True)
        parallel_agents = os.environ.get("APART_PARALLEL_AGENTS", "1").lower() in ("1", "true", "yes")

        # Get current state snapshot once (same for all agents at step start)
        current_state = self.game_engine.get_state_snapshot()

        # Log and emit events for all outgoing messages first
        for agent in self.agents:
            message = agent_messages[agent.name]
            print(f"SimulatorAgent -> {agent.name}: {message}")

            self.logger.info(
                MessageCode.AGT002,
                "Message sent to agent",
                agent_name=agent.name,
                step=step,
                content=message
            )

            emit(
                EventTypes.AGENT_MESSAGE_SENT,
                step=step,
                agent_name=agent.name,
                message=message
            )

            step_messages.append({
                "from": "SimulatorAgent",
                "to": agent.name,
                "content": message
            })

        if parallel_agents and len(self.agents) > 1:
            # Parallel execution using ThreadPoolExecutor
            self.logger.info(
                MessageCode.AGT002,
                f"Processing {len(self.agents)} agents in parallel",
                step=step
            )

            with ThreadPoolExecutor(max_workers=len(self.agents), thread_name_prefix="agent-") as executor:
                # Submit all agent tasks
                futures = {}
                for agent in self.agents:
                    message = agent_messages[agent.name]
                    agent_stats = current_state["agent_vars"].get(agent.name, {})
                    future = executor.submit(
                        self._process_single_agent,
                        agent, step, message, agent_stats
                    )
                    futures[future] = agent

                # Collect results as they complete
                for future in as_completed(futures):
                    result = future.result()
                    agent_name = result["agent_name"]

                    if result["error"]:
                        error_msg = f"Agent {agent_name} failed: {result['error']}"
                        self.logger.error(
                            MessageCode.AGT005,
                            error_msg,
                            agent_name=agent_name,
                            step=step,
                            error=result["error"]
                        )
                        print(f"ERROR: {error_msg}", file=sys.stderr)
                        agent_responses[agent_name] = f"ERROR: {result['error']}"
                    else:
                        response = result["response"]
                        print(f"{agent_name} -> SimulatorAgent: {response}")

                        self.logger.info(
                            MessageCode.AGT003,
                            "Response received from agent",
                            agent_name=agent_name,
                            step=step,
                            response=response
                        )

                        emit(
                            EventTypes.AGENT_RESPONSE_RECEIVED,
                            step=step,
                            agent_name=agent_name,
                            response=response
                        )

                        agent_responses[agent_name] = response
        else:
            # Sequential execution (original behavior)
            for agent in self.agents:
                message = agent_messages[agent.name]
                agent_stats = current_state["agent_vars"].get(agent.name, {})

                result = self._process_single_agent(agent, step, message, agent_stats)
                agent_name = result["agent_name"]

                if result["error"]:
                    error_msg = f"Agent {agent_name} failed: {result['error']}"
                    self.logger.error(
                        MessageCode.AGT005,
                        error_msg,
                        agent_name=agent_name,
                        step=step,
                        error=result["error"]
                    )
                    print(f"ERROR: {error_msg}", file=sys.stderr)
                    agent_responses[agent_name] = f"ERROR: {result['error']}"
                else:
                    response = result["response"]
                    print(f"{agent_name} -> SimulatorAgent: {response}")

                    self.logger.info(
                        MessageCode.AGT003,
                        "Response received from agent",
                        agent_name=agent_name,
                        step=step,
                        response=response
                    )

                    emit(
                        EventTypes.AGENT_RESPONSE_RECEIVED,
                        step=step,
                        agent_name=agent_name,
                        response=response
                    )

                    agent_responses[agent_name] = response

        # Add response messages to step_messages (in agent order for consistency)
        for agent in self.agents:
            if agent.name in agent_responses:
                step_messages.append({
                    "from": agent.name,
                    "to": "SimulatorAgent",
                    "content": agent_responses[agent.name]
                })

        return agent_responses, step_messages

    def _process_step_results(self, step: int, agent_responses: dict, step_messages: list) -> dict:
        """Process step results and return next agent messages."""
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

        # Emit step completed event with state snapshot
        snapshot = self.game_engine.get_state_snapshot()
        emit(
            EventTypes.STEP_COMPLETED,
            step=step,
            global_vars=snapshot.get("global_vars", {}),
            agent_vars=snapshot.get("agent_vars", {})
        )

        return agent_messages

    def _save_final_state(self, step_messages: list):
        """Save final simulation state."""
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

        # Emit simulation completed event
        final_snapshot = self.game_engine.get_state_snapshot()
        emit(
            EventTypes.SIMULATION_COMPLETED,
            step=self.max_steps,
            total_steps=self.max_steps,
            final_state=final_snapshot.get("game_state", {}),
            global_vars=final_snapshot.get("global_vars", {}),
            agent_vars=final_snapshot.get("agent_vars", {})
        )

        print("\nSimulation completed.")
        print(f"Final game state: {self.game_engine.get_state()}")
        print(f"\nResults saved to: {self.persistence.run_dir}")

    def run(self):
        """Run the simulation loop with SimulatorAgent."""
        enable_event_emitter(self.persistence.run_id)

        self.logger.info(
            MessageCode.SIM001,
            "Simulation started",
            num_agents=len(self.agents),
            max_steps=self.max_steps
        )

        # Emit simulation started event
        spatial_graph_data = self.spatial_graph.to_dict() if self.spatial_graph else None
        geojson_data = self.composed_modules.geojson if self.composed_modules else None
        emit(
            EventTypes.SIMULATION_STARTED,
            num_agents=len(self.agents),
            max_steps=self.max_steps,
            agent_names=[a.name for a in self.agents],
            run_dir=str(self.persistence.run_dir),
            spatial_graph=spatial_graph_data,
            geojson=geojson_data
        )

        print(f"Starting simulation with {len(self.agents)} agent(s) for {self.max_steps} steps")
        print(f"Results will be saved to: {self.persistence.run_dir}\n")

        step_messages = []
        try:
            agent_messages = self._initialize_simulation()

            for step in range(1, self.max_steps + 1):
                with PerformanceTimer(self.logger, MessageCode.PRF001, f"Step {step}", step=step):
                    self.logger.info(MessageCode.SIM003, "Step started", step=step, max_steps=self.max_steps)
                    emit(EventTypes.STEP_STARTED, step=step, max_steps=self.max_steps)
                    print(f"\n=== Step {step}/{self.max_steps} ===")

                    agent_responses, step_messages = self._collect_agent_responses(step, agent_messages)
                    agent_messages = self._process_step_results(step, agent_responses, step_messages)

            self._save_final_state(step_messages)

        except KeyboardInterrupt:
            self.logger.warning(MessageCode.SIM002, "Simulation interrupted by user")
            print("\n\nSimulation interrupted by user")
            raise

        except SimulationError:
            raise

        except Exception as e:
            self.logger.critical(MessageCode.SIM002, "Simulation failed with unhandled exception", error=str(e))
            raise

        finally:
            disable_event_emitter()
            self.persistence.close()
