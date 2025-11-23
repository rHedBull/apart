import json
from typing import Any, Dict, List, Optional
from llm.llm_provider import LLMProvider
from core.game_engine import GameEngine
from core.engine_models import ScriptedEvent, EngineOutput, ConstraintHit, StepRecord
from core.engine_validator import EngineValidator, ValidationResult
from utils.logging_config import MessageCode


class SimulationError(Exception):
    """Fatal error in simulation that requires stopping."""
    pass


class SimulatorAgent:
    """LLM-powered simulation orchestrator - the game master."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        game_engine: GameEngine,
        system_prompt: str,
        simulation_plan: str,
        realism_guidelines: str,
        scripted_events: List[ScriptedEvent],
        context_window_size: int = 5,
        logger=None
    ):
        self.llm_provider = llm_provider
        self.game_engine = game_engine
        self.system_prompt = system_prompt
        self.simulation_plan = simulation_plan
        self.realism_guidelines = realism_guidelines
        self.scripted_events = scripted_events
        self.context_window_size = context_window_size
        self.logger = logger

        self.step_history: List[StepRecord] = []
        self.constraint_feedback: List[ConstraintHit] = []

        if self.logger:
            self.logger.info(
                MessageCode.ENG001,
                "SimulatorAgent initialized",
                context_window_size=context_window_size,
                num_scripted_events=len(scripted_events)
            )

    def initialize_simulation(self, agent_names: List[str]) -> Dict[str, str]:
        """Generate initial agent messages for Step 0."""
        current_state = self.game_engine.get_current_state()

        prompt = self._build_initialization_prompt(agent_names, current_state)

        # Call LLM with retry logic
        output = self._call_llm_with_retry(prompt, step_number=0, agent_names=agent_names)

        # Apply state updates
        self.game_engine.apply_state_updates(output.state_updates)

        return output.agent_messages

    def process_step(
        self,
        step_number: int,
        agent_responses: Dict[str, str]
    ) -> Dict[str, str]:
        """
        Process agent responses, update world state, generate new messages.
        Returns: agent_messages for next step
        """
        if self.logger:
            self.logger.info(MessageCode.ENG002, "Processing step", step=step_number)

        current_state = self.game_engine.get_current_state()
        agent_names = list(agent_responses.keys())

        # Build context
        prompt = self._build_step_prompt(step_number, current_state, agent_responses)

        # Call LLM with retry logic
        output = self._call_llm_with_retry(prompt, step_number, agent_names)

        # Apply constraints and get clamped updates
        clamped_updates, constraint_hits = EngineValidator.apply_constraints(
            output.state_updates,
            self.game_engine.global_var_definitions,
            self.game_engine.agent_var_definitions
        )

        # Log constraint hits
        for hit in constraint_hits:
            if self.logger:
                self.logger.warning(
                    MessageCode.ENG009,
                    "Constraint hit",
                    step=step_number,
                    agent=hit.agent_name,
                    var=hit.var_name,
                    attempted=hit.attempted_value,
                    clamped=hit.clamped_value,
                    constraint_type=hit.constraint_type
                )

        # Apply clamped updates to game engine
        self.game_engine.apply_state_updates(clamped_updates)

        if self.logger:
            self.logger.info(
                MessageCode.ENG010,
                "State updates applied",
                step=step_number,
                num_global_changes=len(clamped_updates["global_vars"]),
                num_agent_changes=sum(len(v) for v in clamped_updates["agent_vars"].values())
            )

        # Record step in history
        self._record_step(step_number, clamped_updates, output.events, agent_responses, output.reasoning, constraint_hits)

        # Store constraint hits for next step's context
        self.constraint_feedback = constraint_hits

        return output.agent_messages

    def _call_llm_with_retry(
        self,
        prompt: str,
        step_number: int,
        agent_names: List[str],
        max_attempts: int = 3
    ) -> EngineOutput:
        """Call LLM with validation and retry logic."""
        attempt = 1
        last_error = None

        while attempt <= max_attempts:
            if self.logger and attempt > 1:
                self.logger.info(
                    MessageCode.ENG007,
                    f"Retry attempt {attempt}/{max_attempts}",
                    step=step_number
                )

            try:
                if self.logger:
                    self.logger.debug(MessageCode.ENG003, "Calling LLM", step=step_number)

                # Use structured JSON output for Gemini
                from llm.providers import GeminiProvider
                if isinstance(self.llm_provider, GeminiProvider):
                    response_text = self.llm_provider.generate_response(
                        prompt=prompt,
                        system_prompt=self.system_prompt,
                        force_json=True
                    )
                else:
                    response_text = self.llm_provider.generate_response(
                        prompt=prompt,
                        system_prompt=self.system_prompt
                    )

                if self.logger:
                    self.logger.debug(
                        MessageCode.ENG004,
                        "LLM response received",
                        step=step_number,
                        response_length=len(response_text)
                    )

                # Validate response
                validation_result = self._validate_response(response_text, agent_names)

                if validation_result.success:
                    output = self._parse_response(response_text)

                    if self.logger:
                        self.logger.info(
                            MessageCode.ENG005,
                            "Validation successful",
                            step=step_number
                        )

                    return output
                else:
                    last_error = validation_result.error

                    if self.logger:
                        self.logger.warning(
                            MessageCode.ENG006,
                            "Validation failed",
                            step=step_number,
                            error=validation_result.error,
                            attempt=attempt
                        )

                    # Add error feedback to prompt for retry
                    prompt = self._add_error_feedback(prompt, validation_result.error)

            except Exception as e:
                last_error = str(e)

                if self.logger:
                    self.logger.error(
                        MessageCode.ENG006,
                        "LLM call failed",
                        step=step_number,
                        error=str(e),
                        attempt=attempt
                    )

            attempt += 1

        # Max attempts exceeded
        error_msg = (
            f"SimulatorAgent failed after {max_attempts} attempts at step {step_number}.\n"
            f"Last error: {last_error}"
        )

        if self.logger:
            self.logger.critical(MessageCode.ENG008, error_msg, step=step_number)

        raise SimulationError(error_msg)

    def _validate_response(self, response_text: str, agent_names: List[str]) -> ValidationResult:
        """Validate LLM response."""
        # Step 1: Structure validation
        result = EngineValidator.validate_structure(response_text)
        if not result.success:
            return result

        # Parse JSON
        output = json.loads(response_text)

        # Step 2: Reference validation
        result = EngineValidator.validate_references(
            output,
            agent_names,
            self.game_engine.global_var_definitions,
            self.game_engine.agent_var_definitions
        )
        if not result.success:
            return result

        # Step 3: Type validation
        result = EngineValidator.validate_types(
            output,
            self.game_engine.global_var_definitions,
            self.game_engine.agent_var_definitions
        )
        if not result.success:
            return result

        return ValidationResult(success=True)

    def _parse_response(self, response_text: str) -> EngineOutput:
        """Parse validated response into EngineOutput."""
        # Strip markdown code blocks (validation already did this, but be safe)
        from core.engine_validator import EngineValidator
        cleaned_text = EngineValidator._strip_markdown_code_blocks(response_text)
        data = json.loads(cleaned_text)
        return EngineOutput(
            state_updates=data["state_updates"],
            events=data["events"],
            agent_messages=data["agent_messages"],
            reasoning=data["reasoning"]
        )

    def _record_step(
        self,
        step_number: int,
        changes: Dict[str, Any],
        events: List[Dict[str, Any]],
        agent_responses: Dict[str, str],
        reasoning: str,
        constraint_hits: List[ConstraintHit]
    ):
        """Record step in history with sliding window."""
        record = StepRecord(
            step_number=step_number,
            changes=changes,
            events=events,
            agent_responses=agent_responses,
            reasoning=reasoning,
            constraint_hits=constraint_hits
        )

        self.step_history.append(record)

        # Maintain window size
        if len(self.step_history) > self.context_window_size:
            self.step_history.pop(0)

    def _build_initialization_prompt(self, agent_names: List[str], current_state: Dict[str, Any]) -> str:
        """Build prompt for step 0 initialization."""
        # Get available variable names
        global_var_names = list(self.game_engine.global_var_definitions.keys())
        agent_var_names = list(self.game_engine.agent_var_definitions.keys())

        # Format variable types for clarity
        global_var_types = {name: defn["type"] for name, defn in self.game_engine.global_var_definitions.items()}
        agent_var_types = {name: defn["type"] for name, defn in self.game_engine.agent_var_definitions.items()}

        prompt = f"""=== SIMULATION SETUP ===
{self.system_prompt}

Simulation Plan:
{self.simulation_plan}

Realism Guidelines:
{self.realism_guidelines}

=== INITIAL STATE ===
Global Variables (Available with types):
{self._format_variable_definitions(self.game_engine.global_var_definitions)}
Current Values:
{self._format_variables(current_state["global_vars"])}

Agents: {', '.join(agent_names)}
Per-Agent Variables (Available with types):
{self._format_variable_definitions(self.game_engine.agent_var_definitions)}
Current Values:
{self._format_agent_variables(current_state["agent_vars"])}

=== YOUR TASK ===
Initialize the simulation for Step 0.
Generate initial personalized messages for each agent to begin the simulation.

Return ONLY valid JSON (no markdown, no code blocks):
{{
  "state_updates": {{
    "global_vars": {{}},
    "agent_vars": {{}}
  }},
  "events": [],
  "agent_messages": {{
    "Agent A": "Initial message for Agent A",
    "Agent B": "Initial message for Agent B"
  }},
  "reasoning": "Why you generated these initial messages"
}}

CRITICAL JSON RULES - READ CAREFULLY:
- Return ONLY the JSON object
- Use ACTUAL NUMERIC VALUES ONLY - calculate them yourself, then put the result
  WRONG: "capital": capital * 1.05
  WRONG: "interest_rate": 0.04 + 0.01
  CORRECT: "capital": 1050.0
  CORRECT: "interest_rate": 0.05
- Do NOT use Math.random(), variable names, or any calculations in JSON values
- Do NOT use comments (//) - they are invalid in JSON
- Include ALL agents in agent_messages
- ONLY use variables listed above - no other variables exist
- Match variable types exactly: int=integer number, float=decimal number, bool=true/false, list=[...], dict={...}

REMEMBER: Calculate values in your head and write the FINAL NUMBERS in the JSON.
"""
        return prompt

    def _build_step_prompt(self, step_number: int, current_state: Dict[str, Any], agent_responses: Dict[str, str]) -> str:
        """Build prompt for regular step processing."""
        sections = []

        # Setup section
        sections.append(f"""=== SIMULATION SETUP ===
{self.system_prompt}

Simulation Plan:
{self.simulation_plan}

Realism Guidelines:
{self.realism_guidelines}""")

        # Scripted events
        if self.scripted_events:
            upcoming = [e for e in self.scripted_events if e.step >= step_number]
            if upcoming:
                sections.append("=== UPCOMING SCRIPTED EVENTS ===")
                for event in upcoming[:5]:  # Show next 5 events
                    sections.append(f"Step {event.step}: {event.type} - {event.description}")

        # Current state
        sections.append(f"""=== CURRENT STATE (Step {step_number}) ===
Global Variables:
{self._format_variables(current_state["global_vars"])}

Agent Variables:""")
        for agent_name, vars_dict in current_state["agent_vars"].items():
            sections.append(f"  {agent_name}:")
            sections.append(f"{self._format_variables(vars_dict, indent='    ')}")

        # Recent history
        if self.step_history:
            sections.append(f"=== RECENT HISTORY (Last {len(self.step_history)} steps) ===")
            for record in self.step_history:
                sections.append(f"\nStep {record.step_number}:")

                # Changes (delta only)
                if record.changes["global_vars"] or record.changes["agent_vars"]:
                    sections.append("  Changes:")
                    for var_name, value in record.changes["global_vars"].items():
                        sections.append(f"    Global: {var_name} = {value}")
                    for agent_name, vars_dict in record.changes["agent_vars"].items():
                        for var_name, value in vars_dict.items():
                            sections.append(f"    {agent_name}: {var_name} = {value}")

                # Events
                if record.events:
                    sections.append("  Events:")
                    for event in record.events:
                        sections.append(f"    - {event}")

                # Agent responses
                sections.append("  Agent Responses:")
                for agent_name, response in record.agent_responses.items():
                    sections.append(f"    {agent_name}: \"{response}\"")

                # Reasoning
                sections.append(f"  Reasoning: {record.reasoning}")

                # Constraint hits
                if record.constraint_hits:
                    sections.append("  Constraint Hits:")
                    for hit in record.constraint_hits:
                        agent_str = hit.agent_name or "Global"
                        sections.append(
                            f"    {agent_str} {hit.var_name}: attempted {hit.attempted_value}, "
                            f"clamped to {hit.clamped_value} ({hit.constraint_type})"
                        )

        # Task
        sections.append(f"""=== YOUR TASK ===
Process the agent responses from Step {step_number - 1 if step_number > 1 else 0}.
Update world state realistically for Step {step_number}.
Generate personalized messages for each agent.

Agent Responses from previous step:""")
        for agent_name, response in agent_responses.items():
            sections.append(f"  {agent_name}: \"{response}\"")

        sections.append("""
Return ONLY valid JSON (no markdown, no code blocks):
{
  "state_updates": {
    "global_vars": {"var_name": new_value, ...},
    "agent_vars": {
      "Agent A": {"var_name": new_value, ...},
      "Agent B": {...}
    }
  },
  "events": [
    {
      "type": "event_type",
      "description": "What happened",
      "affects": ["Agent A"],
      "duration": 3
    }
  ],
  "agent_messages": {
    "Agent A": "Personalized narrative for Agent A",
    "Agent B": "Personalized narrative for Agent B"
  },
  "reasoning": "Why you made these updates and generated these events"
}

CRITICAL JSON RULES - READ CAREFULLY:
- Return ONLY the JSON object, no explanation text
- Use ACTUAL NUMERIC VALUES ONLY - calculate them yourself, then put the result
  WRONG: "capital": MistralStrategist.capital * 1.05
  WRONG: "capital": capital * 1.05
  WRONG: "market_volatility": market_volatility + Math.random() * 0.03
  WRONG: "interest_rate": interest_rate + 0.005
  CORRECT: "capital": 2100.0
  CORRECT: "market_volatility": 0.163
  CORRECT: "interest_rate": 0.045
- Do NOT use Math.random(), variable names, or any calculations in JSON values
- Do NOT use comments (//) - they are invalid in JSON
- Only include variables that CHANGED in state_updates
- Include ALL agents in agent_messages
- ONLY use variables shown in the "CURRENT STATE" section above - no other variables exist
- Match variable types exactly: int=integer number (e.g., 5), float=decimal (e.g., 5.0), bool=true/false

REMEMBER: You must calculate the new values in your head and write the FINAL NUMBERS in the JSON.
""")

        return "\n".join(sections)

    def _format_variables(self, vars_dict: Dict[str, Any], indent: str = "  ") -> str:
        """Format variables for prompt."""
        if not vars_dict:
            return f"{indent}(none)"
        lines = []
        for name, value in vars_dict.items():
            lines.append(f"{indent}{name}: {value}")
        return "\n".join(lines)

    def _format_agent_variables(self, agent_vars: Dict[str, Dict[str, Any]]) -> str:
        """Format per-agent variables for prompt."""
        lines = []
        for agent_name, vars_dict in agent_vars.items():
            lines.append(f"  {agent_name}:")
            for var_name, value in vars_dict.items():
                lines.append(f"    {var_name}: {value}")
        return "\n".join(lines)

    def _format_variable_definitions(self, var_defs: Dict[str, Any]) -> str:
        """Format variable definitions with types and constraints."""
        lines = []
        for name, defn in var_defs.items():
            type_str = defn["type"]
            constraints = []
            if "min" in defn:
                constraints.append(f"min={defn['min']}")
            if "max" in defn:
                constraints.append(f"max={defn['max']}")
            constraint_str = f" ({', '.join(constraints)})" if constraints else ""
            lines.append(f"  {name}: {type_str}{constraint_str}")
        return "\n".join(lines)

    def _add_error_feedback(self, prompt: str, error: str) -> str:
        """Add error feedback to prompt for retry."""
        return f"{prompt}\n\nERROR: Your previous response was invalid: {error}\nPlease correct and try again."
