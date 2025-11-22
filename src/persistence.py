import json
from pathlib import Path
from datetime import datetime
from typing import Any


class RunPersistence:
    """Manages persistence of simulation run data."""

    def __init__(self, scenario_name: str, save_frequency: int):
        """
        Initialize persistence layer.

        Args:
            scenario_name: Name of the scenario being run
            save_frequency: How often to save (0=final only, N=every N steps)
        """
        self.scenario_name = self._sanitize_scenario_name(scenario_name)
        self.save_frequency = save_frequency
        self.run_id = self._generate_run_id()
        self.run_dir = self._create_run_directory()
        self.state_file = self.run_dir / "state.json"
        self._initialize_state_file()

    def _sanitize_scenario_name(self, name: str) -> str:
        """Sanitize scenario name for use in filesystem paths."""
        # Remove directory separators and replace invalid chars
        name = name.replace("/", "_").replace("\\", "_").replace(" ", "_")
        # Remove other problematic characters
        invalid_chars = '<>:"|?*'
        for char in invalid_chars:
            name = name.replace(char, "_")
        # Truncate to 50 characters
        return name[:50]

    def _generate_run_id(self) -> str:
        """Generate unique run ID with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return f"run_{self.scenario_name}_{timestamp}"

    def _create_run_directory(self) -> Path:
        """Create unique run directory under results/."""
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        run_dir = results_dir / self.run_id

        # Handle collision: append _1, _2, etc. if directory exists
        counter = 1
        while run_dir.exists():
            run_dir = results_dir / f"{self.run_id}_{counter}"
            counter += 1

        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _initialize_state_file(self):
        """Initialize state.json with metadata."""
        initial_data = {
            "run_id": self.run_id,
            "scenario": self.scenario_name,
            "started_at": datetime.now().isoformat(),
            "snapshots": []
        }
        self._write_state(initial_data)

    def _read_state(self) -> dict[str, Any]:
        """Read current state from file."""
        try:
            with open(self.state_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # If file doesn't exist or is corrupted, return empty structure
            return {
                "run_id": self.run_id,
                "scenario": self.scenario_name,
                "started_at": datetime.now().isoformat(),
                "snapshots": []
            }

    def _write_state(self, data: dict[str, Any]):
        """Write state to file atomically."""
        # Write to temp file first, then rename for atomicity
        temp_file = self.state_file.with_suffix(".json.tmp")
        try:
            with open(temp_file, "w") as f:
                json.dump(data, f, indent=2)
            temp_file.replace(self.state_file)
        except Exception as e:
            print(f"Warning: Failed to save state: {e}")
            if temp_file.exists():
                temp_file.unlink()

    def should_save(self, step: int) -> bool:
        """Determine if current step should be saved based on save_frequency."""
        if self.save_frequency == 0:
            return False
        return step % self.save_frequency == 0

    def save_snapshot(
        self,
        step: int,
        game_state: dict,
        global_vars: dict,
        agent_vars: dict,
        messages: list[dict]
    ):
        """Append snapshot to state.json."""
        state = self._read_state()

        snapshot = {
            "step": step,
            "game_state": game_state,
            "global_vars": global_vars,
            "agent_vars": agent_vars,
            "messages": messages
        }

        state["snapshots"].append(snapshot)
        self._write_state(state)

    def save_final(
        self,
        step: int,
        game_state: dict,
        global_vars: dict,
        agent_vars: dict,
        messages: list[dict]
    ):
        """Always save final state regardless of frequency."""
        self.save_snapshot(step, game_state, global_vars, agent_vars, messages)
