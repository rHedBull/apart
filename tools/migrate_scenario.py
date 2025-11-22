#!/usr/bin/env python3
"""
Migration tool to convert old scenario format to new engine-powered format.
"""

import sys
import yaml
from pathlib import Path


def migrate_scenario(input_path: str, output_path: str = None):
    """Migrate old scenario to new format."""
    # Load old scenario
    with open(input_path, 'r') as f:
        old_config = yaml.safe_load(f)

    # Create new config
    new_config = {
        "max_steps": old_config.get("max_steps", 10)
    }

    # Prompt for engine configuration
    print("\n=== Engine Configuration ===")
    print("The new format requires an LLM-powered simulation engine.")
    print()

    provider = input("LLM Provider (gemini/ollama) [gemini]: ").strip() or "gemini"

    if provider == "gemini":
        model = input("Model [gemini-1.5-flash]: ").strip() or "gemini-1.5-flash"
    else:
        model = input("Model [llama2]: ").strip() or "llama2"

    print("\nEngine System Prompt (multi-line, press Ctrl+D when done):")
    print("Example: You are the game master. Maintain realistic simulation.")
    system_prompt_lines = []
    try:
        while True:
            line = input()
            system_prompt_lines.append(line)
    except EOFError:
        pass
    system_prompt = "\n".join(system_prompt_lines)

    print("\nSimulation Plan (multi-line, press Ctrl+D when done):")
    print("Example: Simulate a 10-step economic scenario with gradual changes.")
    plan_lines = []
    try:
        while True:
            line = input()
            plan_lines.append(line)
    except EOFError:
        pass
    simulation_plan = "\n".join(plan_lines)

    new_config["engine"] = {
        "provider": provider,
        "model": model,
        "system_prompt": system_prompt,
        "simulation_plan": simulation_plan,
        "context_window_size": 5
    }

    # Convert game_state to global_vars if exists
    if "game_state" in old_config:
        print("\nWARNING: game_state section found. This should be converted to global_vars.")
        print("Please manually review and convert.")

    # Copy global_vars if exists
    if "global_vars" in old_config:
        new_config["global_vars"] = old_config["global_vars"]

    # Copy agent_vars if exists
    if "agent_vars" in old_config:
        new_config["agent_vars"] = old_config["agent_vars"]

    # Copy agents
    if "agents" in old_config:
        new_config["agents"] = old_config["agents"]

    # Determine output path
    if output_path is None:
        input_stem = Path(input_path).stem
        output_path = f"scenarios/{input_stem}_migrated.yaml"

    # Save migrated config
    with open(output_path, 'w') as f:
        yaml.dump(new_config, f, default_flow_style=False, sort_keys=False)

    print(f"\n✓ Migrated scenario saved to: {output_path}")
    print("\nIMPORTANT: Please review the migrated scenario:")
    print("1. Ensure engine configuration is correct")
    print("2. Convert any game_state entries to global_vars")
    print("3. Test the scenario before using in production")


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/migrate_scenario.py <input.yaml> [output.yaml]")
        print("\nExample:")
        print("  python tools/migrate_scenario.py scenarios/old_scenario.yaml")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    if not Path(input_path).exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    migrate_scenario(input_path, output_path)


if __name__ == "__main__":
    main()
