import argparse
from pathlib import Path
from orchestrator import Orchestrator


def main():
    parser = argparse.ArgumentParser(description="Run multi-agent simulation")
    parser.add_argument(
        "scenario",
        nargs="?",
        default="scenarios/config.yaml",
        help="Path to scenario configuration file (default: scenarios/config.yaml)"
    )
    parser.add_argument(
        "--save-frequency",
        "-sf",
        type=int,
        default=1,
        help="Save frequency: 0=final only, N=every N steps (default: 1)"
    )

    args = parser.parse_args()

    # Extract scenario name from path for persistence
    scenario_path = args.scenario
    scenario_name = Path(scenario_path).stem

    orchestrator = Orchestrator(scenario_path, scenario_name, args.save_frequency)
    orchestrator.run()


if __name__ == "__main__":
    main()
