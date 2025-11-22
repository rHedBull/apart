import sys
from orchestrator import Orchestrator


def main():
    # Check if a scenario file path was provided as argument
    scenario_path = sys.argv[1] if len(sys.argv) > 1 else "scenarios/config.yaml"

    orchestrator = Orchestrator(scenario_path)
    orchestrator.run()


if __name__ == "__main__":
    main()
