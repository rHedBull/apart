import argparse
import sys
import traceback
from pathlib import Path
from core.orchestrator import Orchestrator
from modules.loader import ModuleDependencyError, ModuleLoadError
from utils.config_parser import ModuleConfigError


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

    try:
        # Validate scenario file exists
        if not Path(scenario_path).exists():
            print(f"Error: Scenario file not found: {scenario_path}", file=sys.stderr)
            sys.exit(1)

        # Initialize and run orchestrator
        orchestrator = Orchestrator(scenario_path, scenario_name, args.save_frequency)
        orchestrator.run()

    except FileNotFoundError as e:
        print(f"Error: Required file not found: {e}", file=sys.stderr)
        sys.exit(1)

    except ValueError as e:
        print(f"Configuration Error: {e}", file=sys.stderr)
        sys.exit(1)

    except ModuleDependencyError as e:
        print(f"\n[MODULE ERROR] Missing dependency", file=sys.stderr)
        print(f"  {e}", file=sys.stderr)
        print(f"\nFix: Add the missing module to your scenario's 'modules' list.", file=sys.stderr)
        sys.exit(1)

    except ModuleLoadError as e:
        print(f"\n[MODULE ERROR] Failed to load module", file=sys.stderr)
        print(f"  {e}", file=sys.stderr)
        sys.exit(1)

    except ModuleConfigError as e:
        print(f"\n[MODULE ERROR] Invalid module configuration", file=sys.stderr)
        print(f"  {e}", file=sys.stderr)
        print(f"\nFix: Add required config in 'module_config' section of your scenario.", file=sys.stderr)
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user", file=sys.stderr)
        sys.exit(130)  # Standard exit code for SIGINT

    except Exception as e:
        print(f"\nFatal Error: {e}", file=sys.stderr)
        print("\nFull traceback:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
