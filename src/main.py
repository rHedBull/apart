from orchestrator import Orchestrator


def main():
    orchestrator = Orchestrator("scenarios/config.yaml")
    orchestrator.run()


if __name__ == "__main__":
    main()
