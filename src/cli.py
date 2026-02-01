"""APART CLI - Command-line interface for running simulations.

This CLI provides commands to interact with the APART simulation server,
allowing users to start, pause, resume, list, and inspect simulation runs.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="apart",
    help="APART - Multi-agent simulation framework for AI safety research",
    add_completion=False,
)

console = Console()
DEFAULT_API_URL = "http://localhost:8000"


def get_api_url() -> str:
    """Get API URL from environment or use default."""
    return os.environ.get("APART_API_URL", DEFAULT_API_URL)


def _handle_request_error(e: Exception, action: str) -> None:
    """Handle request errors with user-friendly messages."""
    console.print(f"[red]Error {action}:[/red] {e}")
    console.print(f"[dim]Make sure the APART server is running at {get_api_url()}[/dim]")
    raise typer.Exit(1)


@app.command()
def run(
    scenario: Path = typer.Argument(
        ...,
        help="Path to the scenario YAML file",
        exists=True,
        dir_okay=False,
        resolve_path=True,
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name", "-n",
        help="Custom name/ID for the simulation run",
    ),
    priority: str = typer.Option(
        "normal",
        "--priority", "-p",
        help="Job queue priority (high, normal, low)",
    ),
) -> None:
    """Start a new simulation by submitting to the job queue.

    Example:
        apart run scenarios/taiwan_strait.yaml --name my_run --priority high
    """
    import requests

    api_url = get_api_url()

    # Validate priority
    valid_priorities = ["high", "normal", "low"]
    if priority.lower() not in valid_priorities:
        console.print(f"[red]Invalid priority:[/red] {priority}")
        console.print(f"[dim]Valid options: {', '.join(valid_priorities)}[/dim]")
        raise typer.Exit(1)

    # Build request payload
    payload = {
        "scenario_path": str(scenario),
        "priority": priority.lower(),
    }
    if name:
        payload["run_id"] = name

    try:
        response = requests.post(f"{api_url}/api/v1/runs", json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()

        console.print(f"[green]Simulation started successfully![/green]")
        console.print(f"  Run ID: [bold]{data['run_id']}[/bold]")
        console.print(f"  Status: {data['status']}")
        console.print(f"  Message: {data['message']}")
        console.print()
        console.print(f"[dim]Track progress: apart show {data['run_id']}[/dim]")

    except requests.exceptions.HTTPError as e:
        if e.response is not None:
            try:
                error_detail = e.response.json().get("detail", str(e))
            except Exception:
                error_detail = e.response.text or str(e)
            console.print(f"[red]Error starting simulation:[/red] {error_detail}")
        else:
            console.print(f"[red]Error starting simulation:[/red] {e}")
        raise typer.Exit(1)
    except requests.exceptions.RequestException as e:
        _handle_request_error(e, "starting simulation")


@app.command()
def pause(
    run_id: str = typer.Argument(
        ...,
        help="ID of the simulation run to pause",
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Force immediate pause (drop current step)",
    ),
) -> None:
    """Pause a running simulation.

    The simulation will pause at the end of the current step unless --force
    is specified, which pauses immediately (potentially losing the current step).

    Example:
        apart pause my_run_id
        apart pause my_run_id --force
    """
    import requests

    api_url = get_api_url()

    try:
        params = {"force": "true"} if force else {}
        response = requests.post(
            f"{api_url}/api/v1/runs/{run_id}/pause",
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        console.print(f"[yellow]Pause requested for simulation {run_id}[/yellow]")
        console.print(f"  Status: {data['status']}")
        console.print(f"  Message: {data['message']}")
        console.print()
        console.print(f"[dim]Resume with: apart resume {run_id}[/dim]")

    except requests.exceptions.HTTPError as e:
        if e.response is not None:
            try:
                error_detail = e.response.json().get("detail", str(e))
            except Exception:
                error_detail = e.response.text or str(e)
            console.print(f"[red]Error pausing simulation:[/red] {error_detail}")
        else:
            console.print(f"[red]Error pausing simulation:[/red] {e}")
        raise typer.Exit(1)
    except requests.exceptions.RequestException as e:
        _handle_request_error(e, "pausing simulation")


@app.command()
def resume(
    run_id: str = typer.Argument(
        ...,
        help="ID of the paused simulation to resume",
    ),
) -> None:
    """Resume a paused simulation.

    The simulation will continue from where it was paused.

    Example:
        apart resume my_run_id
    """
    import requests

    api_url = get_api_url()

    try:
        response = requests.post(
            f"{api_url}/api/v1/runs/{run_id}/resume",
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        console.print(f"[green]Simulation {run_id} resumed![/green]")
        console.print(f"  Status: {data['status']}")
        console.print(f"  Resuming from step: {data['resuming_from_step']}")
        console.print(f"  Message: {data['message']}")

    except requests.exceptions.HTTPError as e:
        if e.response is not None:
            try:
                error_detail = e.response.json().get("detail", str(e))
            except Exception:
                error_detail = e.response.text or str(e)
            console.print(f"[red]Error resuming simulation:[/red] {error_detail}")
        else:
            console.print(f"[red]Error resuming simulation:[/red] {e}")
        raise typer.Exit(1)
    except requests.exceptions.RequestException as e:
        _handle_request_error(e, "resuming simulation")


@app.command("list")
def list_runs(
    status: Optional[str] = typer.Option(
        None,
        "--status", "-s",
        help="Filter by status (pending, running, paused, completed, failed)",
    ),
) -> None:
    """List all simulation runs.

    Shows a table of all runs with their status, progress, and danger signals.

    Example:
        apart list
        apart list --status running
        apart list -s paused
    """
    import requests

    api_url = get_api_url()

    try:
        response = requests.get(f"{api_url}/api/v1/runs", timeout=30)
        response.raise_for_status()
        data = response.json()

        runs = data.get("runs", [])

        # Filter by status if specified
        if status:
            status_lower = status.lower()
            runs = [r for r in runs if r.get("status", "").lower() == status_lower]

        if not runs:
            if status:
                console.print(f"[dim]No simulation runs found with status '{status}'[/dim]")
            else:
                console.print("[dim]No simulation runs found[/dim]")
            return

        # Create table
        table = Table(title="Simulation Runs")
        table.add_column("Run ID", style="cyan", no_wrap=True)
        table.add_column("Scenario", style="white")
        table.add_column("Status", style="white")
        table.add_column("Progress", justify="right")
        table.add_column("Danger Signals", justify="right", style="red")
        table.add_column("Started", style="dim")

        for run in runs:
            run_id = run.get("runId", "unknown")
            scenario = run.get("scenario", "unknown")
            run_status = run.get("status", "unknown")
            current_step = run.get("currentStep", 0)
            total_steps = run.get("totalSteps")
            danger_count = run.get("dangerCount", 0)
            started_at = run.get("startedAt", "")

            # Format status with color
            status_style = {
                "pending": "[dim]pending[/dim]",
                "running": "[green]running[/green]",
                "paused": "[yellow]paused[/yellow]",
                "completed": "[blue]completed[/blue]",
                "failed": "[red]failed[/red]",
                "interrupted": "[magenta]interrupted[/magenta]",
            }.get(run_status, run_status)

            # Format progress
            if total_steps:
                progress = f"{current_step}/{total_steps}"
            else:
                progress = str(current_step)

            # Format danger count
            danger_str = str(danger_count) if danger_count > 0 else "-"

            # Format started time (just show date/time portion)
            started_str = started_at[:19] if started_at else "-"

            table.add_row(
                run_id,
                scenario[:30] + "..." if len(scenario) > 30 else scenario,
                status_style,
                progress,
                danger_str,
                started_str,
            )

        console.print(table)
        console.print(f"\n[dim]Total: {len(runs)} run(s)[/dim]")

    except requests.exceptions.RequestException as e:
        _handle_request_error(e, "listing runs")


@app.command()
def show(
    run_id: str = typer.Argument(
        ...,
        help="ID of the simulation run to show",
    ),
) -> None:
    """Show details of a specific simulation run.

    Displays comprehensive information about a run including status,
    progress, agents, messages, and danger signals.

    Example:
        apart show my_run_id
    """
    import requests

    api_url = get_api_url()

    try:
        response = requests.get(f"{api_url}/api/v1/runs/{run_id}", timeout=30)
        response.raise_for_status()
        data = response.json()

        # Header
        console.print()
        console.print(f"[bold]Simulation: {data.get('runId', run_id)}[/bold]")
        console.print("=" * 50)

        # Basic info
        status = data.get("status", "unknown")
        status_style = {
            "pending": "[dim]pending[/dim]",
            "running": "[green]running[/green]",
            "paused": "[yellow]paused[/yellow]",
            "completed": "[blue]completed[/blue]",
            "failed": "[red]failed[/red]",
            "interrupted": "[magenta]interrupted[/magenta]",
        }.get(status, status)

        console.print(f"  Scenario: {data.get('scenario', 'unknown')}")
        console.print(f"  Status: {status_style}")

        current_step = data.get("currentStep", 0)
        max_steps = data.get("maxSteps")
        if max_steps:
            console.print(f"  Progress: {current_step}/{max_steps} steps")
        else:
            console.print(f"  Current step: {current_step}")

        if data.get("startedAt"):
            console.print(f"  Started: {data['startedAt']}")

        # Agents
        agents = data.get("agentNames", [])
        if agents:
            console.print(f"\n[bold]Agents ({len(agents)}):[/bold]")
            for agent in agents:
                console.print(f"  - {agent}")

        # Danger signals summary
        danger_signals = data.get("dangerSignals", [])
        if danger_signals:
            console.print(f"\n[bold red]Danger Signals ({len(danger_signals)}):[/bold red]")

            # Group by category
            by_category: dict[str, int] = {}
            for signal in danger_signals:
                cat = signal.get("category", "unknown")
                by_category[cat] = by_category.get(cat, 0) + 1

            for cat, count in sorted(by_category.items()):
                console.print(f"  - {cat}: {count}")

            # Show recent signals
            console.print("\n  [dim]Recent signals:[/dim]")
            for signal in danger_signals[-5:]:
                step = signal.get("step", "?")
                cat = signal.get("category", "unknown")
                agent = signal.get("agentName", "")
                metric = signal.get("metric", "")
                agent_str = f" ({agent})" if agent else ""
                console.print(f"    Step {step}: {cat}{agent_str} - {metric}")

        # Messages summary
        messages = data.get("messages", [])
        if messages:
            console.print(f"\n[bold]Messages ({len(messages)}):[/bold]")
            console.print(f"  [dim]Use API for full message history[/dim]")

            # Show last few messages
            console.print("\n  [dim]Recent messages:[/dim]")
            for msg in messages[-3:]:
                step = msg.get("step", "?")
                agent = msg.get("agentName", "unknown")
                direction = msg.get("direction", "")
                content = msg.get("content", "")[:80]
                if len(msg.get("content", "")) > 80:
                    content += "..."
                arrow = "->" if direction == "sent" else "<-"
                console.print(f"    Step {step} {arrow} {agent}: {content}")

        console.print()

    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            console.print(f"[red]Run not found:[/red] {run_id}")
        else:
            try:
                error_detail = e.response.json().get("detail", str(e)) if e.response else str(e)
            except Exception:
                error_detail = str(e)
            console.print(f"[red]Error fetching run details:[/red] {error_detail}")
        raise typer.Exit(1)
    except requests.exceptions.RequestException as e:
        _handle_request_error(e, "fetching run details")


@app.command()
def delete(
    run_id: str = typer.Argument(
        ...,
        help="ID of the simulation run to delete",
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Force delete even if simulation is running",
    ),
    yes: bool = typer.Option(
        False,
        "--yes", "-y",
        help="Skip confirmation prompt",
    ),
) -> None:
    """Delete a simulation run and all its data.

    This removes all data associated with the run including results,
    event history, and database records.

    Example:
        apart delete my_run_id
        apart delete my_run_id --force --yes
    """
    import requests

    api_url = get_api_url()

    # Confirm deletion
    if not yes:
        confirmed = typer.confirm(f"Delete simulation run '{run_id}' and all its data?")
        if not confirmed:
            console.print("[dim]Cancelled[/dim]")
            raise typer.Exit(0)

    try:
        params = {"force": "true"} if force else {}
        response = requests.delete(
            f"{api_url}/api/v1/runs/{run_id}",
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        console.print(f"[green]Deleted simulation {run_id}[/green]")
        if data.get("deleted_results"):
            console.print("  - Removed results directory")
        if data.get("deleted_events"):
            console.print("  - Cleared event history")
        if data.get("deleted_database"):
            console.print("  - Removed database records")

    except requests.exceptions.HTTPError as e:
        if e.response is not None:
            try:
                error_detail = e.response.json().get("detail", str(e))
            except Exception:
                error_detail = e.response.text or str(e)
            console.print(f"[red]Error deleting simulation:[/red] {error_detail}")
        else:
            console.print(f"[red]Error deleting simulation:[/red] {e}")
        raise typer.Exit(1)
    except requests.exceptions.RequestException as e:
        _handle_request_error(e, "deleting simulation")


@app.command()
def status() -> None:
    """Check the status of the APART server.

    Displays server health and queue statistics.
    """
    import requests

    api_url = get_api_url()

    console.print(f"[dim]Checking server at {api_url}...[/dim]\n")

    try:
        # Basic health check
        response = requests.get(f"{api_url}/api/health", timeout=10)
        response.raise_for_status()
        health_data = response.json()
        version = health_data.get("version", "unknown")

        console.print(f"[green]Server is healthy[/green] (v{version})")

        # Detailed health
        try:
            detailed = requests.get(f"{api_url}/api/health/detailed", timeout=10)
            detailed.raise_for_status()
            data = detailed.json()

            console.print(f"\n[bold]Server Status:[/bold]")
            console.print(f"  Tracked runs: {data.get('total_run_ids', 0)}")
            console.print(f"  Event subscribers: {data.get('event_bus_subscribers', 0)}")
            console.print(f"  Persistence mode: {data.get('persistence_mode', 'unknown')}")

            queue_stats = data.get("queue_stats", {})
            if queue_stats:
                # Stats are nested by priority, get totals
                totals = queue_stats.get("total", queue_stats)
                console.print(f"\n[bold]Job Queue:[/bold]")
                console.print(f"  Queued: {totals.get('queued', 0)}")
                console.print(f"  Running: {totals.get('started', 0)}")
                console.print(f"  Finished: {totals.get('finished', 0)}")
                console.print(f"  Failed: {totals.get('failed', 0)}")

        except Exception:
            # Detailed health not available, that's fine
            pass

    except requests.exceptions.ConnectionError:
        console.print(f"[red]Cannot connect to server at {api_url}[/red]")
        console.print("[dim]Make sure the APART server is running:[/dim]")
        console.print("[dim]  apart-server  # or: uvicorn src.server.app:app[/dim]")
        raise typer.Exit(1)
    except requests.exceptions.RequestException as e:
        console.print(f"[red]Error checking server status:[/red] {e}")
        raise typer.Exit(1)


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
