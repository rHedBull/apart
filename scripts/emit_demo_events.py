#!/usr/bin/env python3
"""Emit demo events to test the dashboard."""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from server.event_bus import EventBus, emit_event
from server.routes.simulations import register_simulation, update_simulation_step, update_simulation_state, complete_simulation
from server.models import SimulationStatus

def main():
    run_id = "demo_live"

    # Register simulation
    register_simulation(
        run_id=run_id,
        scenario_name="Dashboard Demo",
        max_steps=3,
        agents=[
            {"name": "Trader Alice", "llm": {"provider": "anthropic", "model": "claude-3-haiku"}},
            {"name": "Trader Bob", "llm": {"provider": "anthropic", "model": "claude-3-haiku"}}
        ]
    )

    # Emit simulation started
    emit_event(
        "simulation_started",
        run_id=run_id,
        step=0,
        num_agents=2,
        max_steps=3,
        agent_names=["Trader Alice", "Trader Bob"]
    )
    print(f"Emitted: simulation_started")
    time.sleep(1)

    # Simulate 3 steps
    messages = [
        ("Trader Alice", "The market looks interesting today. I'm ready to negotiate.", "I prefer safe deals and building trust. What do you have to offer?"),
        ("Trader Bob", "Market price seems stable. Let's explore potential trading strategies.", "I'm always hunting for an edge - stable markets can hide opportunities."),
    ]

    for step in range(1, 4):
        emit_event("step_started", run_id=run_id, step=step, max_steps=3)
        update_simulation_step(run_id, step)
        print(f"Emitted: step_started (step {step})")
        time.sleep(0.5)

        for agent_name, msg, response in messages:
            emit_event(
                "agent_message_sent",
                run_id=run_id,
                step=step,
                agent_name=agent_name,
                message=msg
            )
            print(f"Emitted: agent_message_sent ({agent_name})")
            time.sleep(0.3)

            emit_event(
                "agent_response_received",
                run_id=run_id,
                step=step,
                agent_name=agent_name,
                response=response
            )
            print(f"Emitted: agent_response_received ({agent_name})")
            time.sleep(0.3)

        # Update state
        global_vars = {"market_price": 50 + step * 2}
        agent_vars = {
            "Trader Alice": {"gold": 100 + step * 5, "reputation": 60 + step},
            "Trader Bob": {"gold": 100 - step * 3, "reputation": 40 + step * 2}
        }

        emit_event(
            "step_completed",
            run_id=run_id,
            step=step,
            global_vars=global_vars,
            agent_vars=agent_vars
        )
        update_simulation_state(run_id, global_vars=global_vars, agent_vars=agent_vars)
        print(f"Emitted: step_completed (step {step})")
        time.sleep(0.5)

    # Emit simulation completed
    emit_event(
        "simulation_completed",
        run_id=run_id,
        step=3,
        total_steps=3
    )
    complete_simulation(run_id, SimulationStatus.COMPLETED)
    print("Emitted: simulation_completed")
    print("\nDone! Check the dashboard.")

if __name__ == "__main__":
    main()
