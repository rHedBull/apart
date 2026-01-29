"""
SSE (Server-Sent Events) endpoint for real-time simulation updates.

This provides a streaming connection for clients to receive
simulation events as they happen.
"""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
import asyncio

from server.event_bus import EventBus

router = APIRouter(prefix="/api/events", tags=["events"])


async def event_generator(run_id: str | None = None, include_history: bool = False):
    """
    Generate SSE events from the event bus.

    Args:
        run_id: Optional filter for specific simulation
        include_history: Whether to include historical events
    """
    bus = EventBus.get_instance()

    # Send initial connection event
    yield {
        "event": "connected",
        "data": f'{{"run_id": "{run_id or "all"}"}}'
    }

    async for event in bus.subscribe(run_id=run_id, include_history=include_history):
        yield {
            "event": event.event_type,
            "data": event.to_sse().strip().replace("data: ", "")
        }


@router.get("/stream")
async def stream_all_events(history: bool = False):
    """
    Stream all simulation events via SSE.

    Query params:
        history: If true, include historical events first

    Events:
        - connected: Initial connection confirmation
        - simulation_started: New simulation begins
        - step_started: New step begins
        - agent_message_sent: Message sent to agent
        - agent_response_received: Agent responded
        - state_updated: Simulation state changed
        - danger_signal: Danger signal detected
        - simulation_completed: Simulation finished
    """
    return EventSourceResponse(event_generator(include_history=history))


@router.get("/stream/{run_id}")
async def stream_run_events(run_id: str, history: bool = False):
    """
    Stream events for a specific simulation run.

    Path params:
        run_id: The simulation run to stream

    Query params:
        history: If true, include historical events first
    """
    return EventSourceResponse(event_generator(run_id=run_id, include_history=history))


# Simple health check that also tests SSE
@router.get("/test")
async def test_events():
    """
    Test endpoint - emits a test event.

    Useful for verifying SSE connectivity.
    """
    from server.event_bus import emit_event

    emit_event(
        "test_event",
        run_id="test",
        step=0,
        message="This is a test event"
    )

    return {"status": "ok", "message": "Test event emitted"}


@router.post("/demo")
async def demo_simulation():
    """
    Run a demo simulation with fake events.

    This emits a series of events to test the dashboard.
    All status tracking is done via EventBus events only.
    """
    import asyncio
    from server.event_bus import emit_event

    run_id = f"demo_{int(asyncio.get_event_loop().time())}"

    # Emit simulation started - this is the single source of truth for status
    emit_event(
        "simulation_started",
        run_id=run_id,
        step=0,
        num_agents=2,
        max_steps=3,
        agent_names=["Trader Alice", "Trader Bob"],
        scenario_name="Dashboard Demo"
    )

    messages = [
        ("Trader Alice", "The market looks interesting today.", "I prefer safe deals. What do you have to offer?"),
        ("Trader Bob", "Market seems stable. Let's trade.", "I'm hunting for an edge in this market."),
    ]

    for step in range(1, 4):
        await asyncio.sleep(0.3)
        emit_event("step_started", run_id=run_id, step=step, max_steps=3)

        for agent_name, msg, response in messages:
            await asyncio.sleep(0.2)
            emit_event(
                "agent_message_sent",
                run_id=run_id,
                step=step,
                agent_name=agent_name,
                message=msg
            )

            await asyncio.sleep(0.2)
            emit_event(
                "agent_response_received",
                run_id=run_id,
                step=step,
                agent_name=agent_name,
                response=response
            )

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

    # Emit completion event - this updates status to 'completed'
    emit_event(
        "simulation_completed",
        run_id=run_id,
        step=3,
        total_steps=3
    )

    return {"status": "ok", "run_id": run_id, "message": "Demo simulation completed"}
