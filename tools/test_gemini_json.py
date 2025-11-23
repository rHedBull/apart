#!/usr/bin/env python3
"""Test Gemini provider with structured JSON output."""

import os
import sys
import json

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm.providers import GeminiProvider


def test_gemini_json_output():
    """Test that Gemini returns valid JSON with the simulation schema."""

    print("=" * 60)
    print("Gemini Structured JSON Output Test")
    print("=" * 60)

    # Check if API key is set
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not set. Set it with:")
        print("   export GEMINI_API_KEY='your-key-here'")
        return False

    print(f"✓ API key found: {api_key[:8]}...{api_key[-4:]}")

    # Initialize provider
    try:
        provider = GeminiProvider(model_name="gemini-1.5-flash")
        print("✓ Provider initialized")
    except Exception as e:
        print(f"❌ Failed to initialize provider: {e}")
        return False

    if not provider.is_available():
        print("❌ Provider not available")
        return False

    print("✓ Provider available")

    # Test prompt (similar to what SimulatorAgent would send)
    system_prompt = """You are the simulation engine managing an economic strategy game.
Your role is to simulate realistic economic outcomes based on agent decisions."""

    prompt = """This is Step 1 of the simulation.

Current State:
- Global vars: {"interest_rate": 0.04, "market_volatility": 0.15, "round_number": 1}
- Agent vars: {"Player1": {"capital": 1000.0, "risk_tolerance": 0.5}}

Agent Responses:
- Player1: "I will invest 500 in high-growth tech stocks."

Generate the next simulation step with state updates and agent messages.
"""

    # Test 1: Without structured JSON (old behavior)
    print("\n" + "=" * 60)
    print("Test 1: WITHOUT structured JSON (old behavior)")
    print("=" * 60)

    try:
        response = provider.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            force_json=False
        )
        print(f"Response length: {len(response)} chars")
        print(f"First 300 chars:\n{response[:300]}")

        # Try to parse as JSON
        try:
            json.loads(response)
            print("✅ Valid JSON (unexpected - may have markdown blocks)")
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON (expected): {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

    # Test 2: With structured JSON (new behavior)
    print("\n" + "=" * 60)
    print("Test 2: WITH structured JSON (new behavior)")
    print("=" * 60)

    try:
        response = provider.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            force_json=True
        )
        print(f"Response length: {len(response)} chars")
        print(f"First 300 chars:\n{response[:300]}")

        # Try to parse as JSON
        try:
            data = json.loads(response)
            print("✅ Valid JSON!")

            # Check required keys
            required_keys = ["state_updates", "events", "agent_messages", "reasoning"]
            missing = [k for k in required_keys if k not in data]

            if missing:
                print(f"❌ Missing required keys: {missing}")
                return False

            print(f"✅ All required keys present: {required_keys}")

            # Check structure
            if "global_vars" in data["state_updates"]:
                print(f"✅ state_updates.global_vars present")
            else:
                print(f"❌ state_updates.global_vars missing")
                return False

            if "agent_vars" in data["state_updates"]:
                print(f"✅ state_updates.agent_vars present")
            else:
                print(f"❌ state_updates.agent_vars missing")
                return False

            print("\n✅ SUCCESS: Gemini returns valid structured JSON!")
            print(f"\nFull response:\n{json.dumps(data, indent=2)}")
            return True

        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON: {e}")
            print(f"\nRaw response:\n{response}")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_gemini_json_output()
    sys.exit(0 if success else 1)
