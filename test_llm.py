#!/usr/bin/env python3
"""
MANUAL TEST SCRIPT for LLM integration.

WARNING: This script makes REAL API calls to LLM providers and may incur costs!
Only run this manually when you want to test actual LLM connectivity.

For automated testing without API calls, use:
    pytest tests/unit/test_llm_agent.py

This uses mock providers that don't make real API calls.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
from llm.providers import GeminiProvider
from llm.mock_provider import MockLLMProvider
from core.agent import Agent


def test_gemini_provider():
    """Test Gemini provider directly."""
    print("Testing Gemini Provider...")
    load_dotenv()

    provider = GeminiProvider()

    if not provider.is_available():
        print("❌ Gemini provider not available. Check GEMINI_API_KEY in .env file.")
        return False

    print("✅ Gemini provider is configured")

    try:
        response = provider.generate_response(
            "Say hello in exactly 5 words.",
            system_prompt="You are a helpful assistant."
        )
        print(f"Response: {response}")
        print("✅ Gemini provider working correctly")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_llm_agent():
    """Test Agent with LLM."""
    print("\nTesting LLM Agent...")
    load_dotenv()

    provider = GeminiProvider()
    agent = Agent(
        name="Test Agent",
        llm_provider=provider,
        system_prompt="You are a strategic advisor. Keep responses brief."
    )

    try:
        response = agent.respond("What should we do next?")
        print(f"Agent response: {response}")
        print("✅ LLM Agent working correctly")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_template_agent():
    """Test traditional template-based agent."""
    print("\nTesting Template Agent...")

    agent = Agent(
        name="Template Agent",
        response_template="Processing request"
    )

    response = agent.respond("Test message")
    print(f"Agent response: {response}")
    print("✅ Template Agent working correctly")
    return True


def test_mock_provider():
    """Test mock provider (no API calls)."""
    print("\nTesting Mock LLM Provider (no API calls)...")

    provider = MockLLMProvider(responses=["Mock response 1", "Mock response 2"])
    agent = Agent(
        name="Mock Agent",
        llm_provider=provider,
        system_prompt="You are a test agent"
    )

    response1 = agent.respond("Test message 1")
    response2 = agent.respond("Test message 2")

    print(f"Response 1: {response1}")
    print(f"Response 2: {response2}")
    print(f"Provider was called {provider.call_count} times")
    print("✅ Mock Provider working correctly (no API calls made)")
    return True


if __name__ == "__main__":
    print("=" * 70)
    print("LLM Integration Test")
    print("=" * 70)
    print("\n⚠️  WARNING: Some tests make REAL API calls and may incur costs!")
    print("Safe tests (no API calls) are marked with [SAFE]")
    print("=" * 70)

    results = []

    # Safe tests first (no API calls)
    print("\n--- SAFE TESTS (No API Calls) ---")
    results.append(("[SAFE] Mock Provider", test_mock_provider()))
    results.append(("[SAFE] Template Agent", test_template_agent()))

    # Real API tests
    print("\n--- REAL API TESTS (May incur costs) ---")

    # Ask for confirmation
    try:
        response = input("\nRun tests that make REAL API calls? (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            results.append(("Gemini Provider [REAL API]", test_gemini_provider()))
            results.append(("LLM Agent [REAL API]", test_llm_agent()))
        else:
            print("Skipping real API tests.")
            results.append(("Gemini Provider [REAL API]", None))
            results.append(("LLM Agent [REAL API]", None))
    except (KeyboardInterrupt, EOFError):
        print("\nSkipping real API tests.")
        results.append(("Gemini Provider [REAL API]", None))
        results.append(("LLM Agent [REAL API]", None))

    print("\n" + "=" * 70)
    print("Test Results:")
    print("=" * 70)
    for test_name, result in results:
        if result is None:
            status = "⏭️  SKIPPED"
        elif result:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"
        print(f"{test_name}: {status}")

    # Only check non-skipped tests
    non_skipped_results = [r for _, r in results if r is not None]
    all_passed = all(non_skipped_results) if non_skipped_results else True

    print("\n💡 TIP: Use 'pytest tests/unit/test_llm_agent.py' for automated testing")
    print("   (uses mocks, no API calls)")

    sys.exit(0 if all_passed else 1)
