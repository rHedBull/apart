# Testing Guide

## ✅ Cost-Safe Testing

**All automated tests are 100% safe and make ZERO real API calls.**

This project uses mock providers to ensure:
- ✅ No API costs during testing
- ✅ No API keys required to run tests
- ✅ Fast, deterministic test execution
- ✅ Works offline

## Running Tests

### Full Test Suite (Safe)

```bash
pytest tests/ -v
```

**Result:** 141+ tests, all using mocks, zero API calls.

### LLM-Specific Tests (Safe)

```bash
pytest tests/unit/test_llm_agent.py -v
```

**Result:** 17 LLM tests using `MockLLMProvider`, zero API calls.

### Quick Test

```bash
pytest tests/ -q
```

**Result:** Summary output, zero API calls.

## Manual Testing (Optional)

For verifying real LLM connectivity (requires API key):

```bash
python test_llm.py
```

This script:
1. ✅ Runs safe mock tests first
2. ⚠️ **Asks permission** before making real API calls
3. ✅ Shows clear warnings about costs
4. ✅ Can be cancelled anytime (Ctrl+C)
5. ✅ Skips real API tests if you say "no"

### Example Output

```
--- SAFE TESTS (No API Calls) ---
✅ Mock Provider: PASSED
✅ Template Agent: PASSED

--- REAL API TESTS (May incur costs) ---
Run tests that make REAL API calls? (yes/no): n
⏭️  Gemini Provider: SKIPPED
⏭️  LLM Agent: SKIPPED
```

## Mock Provider

The `MockLLMProvider` is used in all automated tests:

```python
from llm.mock_provider import MockLLMProvider

# Create mock with predefined responses
provider = MockLLMProvider(
    responses=["Response 1", "Response 2", "Response 3"]
)

# Use like a real provider
response = provider.generate_response("Test prompt")
# Returns: "Response 1" (no API call)

# Cycles through responses
response2 = provider.generate_response("Another prompt")
# Returns: "Response 2" (no API call)

# Check call count
print(provider.call_count)  # 2

# Verify prompts were passed correctly
print(provider.last_prompt)  # "Another prompt"
```

### Mock Provider Features

- **No Network Calls:** Completely offline
- **Predefined Responses:** Configure what it returns
- **Call Tracking:** Verify number of calls
- **Prompt Recording:** Check what was sent
- **Cycle Behavior:** Cycles through response list
- **Availability Control:** Can simulate unavailable provider
- **Reset Method:** Clean state between tests

### Example Test

```python
def test_agent_with_llm():
    """Test LLM agent without real API calls."""
    # Create mock provider
    provider = MockLLMProvider(
        responses=["Strategic move A", "Strategic move B"]
    )

    # Create agent with mock
    agent = Agent(
        name="AI Agent",
        llm_provider=provider,
        system_prompt="You are a strategist"
    )

    # Get response (no API call made!)
    response = agent.respond("What should we do?")

    # Verify behavior
    assert response == "Strategic move A"
    assert provider.call_count == 1
    assert provider.last_prompt == "What should we do?"
    assert provider.last_system_prompt == "You are a strategist"
```

## Safety Guarantees

### What Tests DO

✅ Test agent logic with mocks
✅ Verify LLM integration architecture
✅ Check fallback mechanisms
✅ Validate error handling
✅ Ensure response processing works

### What Tests DON'T Do

❌ Make real API calls
❌ Require API keys
❌ Incur costs
❌ Need internet connection
❌ Access external services

## Verification

You can verify no API calls are made:

```bash
# Search for real API usage in tests
grep -r "GeminiProvider\|genai\|google.generativeai" tests/ --include="*.py"

# Should output: "No matches found" or only mock references
```

## CI/CD Integration

Safe to run in CI/CD pipelines:
- No secrets needed (for tests)
- No external dependencies
- Fast execution (<1 second for all tests)
- Deterministic results
- No rate limiting concerns

## Writing New Tests

Always use `MockLLMProvider` in tests:

```python
# ✅ GOOD - Uses mock
from llm.mock_provider import MockLLMProvider

def test_feature():
    provider = MockLLMProvider(responses=["Test response"])
    agent = Agent(name="Test", llm_provider=provider)
    # Test logic here...

# ❌ BAD - Would make real API calls
from llm.providers import GeminiProvider

def test_feature():
    provider = GeminiProvider()  # DON'T DO THIS IN TESTS!
    # This would make real API calls
```

## Summary

🎯 **Key Takeaway:** Run `pytest tests/` with confidence. Zero API calls, zero costs, always.
