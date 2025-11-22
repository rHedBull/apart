# LLM Integration Guide

## Overview

The agent framework now supports LLM-powered agents alongside traditional template-based agents. This allows agents to generate dynamic, context-aware responses using various LLM providers.

## Architecture

### Module Structure

```
src/
├── core/           # Core simulation logic
│   ├── agent.py          # Agent class (supports both LLM and template-based)
│   ├── orchestrator.py   # Orchestrator with LLM integration
│   ├── game_engine.py    # Game state management
│   └── state.py          # State data models
├── llm/            # LLM provider implementations
│   ├── llm_provider.py   # Abstract base class for all providers
│   └── providers.py      # Concrete provider implementations (Gemini, etc.)
├── utils/          # Utilities
│   ├── config_parser.py  # YAML config parsing
│   ├── persistence.py    # State/log persistence
│   ├── logging_config.py # Structured logging
│   └── variables.py      # Variable definitions
└── main.py         # Entry point
```

### Provider Architecture

The system uses an extensible provider pattern:

- **Abstract Base**: `LLMProvider` defines the interface all providers must implement
- **Concrete Providers**: Each LLM service has its own implementation
- **Agent Integration**: Agents can use any provider or fall back to templates

## Setup

### 1. Install Dependencies

```bash
uv pip install python-dotenv google-generativeai
```

### 2. Configure API Keys

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` and add your API key:

```
GEMINI_API_KEY=your_api_key_here
```

Get a free Gemini API key at: https://makersuite.google.com/app/apikey

## Usage

### Configure LLM Agent in YAML

```yaml
agents:
  - name: "AI Strategist"
    llm:
      provider: "gemini"              # Provider type
      model: "gemini-1.5-flash"       # Model name (free tier)
    system_prompt: |                   # Agent personality/role
      You are a strategic advisor in an economic simulation.
      Keep responses concise and action-oriented.
    variables:                          # Per-agent variables (optional)
      economic_strength: 1500.0
      risk_tolerance: 0.6
```

**Important:** LLM agents require a valid API key. If the provider is unavailable (missing/invalid API key, network issues), the simulation will fail with a clear error message explaining the issue and how to fix it.

**Optional:** Add `response_template` to provide a fallback if you want the simulation to continue even when the LLM is unavailable:
```yaml
agents:
  - name: "AI Strategist"
    llm:
      provider: "gemini"
      model: "gemini-1.5-flash"
    system_prompt: "You are a strategic advisor."
    response_template: "Analyzing situation"  # Optional fallback
```

### Mix LLM and Template Agents

You can have both types in the same simulation:

```yaml
agents:
  # LLM-powered agent
  - name: "AI Agent"
    llm:
      provider: "gemini"
      model: "gemini-1.5-flash"
    system_prompt: "You are a helpful assistant."

  # Traditional template-based agent
  - name: "Template Agent"
    response_template: "Acknowledged, processing"
```

### Run Simulation

```bash
python src/main.py scenarios/llm_example.yaml
```

## Available Providers

### Google Gemini (Default)

- **Free Tier**: `gemini-1.5-flash` (recommended)
- **Env Variable**: `GEMINI_API_KEY`
- **Configuration**:
  ```yaml
  llm:
    provider: "gemini"
    model: "gemini-1.5-flash"
  ```

### Future Providers

The architecture supports easy addition of:
- **Ollama** (local models)
- **OpenAI** (GPT-4, etc.)
- **Anthropic Claude**
- Any other LLM service

## Agent Behavior

### LLM-Powered Agent

1. Receives message from orchestrator
2. Sends message + system prompt to LLM provider
3. Returns LLM-generated response
4. Falls back to template (if available) on LLM failure

### Fallback Strategy

If an LLM agent has both `llm_provider` and `response_template`:
- Uses LLM by default
- Falls back to template if LLM fails
- Logs error in response

### Template-Only Agent

Works exactly as before:
- Returns fixed template response
- Includes step counter
- No external API calls

## Testing

### ⚠️ IMPORTANT: No API Calls in Automated Tests

**All automated tests use mock providers and make ZERO real API calls.**
This ensures:
- No costs incurred during testing
- Fast test execution
- No API key required for tests
- Reliable, deterministic test results

### Run All Tests (Safe - No API Calls)

```bash
pytest tests/ -v
```

This runs 141+ tests including 17 LLM-specific tests using `MockLLMProvider`.
**Guaranteed to make no real API calls.**

### Test LLM Integration with Mocks

```bash
pytest tests/unit/test_llm_agent.py -v
```

Tests LLM agent functionality with mock providers:
- Agent initialization with LLM providers
- Response generation (mocked)
- Fallback behavior
- Error handling
- **Zero API calls made**

### Manual Testing with Real API (Optional)

For manual verification of actual LLM connectivity:

```bash
python test_llm.py
```

This script:
1. Runs safe tests first (mocks only)
2. **Asks for confirmation** before making real API calls
3. Can be cancelled at any time (Ctrl+C)
4. Safe to run without API key (skips real API tests)

### Mock Provider

The `MockLLMProvider` class simulates LLM behavior:
- Returns predefined responses
- Cycles through response list
- Tracks call count and prompts
- No network calls
- Instant responses

Example usage in tests:
```python
from llm.mock_provider import MockLLMProvider

provider = MockLLMProvider(responses=["Response 1", "Response 2"])
agent = Agent(name="Test", llm_provider=provider)
response = agent.respond("Test message")
# Returns "Response 1" without any API call
```

## Adding New Providers

### 1. Implement Provider Class

Create in `src/llm/providers.py`:

```python
class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama2"):
        self.base_url = base_url
        self.model = model

    def is_available(self) -> bool:
        # Check if Ollama is running
        pass

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        # Call Ollama API
        pass
```

### 2. Register in Orchestrator

Update `src/core/orchestrator.py`:

```python
if provider_type == "ollama":
    llm_provider = OllamaProvider(
        base_url=llm_config.get("base_url", "http://localhost:11434"),
        model=llm_config.get("model", "llama2")
    )
```

### 3. Update Environment Template

Add to `.env.example`:

```
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2
```

## Performance Considerations

- **LLM Latency**: API calls add 1-5s per response
- **Cost**: Free tiers have rate limits (Gemini: 15 req/min)
- **Fallback**: Template agents provide instant responses
- **Async**: Future enhancement for parallel agent processing

## Troubleshooting

### "GEMINI_API_KEY not configured"

**Solution**: Check `.env` file exists and contains valid API key

### "No module named 'core'"

**Solution**: Tests need `tests/conftest.py` for path setup (already included)

### Import Errors

**Solution**: All imports now use module paths:
- `from core.agent import Agent`
- `from llm.providers import GeminiProvider`
- `from utils.config_parser import ...`

## Examples

See `scenarios/llm_example.yaml` for a complete example with:
- Mixed LLM and template agents
- System prompts
- Variable configurations
- Economic simulation context
