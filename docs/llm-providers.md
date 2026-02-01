# LLM Providers

Apart uses a unified LLM provider system supporting multiple AI services.

## Supported Providers

| Provider | Models | API Key Env | Features |
|----------|--------|-------------|----------|
| **Gemini** | gemini-1.5-flash, gemini-2.5-flash | `GEMINI_API_KEY` | JSON mode, schema validation |
| **OpenAI** | gpt-4o, gpt-4-turbo | `OPENAI_API_KEY` | JSON mode |
| **Anthropic** | claude-3-5-sonnet, claude-3-opus | `ANTHROPIC_API_KEY` | Large context |
| **Grok** | grok-beta | `XAI_API_KEY` | OpenAI-compatible |
| **Ollama** | Any local model | None (local) | Privacy, no cost |

## Configuration

### In Scenario YAML

```yaml
# Engine configuration
engine:
  provider: "gemini"
  model: "gemini-2.5-flash"
  system_prompt: |
    You are the simulation engine...

# Agent-specific LLM
agents:
  - name: "Strategic Agent"
    llm:
      provider: "anthropic"
      model: "claude-3-5-sonnet-20241022"
    system_prompt: |
      You are a strategic decision maker...
```

### Environment Variables

```bash
# Cloud providers
export GEMINI_API_KEY="your-gemini-key"
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export XAI_API_KEY="your-grok-key"

# Ollama (local)
export OLLAMA_BASE_URL="http://localhost:11434"
```

## Provider Details

### Google Gemini (Recommended)

Best for simulation engines due to native JSON schema support.

```yaml
engine:
  provider: "gemini"
  model: "gemini-2.5-flash"
```

**Features:**
- Native JSON mode with schema validation
- Fast inference
- Good instruction following
- Cost-effective

**Recommended Models:**
- `gemini-2.5-flash` - Best balance of speed/quality
- `gemini-1.5-flash` - Faster, slightly lower quality
- `gemini-1.5-pro` - Higher quality, slower

### Ollama (Local)

Best for privacy-sensitive simulations and development.

```yaml
engine:
  provider: "ollama"
  model: "phi4-reasoning:plus"
```

**Setup:**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull recommended models
ollama pull phi4-reasoning:plus
ollama pull mistral:7b
ollama pull llama3.1:8b
```

**Recommended Models:**

| Model | Quality | Speed | Notes |
|-------|---------|-------|-------|
| `phi4-reasoning:plus` | Excellent | Slow | Best reasoning, needs good GPU |
| `mistral:7b` | Good | Fast | Good general purpose |
| `llama3.1:8b` | Good | Medium | Strong instruction following |
| `deepseek-r1:8b` | Good | Slow | Reasoning model |
| `gemma3:1b` | Basic | Very fast | Simple tasks only |

**Configuration:**
```yaml
engine:
  provider: "ollama"
  model: "phi4-reasoning:plus"
  base_url: "http://localhost:11434"  # Optional, default shown
```

### OpenAI

```yaml
engine:
  provider: "openai"
  model: "gpt-4o"
```

**Features:**
- JSON mode support
- Excellent instruction following
- Wide model selection

### Anthropic

```yaml
engine:
  provider: "anthropic"
  model: "claude-3-5-sonnet-20241022"
```

**Features:**
- Large context window (200K tokens)
- Strong reasoning
- Note: No native JSON mode (uses prompt engineering)

### Grok (xAI)

Uses OpenAI-compatible API.

```yaml
engine:
  provider: "grok"
  model: "grok-beta"
```

## Programmatic Usage

```python
from llm.providers import UnifiedLLMProvider

# Create provider
provider = UnifiedLLMProvider(
    provider="gemini",
    model="gemini-2.5-flash",
    api_key="optional-override"  # Falls back to env var
)

# Check availability
if provider.is_available():
    response = provider.generate_response(
        prompt="What is your analysis?",
        system_prompt="You are a strategic advisor.",
        force_json=True  # Request JSON output
    )
```

## JSON Output

### Gemini (Native)

```python
response = provider.generate_response(
    prompt="Analyze the situation",
    system_prompt="Return JSON with 'analysis' and 'recommendations' fields",
    force_json=True,
    json_schema={
        "type": "object",
        "properties": {
            "analysis": {"type": "string"},
            "recommendations": {"type": "array", "items": {"type": "string"}}
        }
    }
)
```

### Other Providers

JSON mode is achieved through prompt engineering:

```python
response = provider.generate_response(
    prompt="Analyze the situation",
    system_prompt="You must respond only with valid JSON.",
    force_json=True
)
```

## Error Handling

```python
from llm.providers import UnifiedLLMProvider

provider = UnifiedLLMProvider(provider="gemini", model="gemini-2.5-flash")

if not provider.is_available():
    print(f"Provider not configured. Set GEMINI_API_KEY.")
else:
    try:
        response = provider.generate_response(prompt="Test")
    except Exception as e:
        print(f"API error: {e}")
```

## Mock Provider (Testing)

For tests, use the mock provider to avoid API calls:

```python
from llm.mock_provider import MockLLMProvider

provider = MockLLMProvider(
    responses=["Response 1", "Response 2"],
    available=True
)

response = provider.generate_response("Test prompt")
# Returns: "Response 1" (no API call)

assert provider.call_count == 1
assert provider.last_prompt == "Test prompt"
```

## Adding New Providers

1. Add provider to `UnifiedLLMProvider.SUPPORTED_PROVIDERS`
2. Add API key mapping to `API_KEY_ENV_MAP`
3. Add package to `PACKAGE_MAP`
4. Implement `_generate_{provider}()` method
5. Update `_initialize_provider()` and `is_available()`

Example structure:
```python
def _generate_newprovider(self, prompt: str, system_prompt: Optional[str], force_json: bool) -> str:
    """Generate response using NewProvider API."""
    # Implementation here
    pass
```
