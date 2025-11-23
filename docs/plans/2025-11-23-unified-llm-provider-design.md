# Unified LLM Provider Design

**Date:** 2025-11-23
**Status:** Approved for implementation

## Overview

Consolidate all external LLM API providers (OpenAI, Grok/xAI, Anthropic/Claude, Gemini, Ollama) into a single unified `UnifiedLLMProvider` class. This replaces the existing separate `GeminiProvider` and `OllamaProvider` classes.

## Architecture

### Single Provider Class Design

One `UnifiedLLMProvider` class handles all LLM providers through conditional logic:
- **OpenAI-compatible APIs:** OpenAI (GPT), Grok, Ollama (same chat completions format)
- **Provider-specific APIs:** Anthropic (messages API), Gemini (generateContent)

### Provider Configuration

YAML configuration uses provider-specific naming:

```yaml
agents:
  - name: "Agent 1"
    llm:
      provider: "openai"  # or "grok", "anthropic", "gemini", "ollama"
      model: "gpt-4o-mini"
```

### Environment Variables

- `OPENAI_API_KEY` - for GPT models
- `XAI_API_KEY` - for Grok models
- `ANTHROPIC_API_KEY` - for Claude models
- `GEMINI_API_KEY` - for Gemini models (existing)
- `OLLAMA_BASE_URL` - for Ollama (optional, defaults to localhost:11434)

## Provider-Specific Details

### Provider Mapping

| Provider | Base URL | Auth Method | Request Format |
|----------|----------|-------------|----------------|
| `openai` | `https://api.openai.com/v1` | Bearer token (OPENAI_API_KEY) | OpenAI chat completions |
| `grok` | `https://api.x.ai/v1` | Bearer token (XAI_API_KEY) | OpenAI-compatible |
| `anthropic` | `https://api.anthropic.com/v1` | x-api-key header (ANTHROPIC_API_KEY) | Anthropic messages API |
| `gemini` | Google SDK | API key config (GEMINI_API_KEY) | generateContent API |
| `ollama` | `http://localhost:11434` | None | OpenAI-compatible |

### Request Formats

**OpenAI-compatible (openai, grok, ollama):**
```python
POST /v1/chat/completions
{
  "model": "...",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ]
}
```

**Anthropic:**
```python
POST /v1/messages
{
  "model": "...",
  "max_tokens": 1024,
  "messages": [{"role": "user", "content": "..."}],
  "system": "..."  # system prompt separate
}
```

**Gemini:**
```python
# Uses google.generativeai SDK
model.generate_content(
  full_prompt,
  generation_config={"response_mime_type": "application/json"}
)
```

## Implementation Structure

### Class Design

```python
class UnifiedLLMProvider(LLMProvider):
    def __init__(self, provider: str, model: str, api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        """
        Args:
            provider: "openai", "grok", "anthropic", "gemini", or "ollama"
            model: Model identifier
            api_key: Optional API key (falls back to env vars)
            base_url: Optional base URL override
        """

    def _get_endpoint(self) -> str:
        """Return correct API endpoint based on provider"""

    def _format_request(self, prompt: str, system_prompt: Optional[str]) -> Dict:
        """Format request for specific provider"""

    def _call_api(self, request_data: Dict) -> str:
        """Make HTTP request or SDK call"""

    def is_available(self) -> bool:
        """Check if provider is configured (only checks active provider)"""

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None,
                         json_schema: Optional[Dict] = None,
                         force_json: bool = False) -> str:
        """Public interface - unchanged from existing providers"""
```

## Integration Changes

### Files to Modify

1. **`src/llm/providers.py`:**
   - Add `UnifiedLLMProvider` class
   - Keep existing `GeminiProvider` and `OllamaProvider` for backwards compatibility

2. **`src/core/orchestrator.py`:**
   - Update provider instantiation to support new provider types
   - Map YAML `provider` field to `UnifiedLLMProvider` initialization

3. **`pyproject.toml`:**
   - Add `openai>=1.0.0`
   - Add `anthropic>=0.8.0`
   - Keep existing `google-generativeai>=0.3.0`
   - Keep existing `requests>=2.31.0`

4. **New example scenarios:**
   - `scenarios/openai_example.yaml`
   - `scenarios/grok_example.yaml`
   - `scenarios/claude_example.yaml`

5. **Documentation:**
   - Update `docs/LLM_INTEGRATION.md`

## Error Handling

### Provider Availability
- `is_available()` only checks the provider specified in the scenario
- Returns helpful error messages for missing API keys or unreachable endpoints
- No unnecessary checks for unused providers

### Error Types
1. **Missing API keys:** Clear message indicating which env var to set
2. **Network errors:** Connection timeouts, unreachable endpoints
3. **Provider-specific errors:**
   - Rate limiting (OpenAI, Anthropic, Gemini)
   - Model not found (Ollama)
   - Invalid API keys
   - Region restrictions

## JSON Schema Support

- **Gemini:** Maintain existing `response_mime_type: "application/json"` with optional schema
- **OpenAI/Grok:** Use native `response_format: {"type": "json_object"}`
- **Anthropic:** Prompt engineering (no native JSON mode)
- **Ollama:** Prompt engineering or JSON mode if model supports

## Testing Strategy

### Unit Tests
- Mock API responses for each provider
- Test request formatting
- Test error handling
- Test provider detection logic

### Integration Tests
- Test actual API calls when credentials available
- Skip tests gracefully when API keys missing
- Test JSON mode for each provider

## Migration Path

**Phase 1:** Add `UnifiedLLMProvider` alongside existing providers
**Phase 2:** Update orchestrator to support new provider types
**Phase 3:** Create example scenarios and update documentation
**Phase 4:** (Optional) Deprecate old provider classes

## Backwards Compatibility

✅ Existing YAML files continue to work
✅ Old provider names still supported
✅ No breaking changes to existing code
