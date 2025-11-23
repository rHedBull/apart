# Gemini JSON Mode Implementation

## Summary

The GeminiProvider has been updated to use **structured JSON output mode** (`response_mime_type: "application/json"`) to eliminate JSON parsing errors caused by markdown formatting and comments in LLM responses.

## The Problem

Before this change, Gemini would return responses like:

```
```json
{
  "state_updates": { ... }
}
```
```

This caused `JSONDecodeError` in the SimulatorAgent because the response included markdown code blocks.

## The Solution

### Approach: JSON Mode Without Strict Schema

We use `response_mime_type: "application/json"` **without** schema validation because:

1. **Gemini's schema requirements are strict** - OBJECT types must have non-empty `properties`
2. **Our response structure is dynamic** - Agent names and variable names vary by scenario
3. **Schema-less JSON mode works perfectly** - Gemini returns clean JSON based on the prompt

### Code Changes

#### 1. Updated `src/llm/providers.py`

```python
if force_json:
    # Use JSON mode without strict schema validation
    generation_config = {"response_mime_type": "application/json"}
    response = self._model.generate_content(
        full_prompt,
        generation_config=generation_config
    )
```

Key features:
- ✅ Supports both Google Generative AI API and Vertex AI
- ✅ Uses `models/` prefix for proper API routing
- ✅ JSON mode for simulation engine (`force_json=True`)
- ✅ Custom schemas supported via `json_schema` parameter

#### 2. Updated `src/core/simulator_agent.py`

```python
from llm.providers import GeminiProvider

if isinstance(self.llm_provider, GeminiProvider):
    response_text = self.llm_provider.generate_response(
        prompt=prompt,
        system_prompt=self.system_prompt,
        force_json=True  # Enable JSON mode
    )
else:
    response_text = self.llm_provider.generate_response(
        prompt=prompt,
        system_prompt=self.system_prompt
    )
```

The SimulatorAgent automatically detects Gemini and enables JSON mode.

## Usage

### Direct API (google.generativeai)

```bash
# Set API key
export GEMINI_API_KEY="your-api-key"

# Run simulation
uv run src/main.py scenarios/gemini_example.yaml
```

### Vertex AI (vertexai)

```python
from llm.providers import GeminiProvider

provider = GeminiProvider(
    model_name="gemini-1.5-flash",
    use_vertex=True,
    project="your-gcp-project"  # or set GCP_PROJECT env var
)
```

## Testing

```bash
# Test JSON mode
export GEMINI_API_KEY="your-key"
uv run python tools/test_gemini_json.py
```

Expected output:
```
✅ Valid JSON!
✅ All required keys present: ['state_updates', 'events', 'agent_messages', 'reasoning']
```

## Benefits

| Before | After |
|--------|-------|
| ❌ Markdown code blocks | ✅ Clean JSON |
| ❌ JavaScript comments | ✅ No comments |
| ❌ Requires cleanup | ✅ Direct parsing |
| ❌ Multiple retries | ✅ Fewer retries |

## Comparison with Ollama

| Feature | Gemini | Ollama |
|---------|--------|--------|
| JSON Mode | ✅ Native | ⚠️ Prompt-based |
| Cleanup Needed | ❌ No | ✅ Yes (markdown, comments) |
| Speed | ⚡ Fast (cloud) | 🐌 Slower (local) |
| Cost | 💰 Paid API | 🆓 Free |
| Reliability | ⚠️ Quota limits | ✅ Always available |

**Recommendation**: Use Ollama (mistral:7b) for development, Gemini for production.

## Troubleshooting

### Error: "models/gemini-1.5-flash is not found for API version v1beta"

**Cause**: Using JSON mode requires v1beta API, model must exist
**Fix**: Ensure your API key has access to the model, or try a different model

### Error: "GEMINI_API_KEY not set"

**Fix**:
```bash
export GEMINI_API_KEY="your-key-here"
```

### Error: "response_schema.properties should be non-empty for OBJECT type"

**This is fixed** - We no longer use schema validation, only JSON mode.

## Architecture

```
User Prompt
    ↓
SimulatorAgent
    ↓
    ├─ Detects GeminiProvider → force_json=True
    ├─ Detects OllamaProvider → no force_json
    └─ Other providers → no force_json
    ↓
GeminiProvider.generate_response()
    ↓
    ├─ force_json=True → generation_config={"response_mime_type": "application/json"}
    ├─ json_schema provided → Use custom schema
    └─ Neither → Standard text generation
    ↓
Gemini API (v1beta with JSON mode)
    ↓
Clean JSON Response
    ↓
EngineValidator (minimal cleanup)
    ↓
Success ✅
```

## Future Improvements

1. **Add schema validation** - Once we have static agent names, use full schemas
2. **Retry logic** - Add exponential backoff for API quota errors
3. **Model selection** - Auto-detect best available model
4. **Caching** - Cache model responses for identical prompts

## References

- [Google Generative AI - JSON Mode](https://ai.google.dev/gemini-api/docs/json-mode)
- [Vertex AI - Structured Output](https://cloud.google.com/vertex-ai/docs/generative-ai/multimodal/generate-content)
- Implementation: `src/llm/providers.py:8-162`
- Integration: `src/core/simulator_agent.py:148-160`
