# Gemini Structured JSON Output

## Overview

The GeminiProvider has been updated to support **structured JSON output** using Google's `response_mime_type` and `response_schema` features. This ensures the simulation engine receives valid, parseable JSON without markdown formatting, comments, or other artifacts.

## Problem

Previously, Gemini would return responses like:

```
```json
{
  "state_updates": { ... }
}
```
```

Or with comments:

```json
{
  "state_updates": { ... }, // Game state updates
}
```

This caused JSON parsing errors in the SimulatorAgent, leading to validation failures.

## Solution

The provider now uses Gemini's native structured output feature:

```python
generation_config = {
    "response_mime_type": "application/json",
    "response_schema": schema_dict
}
```

This guarantees valid JSON that matches the expected schema.

## Usage

### Direct API (google.generativeai)

```python
from llm.providers import GeminiProvider

provider = GeminiProvider(
    api_key="your-key",  # or set GEMINI_API_KEY env var
    model_name="gemini-1.5-flash"
)

# Automatic structured JSON for simulation engine
response = provider.generate_response(
    prompt="Your prompt here",
    system_prompt="System instructions",
    force_json=True  # Uses built-in simulation schema
)
```

### Vertex AI (vertexai)

```python
from llm.providers import GeminiProvider

provider = GeminiProvider(
    model_name="gemini-1.5-flash",
    use_vertex=True,
    project="your-gcp-project",  # or set GCP_PROJECT env var
    location="us-central1"
)

response = provider.generate_response(
    prompt="Your prompt here",
    force_json=True
)
```

### Custom JSON Schema

```python
custom_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    },
    "required": ["name", "age"]
}

response = provider.generate_response(
    prompt="Extract user data: John is 30 years old",
    json_schema=custom_schema
)
```

## SimulatorAgent Integration

The SimulatorAgent automatically detects when using GeminiProvider and enables structured JSON:

```python
# In simulator_agent.py (line 148-160)
from llm.providers import GeminiProvider

if isinstance(self.llm_provider, GeminiProvider):
    response_text = self.llm_provider.generate_response(
        prompt=prompt,
        system_prompt=self.system_prompt,
        force_json=True  # Automatic structured JSON
    )
else:
    response_text = self.llm_provider.generate_response(
        prompt=prompt,
        system_prompt=self.system_prompt
    )
```

## JSON Mode (No Schema Validation)

Due to Gemini's strict schema requirements (OBJECT types must have non-empty `properties`), the provider uses `response_mime_type: "application/json"` **without** schema validation for the simulation engine.

This ensures:
- Valid JSON output (no markdown blocks)
- Flexibility for dynamic properties
- No complex schema definitions needed

```python
# When force_json=True
generation_config = {"response_mime_type": "application/json"}
```

The prompt itself guides the LLM to produce the correct structure.

## Testing

Test the Gemini provider with structured JSON:

```bash
# Set your API key
export GEMINI_API_KEY="your-key-here"

# Run test script
uv run python tools/test_gemini_json.py
```

Expected output:

```
✅ Valid JSON!
✅ All required keys present: ['state_updates', 'events', 'agent_messages', 'reasoning']
✅ state_updates.global_vars present
✅ state_updates.agent_vars present
✅ SUCCESS: Gemini returns valid structured JSON!
```

## Environment Variables

### Direct API
```bash
export GEMINI_API_KEY="your-api-key"
```

### Vertex AI
```bash
export GCP_PROJECT="your-gcp-project"
```

## Benefits

1. **No JSON parsing errors** - Guaranteed valid JSON
2. **No markdown cleanup** - No more stripping ```json blocks
3. **Schema validation** - Response matches expected structure
4. **Type safety** - Fields have correct types (string, int, etc.)
5. **Works with both APIs** - Direct API and Vertex AI

## Comparison

### Before (Text Generation)

```python
# Old implementation
response = model.generate_content(prompt)
# Returns: "```json\n{...}\n```" or "{...} // comment"
```

**Issues:**
- Markdown code blocks
- JavaScript-style comments
- Trailing commas
- Requires cleanup in EngineValidator

### After (Structured JSON)

```python
# New implementation
response = model.generate_content(
    prompt,
    generation_config={
        "response_mime_type": "application/json",
        "response_schema": schema
    }
)
# Returns: Clean JSON string matching schema
```

**Benefits:**
- Clean JSON, no cleanup needed
- Guaranteed schema compliance
- Faster validation
- Fewer retries

## Migration

No changes needed in scenario YAML files. The provider automatically uses structured JSON when `force_json=True` is passed.

Existing scenarios will continue to work:

```yaml
engine:
  provider: "gemini"
  model: "gemini-1.5-flash"
```

## References

- [Google Generative AI - Structured Output](https://ai.google.dev/gemini-api/docs/json-mode)
- [Vertex AI - Structured Output](https://cloud.google.com/vertex-ai/docs/generative-ai/multimodal/generate-content#generate_content_with_structured_output)
