# Ollama Model Testing Results

**Test Date**: 2025-11-23
**Test Script**: `tools/test_ollama_models.py`

## Summary

Tested 8 Ollama models for their ability to generate valid, structured JSON for the simulation engine. The test evaluates:
- JSON validity and structure
- Presence of markdown code blocks
- Invalid JavaScript comments (`//`)
- Use of expressions instead of actual values

## Test Results

### ✅ Excellent Models (Recommended)

These models produce clean, valid JSON without issues:

| Model | Size | JSON Quality | Notes |
|-------|------|--------------|-------|
| **mistral:7b** | 4.1 GB | Perfect | No markdown, no comments, clean output |
| **llama3.1:8b** | 4.9 GB | Perfect | No markdown, no comments, clean output |
| **codellama:latest** | 3.8 GB | Perfect | Code-focused, excellent structured output |
| **deepseek-coder-v2:latest** | 8.9 GB | Perfect | Larger code model, very reliable |

**Recommendation**: Use **mistral:7b** or **llama3.1:8b** for best balance of size and quality.

### ⚠️ Basic Models (Limited Use)

| Model | Size | JSON Quality | Issues |
|-------|------|--------------|--------|
| **gemma3:1b** | 815 MB | Basic | Wraps JSON in markdown blocks; works for simple prompts but may miss required fields in complex scenarios |

**Recommendation**: Use only for simple, well-defined tasks. Not recommended for simulation engine.

### ❌ Problematic Models (Avoid)

| Model | Size | Issue |
|-------|------|-------|
| **deepseek-r1:1.5b** | 1.1 GB | Generates `<think>` reasoning text instead of JSON |
| **deepseek-r1:7b** | 4.7 GB | Timeout / reasoning-focused, not JSON output |
| **deepseek-r1:14b** | 9.0 GB | Timeout / reasoning-focused, not JSON output |

**Note**: DeepSeek-R1 models are designed for chain-of-thought reasoning, not structured output.

## Code Improvements Made

Based on testing, the following improvements were added to handle Ollama quirks:

1. **Markdown Block Stripping** (`src/core/engine_validator.py`)
   - Removes ` ```json\n{...}\n``` ` wrappers
   - Handles both ````json` and plain ` ``` ` blocks

2. **Comment Removal**
   - Strips JavaScript-style `//` comments (invalid in JSON)
   - Applied line-by-line before JSON parsing

3. **Trailing Comma Cleanup**
   - Removes `, }` and `, ]` patterns using regex
   - Prevents common JSON formatting errors

4. **Enhanced Prompts**
   - Added explicit type information (int, float, bool)
   - Listed available variables with constraints
   - Emphasized "no expressions, actual values only"

## Running the Test

```bash
uv run python tools/test_ollama_models.py
```

This will test all installed models and provide a summary report.

## Recommendations for Users

### For Simulation Engine (scenarios/*/engine)

**Best Choice**: **Google Gemini** (`gemini-1.5-flash`)
- Most reliable for complex simulation prompts
- Excellent JSON compliance
- Handles multi-step reasoning well

**Experimental**: `mistral:7b` or `llama3.1:8b`
- ✅ Works for simple JSON generation
- ⚠️  May struggle with complex simulation prompts
- May generate expressions instead of values
- Requires additional prompt engineering
- Use only if privacy/local execution is critical

**Alternative**: `codellama:latest` or `deepseek-coder-v2:latest`
- Untested for full simulation engine
- May work better than mistral:7b (code-focused)

### For Agent LLMs (scenarios/*/agents/*/llm)

**Best Choice**: Any of the excellent models above
- `mistral:7b`: Good for strategic, high-risk agents
- `llama3.1:8b`: Good for balanced, advisory agents
- `gemma3:1b`: Acceptable for simple, templated agents

**Avoid**: deepseek-r1:* models (not designed for structured output)

## Future Testing

To test additional models, edit `tools/test_ollama_models.py` and add model names to the `models` list.

Example:
```python
models = [
    "your-model:tag",
    # ... other models
]
```
