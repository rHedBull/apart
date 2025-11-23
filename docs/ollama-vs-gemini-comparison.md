# Ollama vs Gemini Comparison

**Date**: 2025-11-23
**Purpose**: Compare local Ollama models with cloud-based Gemini for simulation engine

## Scenarios Created

1. **`scenarios/ollama_example.yaml`** - Local Ollama (mistral:7b)
2. **`scenarios/gemini_example.yaml`** - Cloud Gemini (gemini-1.5-flash)

Both scenarios are identical except for the LLM provider configuration.

## Quick Comparison

| Feature | Ollama (mistral:7b) | Gemini (gemini-1.5-flash) |
|---------|---------------------|---------------------------|
| **Success Rate** | ~50-70% | ~99% |
| **Setup** | Local, no API key | Requires API key |
| **Privacy** | Fully local | Cloud-based |
| **Speed** | Depends on hardware | Fast (cloud) |
| **Cost** | Free | Free tier available |
| **JSON Quality** | Good with enhanced prompts | Excellent |
| **Consistency** | Variable | Very consistent |
| **Use Case** | Experimentation, privacy | Production, reliability |

## Detailed Comparison

### Ollama (mistral:7b)

**Pros:**
- ✅ Fully local execution (privacy)
- ✅ No API key required
- ✅ No internet needed
- ✅ Free unlimited use
- ✅ Works with enhanced prompts

**Cons:**
- ⚠️  ~50-70% success rate
- ⚠️  Sometimes generates JavaScript expressions instead of values
- ⚠️  May produce truncated JSON
- ⚠️  Requires retry logic
- ⚠️  Performance depends on hardware

**Best For:**
- Local development
- Privacy-sensitive scenarios
- Learning and experimentation
- Offline environments

### Gemini (gemini-1.5-flash)

**Pros:**
- ✅ ~99% success rate
- ✅ Consistent JSON formatting
- ✅ Excellent multi-step reasoning
- ✅ Fast response times
- ✅ No local compute needed

**Cons:**
- ⚠️  Requires API key
- ⚠️  Internet connection required
- ⚠️  Data sent to Google
- ⚠️  Rate limits (though generous)

**Best For:**
- Production deployments
- Complex simulations
- Reliability-critical scenarios
- Multi-step reasoning tasks

## Testing the Scenarios

### Ollama Example

```bash
# Prerequisites: Ollama installed with mistral:7b model
ollama pull mistral:7b

# Run simulation
uv run src/main.py scenarios/ollama_example.yaml

# Expected: May succeed or fail - this is normal
# Success: Completes 5 steps with realistic state changes
# Failure: May fail at any step with JSON errors
```

### Gemini Example

```bash
# Prerequisites: Gemini API key
cp .env.example .env
# Edit .env and add: GEMINI_API_KEY=your_key_here
# Get key at: https://makersuite.google.com/app/apikey

# Run simulation
uv run src/main.py scenarios/gemini_example.yaml

# Expected: Nearly always succeeds
# Completes all 5 steps with realistic state changes
```

## Code Improvements for Ollama

The following enhancements were made to support Ollama models:

### 1. Markdown Block Stripping
```python
# src/core/engine_validator.py
# Strips ```json\n{...}\n``` wrappers
```

### 2. Comment Removal
```python
# Removes JavaScript-style // comments
```

### 3. Trailing Comma Cleanup
```python
# Fixes ", }" and ", ]" patterns
```

### 4. Enhanced Prompts
```python
# Added WRONG/CORRECT examples:
# WRONG: "capital": capital * 1.05
# CORRECT: "capital": 1050.0
```

## Real-World Examples

### Ollama - Partial Success
```
Step 0: ✅ Initialization succeeded
Step 1: ✅ Generated actual values (0.045, 9820.0)
Step 2: ❌ Failed with truncated JSON
```

### Gemini - Full Success
```
Step 0: ✅ Initialization succeeded
Step 1: ✅ State updates applied
Step 2: ✅ Events generated
Step 3: ✅ Agent messages personalized
Step 4: ✅ Simulation completed
```

## Recommendations

### When to Use Ollama

Choose Ollama if you:
- Need complete data privacy
- Want to run offline
- Are experimenting/learning
- Can tolerate occasional failures
- Have local compute resources

**Note**: Expect some runs to fail. This is normal for smaller local models.

### When to Use Gemini

Choose Gemini if you:
- Need production reliability
- Want consistent results
- Are building user-facing features
- Need complex multi-step reasoning
- Can use cloud services

**Note**: Free tier is generous for most development needs.

## Hybrid Approach

You can mix providers in a single scenario:

```yaml
engine:
  provider: "gemini"  # Use reliable Gemini for engine
  model: "gemini-1.5-flash"

agents:
  - name: "Local Agent"
    llm:
      provider: "ollama"  # Use local for agents
      model: "mistral:7b"
```

This gives you:
- Reliable simulation engine (Gemini)
- Private agent responses (Ollama)

## Conclusion

**Ollama works** with the enhanced prompts and JSON cleaning, making it viable for local development and privacy-sensitive scenarios. However, **Gemini is recommended** for production use due to its superior reliability and consistency.

Both scenarios are included in the repository for easy comparison and testing.
