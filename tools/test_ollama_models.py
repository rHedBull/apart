#!/usr/bin/env python3
"""
Test different Ollama models for JSON generation quality.
Tests their ability to follow strict JSON formatting rules.
"""
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm.providers import OllamaProvider


# Test prompt that matches our actual use case
TEST_PROMPT = """You are a simulation engine. Generate a JSON response for a game state update.

Current State:
- interest_rate: 0.04
- market_volatility: 0.15
- Agent "Player1" capital: 1000.0

Task: Update the state based on this agent action:
Agent "Player1" says: "I invest 500 in high-risk stocks"

Return ONLY valid JSON (no markdown, no code blocks):
{
  "state_updates": {
    "global_vars": {
      "interest_rate": 0.041,
      "market_volatility": 0.16
    },
    "agent_vars": {
      "Player1": {
        "capital": 1200.5
      }
    }
  },
  "events": [
    {
      "type": "investment",
      "description": "Player1 invested in stocks"
    }
  ],
  "agent_messages": {
    "Player1": "Your investment was successful"
  },
  "reasoning": "Market conditions favorable"
}

CRITICAL RULES:
- Return ONLY the JSON object, no explanation text
- Use actual numeric values, NOT expressions (e.g., 1200.5, not "capital * 1.2")
- Do NOT use comments (//) - they are invalid in JSON
- Match variable types exactly: int=integer, float=decimal number
"""

SYSTEM_PROMPT = """You are a simulation engine that generates valid JSON responses.
You must follow JSON syntax exactly. Never use code, expressions, or comments."""


def test_model(model_name: str) -> dict:
    """Test a single Ollama model."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print('='*60)

    try:
        provider = OllamaProvider(
            model=model_name,
            base_url="http://localhost:11434"
        )

        # Generate response
        print("Generating response...")
        response = provider.generate_response(
            prompt=TEST_PROMPT,
            system_prompt=SYSTEM_PROMPT
        )

        print(f"\nRaw response length: {len(response)} chars")
        print(f"First 200 chars: {response[:200]}")

        # Try to parse as JSON
        try:
            # Strip markdown code blocks if present
            cleaned = response.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                lines = lines[1:]  # Remove first line
                for i, line in enumerate(lines):
                    if line.strip() == "```":
                        lines = lines[:i]
                        break
                cleaned = "\n".join(lines)

            # Remove comments
            import re
            lines = cleaned.split("\n")
            cleaned_lines = []
            for line in lines:
                comment_pos = line.find("//")
                if comment_pos != -1:
                    line = line[:comment_pos].rstrip()
                cleaned_lines.append(line)
            cleaned = "\n".join(cleaned_lines)

            # Remove trailing commas
            cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)

            # Try to parse
            data = json.loads(cleaned)

            # Validate structure
            required_keys = ["state_updates", "events", "agent_messages", "reasoning"]
            missing = [k for k in required_keys if k not in data]

            if missing:
                return {
                    "model": model_name,
                    "success": False,
                    "error": f"Missing keys: {missing}",
                    "has_markdown": "```" in response,
                    "has_comments": "//" in response,
                    "has_expressions": any(expr in response for expr in ["Math.", "capital *", "* 1.", "+ 0."]),
                    "response_preview": response[:300]
                }

            # Check for expressions instead of values
            response_lower = response.lower()
            has_expressions = any(expr in response for expr in [
                "Math.random", "capital *", "interest_rate +", "market_volatility +",
                "* 1.", "+ 0.", "- 0."
            ])

            return {
                "model": model_name,
                "success": True,
                "has_markdown": "```" in response,
                "has_comments": "//" in response,
                "has_expressions": has_expressions,
                "valid_json": True,
                "response_preview": response[:300]
            }

        except json.JSONDecodeError as e:
            return {
                "model": model_name,
                "success": False,
                "error": f"JSON parse error: {e}",
                "has_markdown": "```" in response,
                "has_comments": "//" in response,
                "has_expressions": any(expr in response for expr in ["Math.", "capital *", "* 1."]),
                "response_preview": response[:300]
            }

    except Exception as e:
        return {
            "model": model_name,
            "success": False,
            "error": f"Provider error: {e}",
            "has_markdown": False,
            "has_comments": False,
            "has_expressions": False,
            "response_preview": ""
        }


def main():
    """Test multiple Ollama models."""
    # Models to test (installed on system)
    models = [
        "gemma3:1b",           # Smallest - 815 MB
        "deepseek-r1:1.5b",    # Small reasoning model
        "mistral:7b",          # 7B general purpose
        "llama3.1:8b",         # 8B general purpose
        "codellama:latest",    # Code-focused 7B
        "deepseek-r1:7b",      # 7B reasoning model
        "deepseek-coder-v2:latest",  # Large code model
        "deepseek-r1:14b",     # 14B reasoning model
    ]

    print("Ollama Model JSON Generation Test")
    print("="*60)
    print("Testing models' ability to generate valid JSON...")
    print("This will test each model with a realistic simulation prompt.")

    results = []
    for model in models:
        result = test_model(model)
        results.append(result)

        # Print immediate result
        if result["success"]:
            status = "✅ SUCCESS"
        else:
            status = f"❌ FAILED: {result['error']}"

        print(f"\n{status}")
        print(f"  Has markdown blocks: {result['has_markdown']}")
        print(f"  Has comments (//): {result['has_comments']}")
        print(f"  Has expressions: {result['has_expressions']}")
        print(f"  Preview: {result['response_preview'][:150]}...")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Model':<20} {'Status':<10} {'Markdown':<10} {'Comments':<10} {'Expressions':<12}")
    print("-"*60)

    for r in results:
        status = "✅ PASS" if r["success"] else "❌ FAIL"
        markdown = "Yes" if r.get("has_markdown") else "No"
        comments = "Yes" if r.get("has_comments") else "No"
        expressions = "Yes" if r.get("has_expressions") else "No"

        print(f"{r['model']:<20} {status:<10} {markdown:<10} {comments:<10} {expressions:<12}")

    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)

    good_models = [r for r in results if r["success"] and not r.get("has_expressions", False)]
    problematic_models = [r for r in results if not r["success"] or r.get("has_expressions", False)]

    if good_models:
        print("✅ Recommended models (clean JSON output):")
        for r in good_models:
            print(f"   - {r['model']}")

    if problematic_models:
        print("\n⚠️  Problematic models (avoid for simulation engine):")
        for r in problematic_models:
            issues = []
            if not r["success"]:
                issues.append("invalid JSON")
            if r.get("has_expressions"):
                issues.append("uses expressions")
            print(f"   - {r['model']}: {', '.join(issues)}")


if __name__ == "__main__":
    main()
