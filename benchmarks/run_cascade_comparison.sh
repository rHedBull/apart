#!/bin/bash
# Quick setup and run script for Cascade Monitor Comparison Benchmark

echo "=== AI Safety Monitor Model Comparison Benchmark ==="
echo ""
echo "This benchmark compares 3 models in the AI Safety Monitor role:"
echo "  1. Gemini 2.0 Flash Experimental (baseline)"
echo "  2. Gemini Exp 1206 (Pro-level)"
echo "  3. Grok Beta (xAI)"
echo ""

# Check API keys
echo "Checking API key configuration..."
if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ GEMINI_API_KEY not set"
    echo "   Get key from: https://makersuite.google.com/app/apikey"
    echo "   Add to .env: GEMINI_API_KEY=your_key_here"
    MISSING_KEYS=1
else
    echo "✓ GEMINI_API_KEY configured"
fi

if [ -z "$XAI_API_KEY" ]; then
    echo "⚠️  XAI_API_KEY not set (needed for Grok)"
    echo "   Get key from: https://console.x.ai/"
    echo "   Add to .env: XAI_API_KEY=your_key_here"
    echo ""
    echo "   Benchmark will run Gemini models only without this key."
    echo ""
fi

if [ -n "$MISSING_KEYS" ]; then
    echo ""
    echo "Please configure missing API keys in .env file and try again."
    exit 1
fi

echo ""
echo "Starting benchmark..."
echo "Expected duration: 15-30 minutes (3 runs × 10 turns each)"
echo ""

# Run the benchmark
python3 tools/benchmark.py benchmarks/cascade_monitor_comparison.yaml

echo ""
echo "=== Benchmark Complete ==="
echo "Results saved to: benchmarks/results/cascade_monitor_comparison/"
echo ""
echo "View results:"
echo "  - Markdown: cat benchmarks/results/cascade_monitor_comparison/comparison_table.md"
echo "  - HTML: open benchmarks/results/cascade_monitor_comparison/comparison_table.html"
echo "  - JSON: cat benchmarks/results/cascade_monitor_comparison/benchmark_summary.json"
