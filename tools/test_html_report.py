"""
Test script for HTML report generator.
Creates sample benchmark data and generates a test HTML report.
"""

import json
from pathlib import Path
from html_report_generator import generate_html_report


# Sample benchmark data matching the RunMetrics structure
sample_data = [
    {
        "model_name": "gemini_A_mistral_B",
        "provider": "Agent A:gemini-flash, Agent B:mistral-7b",
        "start_time": "2025-11-23T10:00:00",
        "end_time": "2025-11-23T10:02:30",
        "Performance Metrics": {
            "total_time": 150.5,
            "step_times": [12.3, 11.8, 13.5, 12.1, 11.9, 13.2, 12.7, 13.0, 12.5, 11.8],
            "avg_step_time": 12.48,
            "total_tokens": 15000,
            "avg_tokens_per_step": 1500.0
        },
        "Quality Metrics": {
            "variable_changes": [
                {
                    "step": 1,
                    "global_vars": {"round_number": 1, "market_factor": 1.0},
                    "agent_vars": {
                        "Agent A": {"score": 100, "risk_level": 0.8},
                        "Agent B": {"score": 100, "risk_level": 0.3}
                    }
                },
                {
                    "step": 2,
                    "global_vars": {"round_number": 2, "market_factor": 1.05},
                    "agent_vars": {
                        "Agent A": {"score": 120, "risk_level": 0.85},
                        "Agent B": {"score": 105, "risk_level": 0.35}
                    }
                },
                {
                    "step": 3,
                    "global_vars": {"round_number": 3, "market_factor": 0.95},
                    "agent_vars": {
                        "Agent A": {"score": 135, "risk_level": 0.75},
                        "Agent B": {"score": 110, "risk_level": 0.30}
                    }
                },
                {
                    "step": 4,
                    "global_vars": {"round_number": 4, "market_factor": 1.1},
                    "agent_vars": {
                        "Agent A": {"score": 155, "risk_level": 0.90},
                        "Agent B": {"score": 118, "risk_level": 0.28}
                    }
                },
                {
                    "step": 5,
                    "global_vars": {"round_number": 5, "market_factor": 1.0},
                    "agent_vars": {
                        "Agent A": {"score": 170, "risk_level": 0.88},
                        "Agent B": {"score": 125, "risk_level": 0.32}
                    }
                }
            ],
            "final_state": {
                "game_state": {"resources": 295, "difficulty": "medium", "round": 5},
                "global_vars": {"round_number": 5, "market_factor": 1.0},
                "agent_vars": {
                    "Agent A": {"score": 170, "risk_level": 0.88},
                    "Agent B": {"score": 125, "risk_level": 0.32}
                }
            },
            "decision_count": 50,
            "constraint_violations": 2
        },
        "Reliability Metrics": {
            "completed": True,
            "error_count": 0,
            "step_failures": [],
            "error_messages": []
        },
        "Custom Metrics": {
            "custom_metrics": {
                "interest_rate": 0.05,
                "market_volatility": 0.15
            }
        }
    },
    {
        "model_name": "both_mistral",
        "provider": "Agent A:mistral-7b, Agent B:mistral-7b",
        "start_time": "2025-11-23T10:05:00",
        "end_time": "2025-11-23T10:08:45",
        "Performance Metrics": {
            "total_time": 225.3,
            "step_times": [18.5, 19.2, 21.5, 20.1, 22.3, 23.5, 21.8, 22.5, 24.1, 23.8],
            "avg_step_time": 21.73,
            "total_tokens": 12000,
            "avg_tokens_per_step": 1200.0
        },
        "Quality Metrics": {
            "variable_changes": [
                {
                    "step": 1,
                    "global_vars": {"round_number": 1, "market_factor": 1.0},
                    "agent_vars": {
                        "Agent A": {"score": 100, "risk_level": 0.5},
                        "Agent B": {"score": 100, "risk_level": 0.3}
                    }
                },
                {
                    "step": 2,
                    "global_vars": {"round_number": 2, "market_factor": 1.05},
                    "agent_vars": {
                        "Agent A": {"score": 110, "risk_level": 0.52},
                        "Agent B": {"score": 108, "risk_level": 0.32}
                    }
                },
                {
                    "step": 3,
                    "global_vars": {"round_number": 3, "market_factor": 0.95},
                    "agent_vars": {
                        "Agent A": {"score": 118, "risk_level": 0.48},
                        "Agent B": {"score": 115, "risk_level": 0.28}
                    }
                },
                {
                    "step": 4,
                    "global_vars": {"round_number": 4, "market_factor": 1.1},
                    "agent_vars": {
                        "Agent A": {"score": 130, "risk_level": 0.55},
                        "Agent B": {"score": 125, "risk_level": 0.30}
                    }
                },
                {
                    "step": 5,
                    "global_vars": {"round_number": 5, "market_factor": 1.0},
                    "agent_vars": {
                        "Agent A": {"score": 140, "risk_level": 0.53},
                        "Agent B": {"score": 135, "risk_level": 0.31}
                    }
                }
            ],
            "final_state": {
                "game_state": {"resources": 275, "difficulty": "medium", "round": 5},
                "global_vars": {"round_number": 5, "market_factor": 1.0},
                "agent_vars": {
                    "Agent A": {"score": 140, "risk_level": 0.53},
                    "Agent B": {"score": 135, "risk_level": 0.31}
                }
            },
            "decision_count": 48,
            "constraint_violations": 1
        },
        "Reliability Metrics": {
            "completed": True,
            "error_count": 0,
            "step_failures": [],
            "error_messages": []
        },
        "Custom Metrics": {
            "custom_metrics": {
                "interest_rate": 0.05,
                "market_volatility": 0.12
            }
        }
    },
    {
        "model_name": "mistral_A_gemini_B",
        "provider": "Agent A:mistral-7b, Agent B:gemini-flash",
        "start_time": "2025-11-23T10:10:00",
        "end_time": "2025-11-23T10:11:20",
        "Performance Metrics": {
            "total_time": 80.5,
            "step_times": [10.5, 9.8, 8.5, 7.1, 6.9],
            "avg_step_time": 8.56,
            "total_tokens": 8000,
            "avg_tokens_per_step": 1600.0
        },
        "Quality Metrics": {
            "variable_changes": [
                {
                    "step": 1,
                    "global_vars": {"round_number": 1, "market_factor": 1.0},
                    "agent_vars": {
                        "Agent A": {"score": 100, "risk_level": 0.5},
                        "Agent B": {"score": 100, "risk_level": 0.4}
                    }
                },
                {
                    "step": 2,
                    "global_vars": {"round_number": 2, "market_factor": 1.05},
                    "agent_vars": {
                        "Agent A": {"score": 115, "risk_level": 0.52},
                        "Agent B": {"score": 112, "risk_level": 0.42}
                    }
                },
                {
                    "step": 3,
                    "global_vars": {"round_number": 3, "market_factor": 0.95},
                    "agent_vars": {
                        "Agent A": {"score": 125, "risk_level": 0.48},
                        "Agent B": {"score": 120, "risk_level": 0.38}
                    }
                },
                {
                    "step": 4,
                    "global_vars": {"round_number": 4, "market_factor": 1.1},
                    "agent_vars": {
                        "Agent A": {"score": 140, "risk_level": 0.55},
                        "Agent B": {"score": 135, "risk_level": 0.45}
                    }
                },
                {
                    "step": 5,
                    "global_vars": {"round_number": 5, "market_factor": 1.0},
                    "agent_vars": {
                        "Agent A": {"score": 150, "risk_level": 0.53},
                        "Agent B": {"score": 148, "risk_level": 0.43}
                    }
                }
            ],
            "final_state": {
                "game_state": {"resources": 298, "difficulty": "easy", "round": 5},
                "global_vars": {"round_number": 5, "market_factor": 1.0},
                "agent_vars": {
                    "Agent A": {"score": 150, "risk_level": 0.53},
                    "Agent B": {"score": 148, "risk_level": 0.43}
                }
            },
            "decision_count": 45,
            "constraint_violations": 5
        },
        "Reliability Metrics": {
            "completed": False,
            "error_count": 3,
            "step_failures": [3, 4],
            "error_messages": [
                "Agent timeout at step 3",
                "Rate limit exceeded at step 4",
                "Invalid response format"
            ]
        },
        "Custom Metrics": {
            "custom_metrics": {
                "interest_rate": 0.05,
                "market_volatility": 0.18
            }
        }
    }
]


def main():
    """Test the HTML report generator."""
    print("Testing HTML Report Generator...")

    # Create test output directory
    test_dir = Path("benchmarks/results/test")
    test_dir.mkdir(parents=True, exist_ok=True)

    # Save sample JSON
    json_path = test_dir / "sample_benchmark.json"
    with json_path.open('w') as f:
        json.dump(sample_data, f, indent=2)
    print(f"✓ Created sample JSON: {json_path}")

    # Generate HTML report
    html_path = test_dir / "sample_benchmark.html"
    generate_html_report(json_path, html_path, "Sample Benchmark Test")

    print(f"\n{'='*70}")
    print("HTML Report Generated Successfully!")
    print(f"{'='*70}")
    print(f"\nOpen the report in your browser:")
    print(f"  file://{html_path.absolute()}")
    print(f"\nOr run:")
    print(f"  xdg-open {html_path}")
    print()


if __name__ == "__main__":
    main()
