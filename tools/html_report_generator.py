"""
HTML Report Generator for Benchmark Results

Generates self-contained HTML reports with interactive charts and visualizations
for benchmark results using Chart.js.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


class HTMLReportGenerator:
    """Generates interactive HTML reports from benchmark JSON results."""

    def __init__(self, results_data: List[Dict[str, Any]], benchmark_name: str,
                 benchmark_config: Dict[str, Any] = None, scenario_config: Dict[str, Any] = None):
        """
        Initialize the HTML report generator.

        Args:
            results_data: List of RunMetrics dictionaries from benchmark results
            benchmark_name: Name of the benchmark
            benchmark_config: Benchmark configuration dictionary
            scenario_config: Base scenario configuration dictionary
        """
        self.results = results_data
        self.benchmark_name = benchmark_name
        self.benchmark_config = benchmark_config or {}
        self.scenario_config = scenario_config or {}
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def generate(self, output_path: Path) -> None:
        """
        Generate the complete HTML report.

        Args:
            output_path: Path where the HTML file should be saved
        """
        html_content = self._build_html()
        output_path.write_text(html_content)

    def _build_html(self) -> str:
        """Build the complete HTML document."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.benchmark_name} - Benchmark Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        {self._get_css()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{self.benchmark_name}</h1>
            <p class="subtitle">Benchmark Report - Generated {self.timestamp}</p>
        </header>

        {self._build_scenario_section()}
        {self._build_summary_section()}
        {self._build_charts_section()}
        {self._build_details_section()}
    </div>

    <script>
        const benchmarkData = {json.dumps(self.results, indent=2)};
        {self._get_javascript()}
    </script>
</body>
</html>"""

    def _get_css(self) -> str:
        """Get the CSS styles for the report."""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Georgia', 'Times New Roman', serif;
            background: #f5f5f5;
            min-height: 100vh;
            padding: 40px 20px;
            color: #1a1a1a;
            line-height: 1.6;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.08);
        }

        header {
            background: #ffffff;
            color: #1a1a1a;
            padding: 60px 60px 40px 60px;
            border-bottom: 3px solid #1a1a1a;
        }

        header h1 {
            font-size: 2.2rem;
            margin-bottom: 15px;
            font-weight: 400;
            letter-spacing: -0.5px;
            color: #1a1a1a;
        }

        .subtitle {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            font-size: 0.95rem;
            color: #666;
            font-weight: 400;
        }

        .section {
            padding: 50px 60px;
            border-bottom: 1px solid #e0e0e0;
        }

        .section:last-child {
            border-bottom: none;
        }

        .section h2 {
            font-size: 1.5rem;
            margin-bottom: 30px;
            color: #1a1a1a;
            font-weight: 400;
            letter-spacing: -0.3px;
            text-transform: uppercase;
            font-size: 1.1rem;
            letter-spacing: 1px;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1px;
            background: #e0e0e0;
            border: 1px solid #e0e0e0;
            margin-bottom: 40px;
        }

        .summary-card {
            background: white;
            padding: 30px 20px;
            text-align: center;
        }

        .summary-card .value {
            font-size: 2.8rem;
            font-weight: 300;
            color: #1a1a1a;
            margin-bottom: 8px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }

        .summary-card .label {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            font-size: 0.75rem;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            font-weight: 500;
        }

        .chart-container {
            position: relative;
            height: 400px;
            margin-bottom: 50px;
            background: white;
            padding: 30px;
            border: 1px solid #e0e0e0;
        }

        .chart-title {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #666;
            margin-bottom: 20px;
            font-weight: 500;
        }

        .chart-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 30px;
            margin-bottom: 50px;
        }

        .details-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 30px;
            border: 1px solid #e0e0e0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            font-size: 0.9rem;
        }

        .details-table th {
            background: #f8f8f8;
            color: #1a1a1a;
            padding: 16px 20px;
            text-align: left;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 1.2px;
            border-bottom: 2px solid #1a1a1a;
        }

        .details-table td {
            padding: 16px 20px;
            border-bottom: 1px solid #e0e0e0;
        }

        .details-table tbody tr:hover {
            background: #fafafa;
        }

        .details-table tbody tr:last-child td {
            border-bottom: none;
        }

        .status-badge {
            display: inline-block;
            padding: 5px 14px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }

        .status-completed {
            background: #22c55e;
            color: white;
        }

        .status-failed {
            background: #ef4444;
            color: white;
        }

        .metric-value {
            font-weight: 500;
            color: #1a1a1a;
            font-variant-numeric: tabular-nums;
        }

        @media print {
            body {
                background: white;
                padding: 0;
            }

            .container {
                box-shadow: none;
            }
        }

        .scenario-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 30px;
            margin-bottom: 30px;
        }

        .scenario-card {
            background: #fafafa;
            border: 1px solid #e0e0e0;
            padding: 25px;
        }

        .scenario-card h3 {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 15px;
            color: #1a1a1a;
            font-weight: 600;
            border-bottom: 2px solid #1a1a1a;
            padding-bottom: 8px;
        }

        .scenario-card .info-row {
            margin-bottom: 12px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            font-size: 0.9rem;
        }

        .scenario-card .label {
            font-weight: 600;
            color: #666;
            display: inline-block;
            min-width: 120px;
        }

        .scenario-card .value {
            color: #1a1a1a;
        }

        .scenario-card code {
            background: #fff;
            padding: 2px 6px;
            border: 1px solid #e0e0e0;
            border-radius: 3px;
            font-size: 0.85rem;
        }

        .run-config {
            background: #fff;
            border: 1px solid #e0e0e0;
            padding: 20px;
            margin-bottom: 20px;
        }

        .run-config h4 {
            font-size: 0.9rem;
            font-weight: 600;
            margin-bottom: 12px;
            color: #1a1a1a;
        }

        .run-config ul {
            margin: 0;
            padding-left: 20px;
            list-style: none;
        }

        .run-config li {
            margin-bottom: 6px;
            font-size: 0.85rem;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }

        .run-config li:before {
            content: "→";
            margin-right: 8px;
            color: #666;
        }

        @media (max-width: 768px) {
            .chart-row {
                grid-template-columns: 1fr;
            }

            header, .section {
                padding: 30px 25px;
            }

            header h1 {
                font-size: 1.6rem;
            }

            .summary-grid {
                grid-template-columns: 1fr 1fr;
            }

            .scenario-grid {
                grid-template-columns: 1fr;
            }
        }
        """

    def _build_scenario_section(self) -> str:
        """Build the scenario configuration section."""
        if not self.benchmark_config and not self.scenario_config:
            return ""

        # Build benchmark runs HTML
        runs_html = ""
        model_pool = self.benchmark_config.get("model_pool", {})
        benchmark_runs = self.benchmark_config.get("benchmark_runs", [])

        for run in benchmark_runs:
            mappings = []
            for agent, model_id in run.get("agent_model_mapping", {}).items():
                model = model_pool.get(model_id, {})
                mappings.append(f"<li><code>{agent}</code> → {model_id} ({model.get('provider', 'N/A')}/{model.get('model', 'N/A')})</li>")

            if "engine_model" in run:
                engine_id = run["engine_model"]
                engine = model_pool.get(engine_id, {})
                mappings.append(f"<li><code>[Engine]</code> → {engine_id} ({engine.get('provider', 'N/A')}/{engine.get('model', 'N/A')})</li>")

            mappings_html = "\n".join(mappings)
            runs_html += f"""
            <div class="run-config">
                <h4>{run.get('name', 'Unnamed Run')}</h4>
                <p style="margin-bottom: 10px; color: #666; font-size: 0.85rem;">{run.get('description', 'No description')}</p>
                <ul>
                    {mappings_html}
                </ul>
            </div>
            """

        # Build scenario info HTML
        scenario_cards = []

        # Basic info card
        max_steps = self.scenario_config.get('max_steps', 'N/A')
        orch_msg = self.scenario_config.get('orchestrator_message', 'N/A')
        basic_info = f"""
        <div class="scenario-card">
            <h3>Basic Info</h3>
            <div class="info-row">
                <span class="label">Max Steps:</span>
                <span class="value">{max_steps}</span>
            </div>
            <div class="info-row">
                <span class="label">Orchestrator:</span>
                <span class="value">{orch_msg}</span>
            </div>
        </div>
        """
        scenario_cards.append(basic_info)

        # Agents card
        agents = self.scenario_config.get("agents", [])
        if agents:
            agents_rows = []
            for agent in agents:
                name = agent.get('name', 'Unnamed')
                vars = agent.get('variables', {})
                vars_str = ', '.join(f"{k}={v}" for k, v in vars.items()) if vars else 'None'
                agents_rows.append(f"""
                <div class="info-row">
                    <span class="label">{name}:</span>
                    <span class="value">{vars_str}</span>
                </div>
                """)

            agents_card = f"""
            <div class="scenario-card">
                <h3>Agents</h3>
                {''.join(agents_rows)}
            </div>
            """
            scenario_cards.append(agents_card)

        # Global variables card
        global_vars = self.scenario_config.get("global_vars", {})
        if global_vars:
            vars_rows = []
            for var_name, var_config in global_vars.items():
                var_type = var_config.get('type', 'any')
                default = var_config.get('default', 'N/A')
                vars_rows.append(f"""
                <div class="info-row">
                    <code>{var_name}</code> ({var_type}): {default}
                </div>
                """)

            vars_card = f"""
            <div class="scenario-card">
                <h3>Global Variables</h3>
                {''.join(vars_rows)}
            </div>
            """
            scenario_cards.append(vars_card)

        scenario_grid = f"""
        <div class="scenario-grid">
            {''.join(scenario_cards)}
        </div>
        """

        return f"""
        <div class="section">
            <h2>Scenario Configuration</h2>
            <h3 style="font-size: 1rem; margin-bottom: 20px; font-weight: 600; color: #1a1a1a;">Benchmark Runs</h3>
            {runs_html}
            <h3 style="font-size: 1rem; margin-bottom: 20px; margin-top: 30px; font-weight: 600; color: #1a1a1a;">Base Scenario</h3>
            {scenario_grid}
        </div>
        """

    def _build_summary_section(self) -> str:
        """Build the summary statistics section."""
        total_runs = len(self.results)
        completed_runs = sum(1 for r in self.results if r.get("completed", False))
        total_errors = sum(r.get("error_count", 0) for r in self.results)
        avg_time = sum(r.get("total_time", 0) for r in self.results) / total_runs if total_runs > 0 else 0

        return f"""
        <div class="section">
            <h2>Summary</h2>
            <div class="summary-grid">
                <div class="summary-card">
                    <div class="value">{total_runs}</div>
                    <div class="label">Total Runs</div>
                </div>
                <div class="summary-card">
                    <div class="value">{completed_runs}</div>
                    <div class="label">Completed</div>
                </div>
                <div class="summary-card">
                    <div class="value">{total_runs - completed_runs}</div>
                    <div class="label">Failed</div>
                </div>
                <div class="summary-card">
                    <div class="value">{total_errors}</div>
                    <div class="label">Total Errors</div>
                </div>
                <div class="summary-card">
                    <div class="value">{avg_time:.2f}s</div>
                    <div class="label">Avg Time</div>
                </div>
            </div>
        </div>
        """

    def _build_charts_section(self) -> str:
        """Build the charts section with multiple visualizations."""
        return """
        <div class="section">
            <h2>Performance Analysis</h2>
            <div class="chart-container">
                <div class="chart-title">Figure 1. Total Execution Time by Model Configuration</div>
                <canvas id="performanceChart"></canvas>
            </div>

            <div class="chart-row">
                <div class="chart-container">
                    <div class="chart-title">Figure 2. Step-by-Step Performance</div>
                    <canvas id="stepTimesChart"></canvas>
                </div>
                <div class="chart-container">
                    <div class="chart-title">Figure 3. Error Distribution</div>
                    <canvas id="errorChart"></canvas>
                </div>
            </div>

            <h2>Agent Behavior Analysis</h2>
            <div class="chart-container" style="height: 500px;">
                <div class="chart-title">Figure 4. Agent Variable Evolution Over Time</div>
                <canvas id="variableEvolutionChart"></canvas>
            </div>
        </div>
        """

    def _build_details_section(self) -> str:
        """Build the detailed results table."""
        rows = []
        for result in self.results:
            model_name = result.get("model_name", "Unknown")
            provider = result.get("provider", "Unknown")

            completed = result.get("completed", False)
            status_class = "status-completed" if completed else "status-failed"
            status_text = "✓ Completed" if completed else "✗ Failed"

            total_time = result.get("total_time", 0)
            avg_step_time = result.get("avg_step_time", 0)
            error_count = result.get("error_count", 0)
            violations = result.get("constraint_violations", 0)
            decisions = result.get("decision_count", 0)

            rows.append(f"""
                <tr>
                    <td><strong>{model_name}</strong></td>
                    <td>{provider}</td>
                    <td><span class="status-badge {status_class}">{status_text}</span></td>
                    <td class="metric-value">{total_time:.2f}s</td>
                    <td class="metric-value">{avg_step_time:.3f}s</td>
                    <td class="metric-value">{error_count}</td>
                    <td class="metric-value">{violations}</td>
                    <td class="metric-value">{decisions}</td>
                </tr>
            """)

        table_rows = "\n".join(rows)

        return f"""
        <div class="section">
            <h2>Detailed Results</h2>
            <table class="details-table">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Provider</th>
                        <th>Status</th>
                        <th>Total Time</th>
                        <th>Avg Step Time</th>
                        <th>Errors</th>
                        <th>Violations</th>
                        <th>Decisions</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>
        """

    def _get_javascript(self) -> str:
        """Get the JavaScript code for chart rendering."""
        return """
        // Chart.js configuration
        Chart.defaults.font.family = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
        Chart.defaults.font.size = 11;
        Chart.defaults.color = '#666';

        // Professional monochromatic color palette (grayscale with subtle variations)
        const chartColors = {
            gray1: '#1a1a1a',
            gray2: '#404040',
            gray3: '#666666',
            gray4: '#8c8c8c',
            gray5: '#b3b3b3',
            gray6: '#d9d9d9'
        };

        const gradientColors = [
            chartColors.gray1,
            chartColors.gray2,
            chartColors.gray3,
            chartColors.gray4,
            chartColors.gray5,
            chartColors.gray6
        ];

        // Performance Comparison Chart
        const performanceChart = new Chart(
            document.getElementById('performanceChart'),
            {
                type: 'bar',
                data: {
                    labels: benchmarkData.map(d => d.model_name),
                    datasets: [{
                        label: 'Total Time (seconds)',
                        data: benchmarkData.map(d => d.total_time || 0),
                        backgroundColor: gradientColors,
                        borderColor: gradientColors,
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: false
                        },
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Time (seconds)',
                                font: { size: 11 }
                            },
                            grid: {
                                color: '#e0e0e0',
                                borderColor: '#1a1a1a'
                            },
                            ticks: {
                                font: { size: 10 }
                            }
                        },
                        x: {
                            grid: {
                                display: false,
                                borderColor: '#1a1a1a'
                            },
                            ticks: {
                                font: { size: 10 }
                            }
                        }
                    }
                }
            }
        );

        // Step Times Chart (Line chart showing step-by-step performance)
        const stepTimesDatasets = benchmarkData.map((result, idx) => {
            const stepTimes = result.step_times || [];
            return {
                label: result.model_name,
                data: stepTimes,
                borderColor: gradientColors[idx % gradientColors.length],
                backgroundColor: 'transparent',
                tension: 0.1,
                fill: false,
                borderWidth: 2,
                pointRadius: 3,
                pointBackgroundColor: gradientColors[idx % gradientColors.length],
                pointBorderColor: '#fff',
                pointBorderWidth: 1
            };
        });

        const stepTimesChart = new Chart(
            document.getElementById('stepTimesChart'),
            {
                type: 'line',
                data: {
                    labels: benchmarkData[0]?.step_times?.map((_, i) => `Step ${i + 1}`) || [],
                    datasets: stepTimesDatasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: false
                        },
                        legend: {
                            display: true,
                            position: 'bottom',
                            labels: {
                                boxWidth: 12,
                                boxHeight: 12,
                                padding: 15,
                                font: { size: 10 }
                            }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Time (seconds)',
                                font: { size: 11 }
                            },
                            grid: {
                                color: '#e0e0e0',
                                borderColor: '#1a1a1a'
                            },
                            ticks: {
                                font: { size: 10 }
                            }
                        },
                        x: {
                            grid: {
                                display: false,
                                borderColor: '#1a1a1a'
                            },
                            ticks: {
                                font: { size: 10 }
                            }
                        }
                    }
                }
            }
        );

        // Error Distribution Chart
        const errorChart = new Chart(
            document.getElementById('errorChart'),
            {
                type: 'bar',
                data: {
                    labels: benchmarkData.map(d => d.model_name),
                    datasets: [{
                        label: 'Error Count',
                        data: benchmarkData.map(d => d.error_count || 0),
                        backgroundColor: chartColors.gray3,
                        borderColor: chartColors.gray1,
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: false
                        },
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: {
                                stepSize: 1,
                                font: { size: 10 }
                            },
                            grid: {
                                color: '#e0e0e0',
                                borderColor: '#1a1a1a'
                            }
                        },
                        x: {
                            grid: {
                                display: false,
                                borderColor: '#1a1a1a'
                            },
                            ticks: {
                                font: { size: 10 }
                            }
                        }
                    }
                }
            }
        );

        // Variable Evolution Chart (tracking agent variables over time)
        function extractVariableEvolution() {
            const datasets = [];

            benchmarkData.forEach((result, resultIdx) => {
                const variableChanges = result.variable_changes || [];

                // Extract agent variables (e.g., scores)
                const agentNames = variableChanges.length > 0 ? Object.keys(variableChanges[0].agent_vars || {}) : [];

                agentNames.forEach((agentName, agentIdx) => {
                    const scores = variableChanges.map(vc => vc.agent_vars[agentName]?.score || 0);

                    if (scores.some(s => s !== 0)) {
                        datasets.push({
                            label: `${result.model_name} - ${agentName} Score`,
                            data: scores,
                            borderColor: gradientColors[(resultIdx * agentNames.length + agentIdx) % gradientColors.length],
                            backgroundColor: 'transparent',
                            tension: 0.1,
                            borderWidth: 2,
                            pointRadius: 3,
                            pointBackgroundColor: gradientColors[(resultIdx * agentNames.length + agentIdx) % gradientColors.length],
                            pointBorderColor: '#fff',
                            pointBorderWidth: 1
                        });
                    }
                });
            });

            return datasets;
        }

        const variableDatasets = extractVariableEvolution();
        const maxSteps = Math.max(...benchmarkData.map(d => d.variable_changes?.length || 0));

        const variableEvolutionChart = new Chart(
            document.getElementById('variableEvolutionChart'),
            {
                type: 'line',
                data: {
                    labels: Array.from({ length: maxSteps }, (_, i) => `Step ${i + 1}`),
                    datasets: variableDatasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: false
                        },
                        legend: {
                            display: true,
                            position: 'bottom',
                            labels: {
                                boxWidth: 12,
                                boxHeight: 12,
                                padding: 15,
                                font: { size: 10 }
                            }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Value',
                                font: { size: 11 }
                            },
                            grid: {
                                color: '#e0e0e0',
                                borderColor: '#1a1a1a'
                            },
                            ticks: {
                                font: { size: 10 }
                            }
                        },
                        x: {
                            grid: {
                                display: false,
                                borderColor: '#1a1a1a'
                            },
                            ticks: {
                                font: { size: 10 }
                            }
                        }
                    }
                }
            }
        );
        """


def generate_html_report(
    json_results_path: Path,
    output_path: Path,
    benchmark_name: str,
    benchmark_config: Dict[str, Any] = None,
    scenario_config: Dict[str, Any] = None
) -> None:
    """
    Generate an HTML report from JSON benchmark results.

    Args:
        json_results_path: Path to the JSON results file
        output_path: Path where the HTML report should be saved
        benchmark_name: Name of the benchmark
        benchmark_config: Benchmark configuration dictionary
        scenario_config: Base scenario configuration dictionary

    Raises:
        FileNotFoundError: If the JSON results file doesn't exist
        json.JSONDecodeError: If the JSON file is malformed
    """
    # Load JSON results
    with json_results_path.open('r') as f:
        results_data = json.load(f)

    # Generate report
    generator = HTMLReportGenerator(results_data, benchmark_name, benchmark_config, scenario_config)
    generator.generate(output_path)

    print(f"✓ HTML report generated: {output_path}")
