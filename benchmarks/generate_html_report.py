#!/usr/bin/env python3
"""
HTML Report Generator for Benchmark Results

Generates interactive HTML reports from benchmark JSON results.
Includes comparison views, performance charts, and regression highlights.

Usage:
    python generate_html_report.py [--result-dir DIR] [--output FILE]
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


# ============================================================================
# HTML Templates
# ============================================================================

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smatrix_2D Benchmark Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        .header .meta {{
            font-size: 0.9em;
            opacity: 0.9;
        }}

        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
        }}

        .summary-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            text-align: center;
        }}

        .summary-card h3 {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 10px;
            text-transform: uppercase;
        }}

        .summary-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }}

        .summary-card.pass .value {{
            color: #28a745;
        }}

        .summary-card.fail .value {{
            color: #dc3545;
        }}

        .tabs {{
            display: flex;
            background: #f1f3f5;
            border-bottom: 2px solid #dee2e6;
        }}

        .tab {{
            flex: 1;
            padding: 15px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: 500;
            color: #666;
        }}

        .tab:hover {{
            background: #e9ecef;
        }}

        .tab.active {{
            background: white;
            color: #667eea;
            border-bottom: 2px solid #667eea;
        }}

        .tab-content {{
            display: none;
            padding: 30px;
        }}

        .tab-content.active {{
            display: block;
        }}

        .config-section {{
            margin-bottom: 40px;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            overflow: hidden;
        }}

        .config-header {{
            background: #f8f9fa;
            padding: 20px;
            border-bottom: 1px solid #dee2e6;
        }}

        .config-header h2 {{
            color: #333;
            margin-bottom: 5px;
        }}

        .config-header .meta {{
            color: #666;
            font-size: 0.9em;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            padding: 20px;
        }}

        .metric-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #667eea;
        }}

        .metric-card.regression {{
            border-left-color: #dc3545;
            background: #ffe6e6;
        }}

        .metric-card.improvement {{
            border-left-color: #28a745;
            background: #e6ffe6;
        }}

        .metric-label {{
            font-size: 0.85em;
            color: #666;
            margin-bottom: 5px;
        }}

        .metric-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #333;
        }}

        .metric-change {{
            font-size: 0.9em;
            margin-top: 5px;
        }}

        .metric-change.positive {{
            color: #28a745;
        }}

        .metric-change.negative {{
            color: #dc3545;
        }}

        .chart {{
            margin: 20px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }}

        .chart-bar {{
            height: 30px;
            margin: 10px 0;
            background: #e9ecef;
            border-radius: 4px;
            overflow: hidden;
            position: relative;
        }}

        .chart-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.5s ease;
            display: flex;
            align-items: center;
            padding-left: 10px;
            color: white;
            font-weight: 500;
        }}

        .chart-label {{
            position: absolute;
            left: 10px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 0.85em;
            color: #333;
            z-index: 1;
        }}

        .regression-alert {{
            background: #ffe6e6;
            border: 1px solid #dc3545;
            border-radius: 6px;
            padding: 15px;
            margin: 10px 0;
        }}

        .regression-alert h4 {{
            color: #dc3545;
            margin-bottom: 10px;
        }}

        .regression-item {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #ffc9c9;
        }}

        .regression-item:last-child {{
            border-bottom: none;
        }}

        .timestamp {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.85em;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}

        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}

        th {{
            background: #f8f9fa;
            font-weight: 600;
            color: #333;
        }}

        tr:hover {{
            background: #f8f9fa;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Smatrix_2D Benchmark Report</h1>
            <div class="meta">
                Generated: {timestamp} |
                Git Commit: {git_commit} |
                GPU: {gpu_name}
            </div>
        </div>

        <div class="summary">
            {summary_cards}
        </div>

        <div class="tabs">
            <div class="tab active" onclick="showTab('overview')">Overview</div>
            <div class="tab" onclick="showTab('details')">Detailed Results</div>
            <div class="tab" onclick="showTab('comparison')">Comparison</div>
        </div>

        <div id="overview" class="tab-content active">
            {overview_content}
        </div>

        <div id="details" class="tab-content">
            {details_content}
        </div>

        <div id="comparison" class="tab-content">
            {comparison_content}
        </div>

        <div class="timestamp">
            Report generated by Smatrix_2D Automated Benchmarking Suite
        </div>
    </div>

    <script>
        function showTab(tabName) {{
            // Hide all tabs
            const tabs = document.querySelectorAll('.tab-content');
            tabs.forEach(tab => tab.classList.remove('active'));

            const tabButtons = document.querySelectorAll('.tab');
            tabButtons.forEach(btn => btn.classList.remove('active'));

            // Show selected tab
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
        }}
    </script>
</body>
</html>
"""


# ============================================================================
# Report Generator
# ============================================================================

class HTMLReportGenerator:
    """Generate HTML reports from benchmark results."""

    def __init__(self, result_dir: Path):
        """Initialize report generator.

        Args:
            result_dir: Directory containing benchmark JSON results
        """
        self.result_dir = Path(result_dir)
        self.results = []
        self.reports = []

        # Load all results
        self._load_results()

    def _load_results(self):
        """Load all benchmark results from directory."""
        for json_file in self.result_dir.glob("*.json"):
            if 'report' in json_file.name:
                # Load regression report
                with open(json_file) as f:
                    self.reports.append(json.load(f))
            else:
                # Load benchmark result
                with open(json_file) as f:
                    self.results.append(json.load(f))

    def generate(self, output_file: Path) -> None:
        """Generate HTML report.

        Args:
            output_file: Output HTML file path
        """
        # Generate sections
        summary_cards = self._generate_summary()
        overview_content = self._generate_overview()
        details_content = self._generate_details()
        comparison_content = self._generate_comparison()

        # Get metadata
        if self.results:
            latest = self.results[-1]
            timestamp = latest.get('timestamp', datetime.now().isoformat())
            git_commit = latest.get('git_commit', 'unknown')
            gpu_name = latest.get('gpu_info', {}).get('device', 'Unknown')
        else:
            timestamp = datetime.now().isoformat()
            git_commit = 'unknown'
            gpu_name = 'Unknown'

        # Render template
        html = HTML_TEMPLATE.format(
            timestamp=timestamp,
            git_commit=git_commit,
            gpu_name=gpu_name,
            summary_cards=summary_cards,
            overview_content=overview_content,
            details_content=details_content,
            comparison_content=comparison_content,
        )

        # Write output
        with open(output_file, 'w') as f:
            f.write(html)

        print(f"HTML report generated: {output_file}")

    def _generate_summary(self) -> str:
        """Generate summary cards section."""
        cards = []

        # Count configurations
        configs = set(r['config_name'] for r in self.results)
        cards.append(f"""
            <div class="summary-card">
                <h3>Configurations</h3>
                <div class="value">{len(configs)}</div>
            </div>
        """)

        # Count regressions
        regressions = sum(1 for r in self.reports if r.get('has_regression', False))
        regression_class = 'fail' if regressions > 0 else 'pass'
        cards.append(f"""
            <div class="summary-card {regression_class}">
                <h3>Regressions</h3>
                <div class="value">{regressions}</div>
            </div>
        """)

        # Average performance
        if self.results:
            avg_steps_per_sec = np.mean([r['steps_per_second'] for r in self.results])
            cards.append(f"""
                <div class="summary-card">
                    <h3>Avg Speed</h3>
                    <div class="value">{avg_steps_per_sec:.1f}</div>
                    <div style="font-size: 0.8em; color: #666;">steps/sec</div>
                </div>
            """)

            # Conservation pass rate
            pass_count = sum(1 for r in self.results if r.get('conservation_valid', False))
            pass_rate = pass_count / len(self.results) * 100
            pass_class = 'pass' if pass_rate == 100 else 'fail'
            cards.append(f"""
                <div class="summary-card {pass_class}">
                    <h3>Conservation</h3>
                    <div class="value">{pass_rate:.0f}%</div>
                    <div style="font-size: 0.8em; color: #666;">pass rate</div>
                </div>
            """)

        return '\n'.join(cards)

    def _generate_overview(self) -> str:
        """Generate overview tab content."""
        content = []

        for result in sorted(self.results, key=lambda r: r['config_name']):
            config_name = result['config_name']

            # Config header
            content.append(f"""
                <div class="config-section">
                    <div class="config-header">
                        <h2>{config_name}</h2>
                        <div class="meta">
                            {result['grid']['Nx']}×{result['grid']['Nz']}×{result['grid']['Ntheta']}×{result['grid']['Ne']} |
                            {result['total_dof']:,} DOF |
                            {result['grid_size_mb']:.1f} MB
                        </div>
                    </div>
                    <div class="metrics-grid">
            """)

            # Performance metrics
            content.append(f"""
                <div class="metric-card">
                    <div class="metric-label">Average Step Time</div>
                    <div class="metric-value">{result['avg_step_time_ms']:.2f} ms</div>
                    <div class="metric-change">
                        {result['steps_per_second']:.1f} steps/sec
                    </div>
                </div>
            """)

            content.append(f"""
                <div class="metric-card">
                    <div class="metric-label">Total Runtime</div>
                    <div class="metric-value">{result['total_time_sec']:.2f} s</div>
                </div>
            """)

            memory_mb = result['memory_usage']['total_mb']
            content.append(f"""
                <div class="metric-card">
                    <div class="metric-label">Memory Usage</div>
                    <div class="metric-value">{memory_mb:.0f} MB</div>
                </div>
            """)

            # Conservation status
            cons_valid = result.get('conservation_valid', False)
            cons_class = 'pass' if cons_valid else 'fail'
            content.append(f"""
                <div class="metric-card {cons_class}">
                    <div class="metric-label">Conservation</div>
                    <div class="metric-value">{'PASS' if cons_valid else 'FAIL'}</div>
                    <div class="metric-change">
                        Error: {result['conservation_error']:.2e}
                    </div>
                </div>
            """)

            content.append('</div></div>')

        return '\n'.join(content)

    def _generate_details(self) -> str:
        """Generate detailed results tab content."""
        content = []

        for result in sorted(self.results, key=lambda r: r['config_name']):
            config_name = result['config_name']

            content.append(f"""
                <div class="config-section">
                    <div class="config-header">
                        <h2>{config_name} - Detailed Metrics</h2>
                    </div>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Configuration</td>
                            <td>{result['grid']['Nx']}×{result['grid']['Nz']}×{result['grid']['Ntheta']}×{result['grid']['Ne']}</td>
                        </tr>
                        <tr>
                            <td>Total DOF</td>
                            <td>{result['total_dof']:,}</td>
                        </tr>
                        <tr>
                            <td>Grid Size</td>
                            <td>{result['grid_size_mb']:.2f} MB</td>
                        </tr>
                        <tr>
                            <td>Phase Space</td>
                            <td>{result['memory_usage']['phase_space_mb']:.2f} MB</td>
                        </tr>
                        <tr>
                            <td>LUTs</td>
                            <td>{result['memory_usage']['luts_mb']:.2f} MB</td>
                        </tr>
                        <tr>
                            <td>Setup Time</td>
                            <td>{result['setup_time_sec']:.3f} s</td>
                        </tr>
                        <tr>
                            <td>Total Time</td>
                            <td>{result['total_time_sec']:.3f} s</td>
                        </tr>
                        <tr>
                            <td>Avg Step Time</td>
                            <td>{result['avg_step_time_ms']:.2f} ms</td>
                        </tr>
                        <tr>
                            <td>Min Step Time</td>
                            <td>{result['min_step_time_ms']:.2f} ms</td>
                        </tr>
                        <tr>
                            <td>Max Step Time</td>
                            <td>{result['max_step_time_ms']:.2f} ms</td>
                        </tr>
                        <tr>
                            <td>Steps/Second</td>
                            <td>{result['steps_per_second']:.1f}</td>
                        </tr>
                        <tr>
                            <td>GPU Memory</td>
                            <td>{result['memory_usage']['gpu_memory_mb']:.0f} MB</td>
                        </tr>
                        <tr>
                            <td>CPU Memory</td>
                            <td>{result['memory_usage']['cpu_memory_mb']:.0f} MB</td>
                        </tr>
                        <tr>
                            <td>Final Mass</td>
                            <td>{result['final_mass']:.6e}</td>
                        </tr>
                        <tr>
                            <td>Deposited Energy</td>
                            <td>{result['total_deposited_energy']:.6e} MeV</td>
                        </tr>
                        <tr>
                            <td>Conservation Error</td>
                            <td>{result['conservation_error']:.2e}</td>
                        </tr>
                        <tr>
                            <td>Conservation Valid</td>
                            <td>{'Yes' if result['conservation_valid'] else 'No'}</td>
                        </tr>
                    </table>
                </div>
            """)

        return '\n'.join(content)

    def _generate_comparison(self) -> str:
        """Generate comparison tab content with regressions."""
        content = []

        # Group reports by config
        reports_by_config = {}
        for report in self.reports:
            config_name = report['config_name']
            if config_name not in reports_by_config:
                reports_by_config[config_name] = []
            reports_by_config[config_name].append(report)

        # Show regressions
        if reports_by_config:
            content.append("<h2>Regression Analysis</h2>")

            for config_name, reports in sorted(reports_by_config.items()):
                for report in reports:
                    if report.get('has_regression', False):
                        content.append(f"""
                            <div class="regression-alert">
                                <h4>Regression Detected: {config_name}</h4>
                        """)

                        for reg in report.get('regressions', []):
                            content.append(f"""
                                <div class="regression-item">
                                    <span><strong>{reg['metric']}</strong></span>
                                    <span>
                                        Baseline: {reg['baseline']:.2f} →
                                        Current: {reg['current']:.2f}
                                        (<span class="negative">+{reg['change_pct']:.1f}%</span>)
                                    </span>
                                </div>
                            """)

                        content.append("</div>")

                    # Show improvements
                    if report.get('improvements'):
                        content.append(f"""
                            <div class="config-section">
                                <div class="config-header">
                                    <h2>Improvements: {config_name}</h2>
                                </div>
                        """)

                        for imp in report.get('improvements', []):
                            content.append(f"""
                                <div class="metric-card improvement">
                                    <div class="metric-label">{imp['metric']}</div>
                                    <div class="metric-value">{imp['change_pct']:+.1f}%</div>
                                    <div class="metric-change positive">
                                        {imp['baseline']:.2f} → {imp['current']:.2f}
                                    </div>
                                </div>
                            """)

                        content.append("</div>")

        # Performance comparison chart
        if self.results:
            content.append("<h2>Performance Comparison</h2>")

            for result in sorted(self.results, key=lambda r: r['config_name']):
                config_name = result['config_name']
                steps_per_sec = result['steps_per_second']

                # Scale bar width (assume max ~1000 steps/sec for visualization)
                bar_width = min(steps_per_sec / 10, 100)

                content.append(f"""
                    <div class="chart">
                        <div class="chart-bar">
                            <span class="chart-label">{config_name}: {steps_per_sec:.1f} steps/sec</span>
                            <div class="chart-fill" style="width: {bar_width}%;"></div>
                        </div>
                    </div>
                """)

        return '\n'.join(content) if content else "<p>No comparison data available</p>"


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main report generation."""
    parser = argparse.ArgumentParser(
        description='Generate HTML benchmark reports'
    )
    parser.add_argument(
        '--result-dir',
        type=Path,
        default=Path(__file__).parent / "results",
        help='Directory containing benchmark results (default: benchmarks/results/)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path(__file__).parent / "benchmark_report.html",
        help='Output HTML file (default: benchmarks/benchmark_report.html)'
    )

    args = parser.parse_args()

    # Generate report
    generator = HTMLReportGenerator(args.result_dir)
    generator.generate(args.output)


if __name__ == '__main__':
    main()
