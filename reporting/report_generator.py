###############################################################################
# Render HTML / Markdown / CSV from BenchmarkReport
###############################################################################
import json
from pathlib import Path
from typing import Dict, Any
import csv

from benchmarks.benchmark_runner import BenchmarkReport


def render_report(report: BenchmarkReport, output_dir: str) -> None:
    """Render benchmark report in multiple formats."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Generate timestamp-based filename prefix
    prefix = f"{report.experiment_name}_{report.timestamp}"
    
    # Render in different formats
    _render_json(report, output_path / f"{prefix}_report.json")
    _render_markdown(report, output_path / f"{prefix}_report.md")
    _render_csv(report, output_path / f"{prefix}_results.csv")
    _render_html(report, output_path / f"{prefix}_report.html")
    
    print(f"Reports generated in: {output_path}")


def _render_json(report: BenchmarkReport, filepath: Path) -> None:
    """Render report as JSON."""
    with open(filepath, 'w') as f:
        json.dump(report.to_dict(), f, indent=2)


def _render_markdown(report: BenchmarkReport, filepath: Path) -> None:
    """Render report as Markdown."""
    content = _generate_markdown_content(report)
    filepath.write_text(content)


def _render_csv(report: BenchmarkReport, filepath: Path) -> None:
    """Render results as CSV."""
    if not report.results:
        return
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = [
            'genome_id', 'fitness', 'execution_time',
            'lora_rank', 'lora_alpha', 'lora_dropout',
            'ca_complexity', 'ca_intensity', 'ca_periodicity', 'ca_convergence'
        ]
        writer.writerow(header)
        
        # Data rows
        for result in report.results:
            row = [
                result.genome_id,
                result.fitness,
                result.execution_time,
                result.lora_config['r'],
                result.lora_config['alpha'],
                result.lora_config['dropout'],
                result.ca_features['complexity'],
                result.ca_features['intensity'],
                result.ca_features['periodicity'],
                result.ca_features['convergence']
            ]
            writer.writerow(row)


def _render_html(report: BenchmarkReport, filepath: Path) -> None:
    """Render report as HTML."""
    content = _generate_html_content(report)
    filepath.write_text(content)


def _generate_markdown_content(report: BenchmarkReport) -> str:
    """Generate Markdown content for the report."""
    content = f"""# CoralX Experiment Report

## Experiment Details
- **Name**: {report.experiment_name}
- **Timestamp**: {report.timestamp}
- **Number of Genomes**: {len(report.results)}

## Configuration
```json
{json.dumps(report.config, indent=2)}
```

## Results Summary

### Best Genome
"""
    
    if report.best_genome:
        content += f"""
- **ID**: {report.best_genome.genome_id}
- **Fitness**: {report.best_genome.fitness:.4f}
- **LoRA Rank**: {report.best_genome.lora_config['r']}
- **LoRA Alpha**: {report.best_genome.lora_config['alpha']:.2f}
- **LoRA Dropout**: {report.best_genome.lora_config['dropout']:.3f}
- **CA Complexity**: {report.best_genome.ca_features['complexity']:.4f}
- **CA Intensity**: {report.best_genome.ca_features['intensity']:.4f}
- **Execution Time**: {report.best_genome.execution_time:.2f}s
"""
    else:
        content += "\nNo evaluated genomes found."
    
    content += f"""
### Statistics
"""
    
    for key, value in report.statistics.items():
        content += f"- **{key.replace('_', ' ').title()}**: {value:.4f}\n"
    
    content += """
## Individual Results

| Genome ID | Fitness | LoRA Rank | LoRA Alpha | LoRA Dropout | CA Complexity | CA Intensity | Exec Time |
|-----------|---------|-----------|------------|--------------|---------------|--------------|-----------|
"""
    
    for result in sorted(report.results, key=lambda r: r.fitness, reverse=True):
        content += f"| {result.genome_id} | {result.fitness:.4f} | {result.lora_config['r']} | "
        content += f"{result.lora_config['alpha']:.2f} | {result.lora_config['dropout']:.3f} | "
        content += f"{result.ca_features['complexity']:.4f} | {result.ca_features['intensity']:.4f} | "
        content += f"{result.execution_time:.2f}s |\n"
    
    return content


def _generate_html_content(report: BenchmarkReport) -> str:
    """Generate HTML content for the report."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report.experiment_name} - CoralX Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #2980b9; }}
        .stat-label {{ font-size: 12px; color: #7f8c8d; text-transform: uppercase; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .best-genome {{
            background: #d5f4e6;
            border: 2px solid #27ae60;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .config-box {{
            background: #2c3e50;
            color: white;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>CoralX Experiment Report</h1>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{len(report.results)}</div>
                <div class="stat-label">Genomes Evaluated</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{report.statistics.get('max_fitness', 0):.3f}</div>
                <div class="stat-label">Best Fitness</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{report.statistics.get('mean_fitness', 0):.3f}</div>
                <div class="stat-label">Mean Fitness</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{report.statistics.get('total_execution_time', 0):.1f}s</div>
                <div class="stat-label">Total Time</div>
            </div>
        </div>

        <h2>Experiment Details</h2>
        <p><strong>Name:</strong> {report.experiment_name}</p>
        <p><strong>Timestamp:</strong> {report.timestamp}</p>
        
        {_generate_best_genome_html(report.best_genome)}
        
        <h2>Configuration</h2>
        <div class="config-box">
            <pre>{json.dumps(report.config, indent=2)}</pre>
        </div>
        
        <h2>All Results</h2>
        {_generate_results_table_html(report.results)}
        
        <h2>Statistics</h2>
        {_generate_statistics_html(report.statistics)}
    </div>
</body>
</html>"""


def _generate_best_genome_html(best_genome) -> str:
    """Generate HTML for best genome section."""
    if not best_genome:
        return "<p>No evaluated genomes found.</p>"
    
    return f"""
        <h2>Best Genome</h2>
        <div class="best-genome">
            <h3>{best_genome.genome_id}</h3>
            <p><strong>Fitness:</strong> {best_genome.fitness:.4f}</p>
            <p><strong>LoRA Configuration:</strong> r={best_genome.lora_config['r']}, 
               α={best_genome.lora_config['alpha']:.2f}, 
               dropout={best_genome.lora_config['dropout']:.3f}</p>
            <p><strong>CA Features:</strong> 
               complexity={best_genome.ca_features['complexity']:.4f},
               intensity={best_genome.ca_features['intensity']:.4f}</p>
            <p><strong>Execution Time:</strong> {best_genome.execution_time:.2f}s</p>
        </div>
    """


def _generate_results_table_html(results) -> str:
    """Generate HTML table for results."""
    if not results:
        return "<p>No results to display.</p>"
    
    table_html = """
        <table>
            <thead>
                <tr>
                    <th>Genome ID</th>
                    <th>Fitness</th>
                    <th>LoRA Rank</th>
                    <th>LoRA Alpha</th>
                    <th>LoRA Dropout</th>
                    <th>CA Complexity</th>
                    <th>CA Intensity</th>
                    <th>Exec Time</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for result in sorted(results, key=lambda r: r.fitness, reverse=True):
        table_html += f"""
                <tr>
                    <td>{result.genome_id}</td>
                    <td>{result.fitness:.4f}</td>
                    <td>{result.lora_config['r']}</td>
                    <td>{result.lora_config['alpha']:.2f}</td>
                    <td>{result.lora_config['dropout']:.3f}</td>
                    <td>{result.ca_features['complexity']:.4f}</td>
                    <td>{result.ca_features['intensity']:.4f}</td>
                    <td>{result.execution_time:.2f}s</td>
                </tr>
        """
    
    table_html += """
            </tbody>
        </table>
    """
    
    return table_html


def _generate_statistics_html(statistics: Dict[str, float]) -> str:
    """Generate HTML for statistics section."""
    stats_html = "<div class='stats-grid'>"
    
    for key, value in statistics.items():
        if key != 'num_genomes':  # Already shown in main stats
            label = key.replace('_', ' ').title()
            stats_html += f"""
                <div class="stat-card">
                    <div class="stat-value">{value:.4f}</div>
                    <div class="stat-label">{label}</div>
                </div>
            """
    
    stats_html += "</div>"
    return stats_html 