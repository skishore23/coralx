"""
CORAL Evolution Reporting System
Streamlined reporting for evolution runs and benchmarking
"""
import json
import time
import os
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
from pathlib import Path

class EvolutionReporter:
    """Handles evolution reports, benchmarks, and analysis"""
    
    def __init__(self, output_dir: str = None):
        # Auto-detect Modal environment and use appropriate path
        if output_dir is None:
            if os.path.exists("/reports"):  # Modal volume path
                output_dir = "/reports"
            else:  # Local development
                output_dir = "results"
                
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "evolution").mkdir(exist_ok=True)
        (self.output_dir / "benchmarks").mkdir(exist_ok=True)
        
    def save_evolution_report(self, report_data: Dict, config_type: str = "default") -> str:
        """Save evolution run report with analysis"""
        
        timestamp = int(time.time())
        filename = f"coral_evolution_{config_type}_{timestamp}.json"
        filepath = self.output_dir / "evolution" / filename
        
        # Add metadata
        enhanced_report = {
            **report_data,
            "metadata": {
                "timestamp": timestamp,
                "datetime": datetime.now().isoformat(),
                "config_type": config_type,
                "coral_version": "2.0"
            },
            "analysis": self._analyze_evolution_run(report_data)
        }
        
        # Save JSON report
        with open(filepath, 'w') as f:
            json.dump(enhanced_report, f, indent=2)
        
        # Generate markdown summary
        md_filepath = self._generate_markdown_report(enhanced_report, filepath.stem)
        
        print(f"📊 Evolution report saved: {filepath}")
        print(f"📋 Markdown summary: {md_filepath}")
        
        return str(filepath)
    
    def generate_benchmark_report(self, evolution_reports: List[str]) -> str:
        """Generate benchmark comparison from multiple evolution runs"""
        
        timestamp = int(time.time())
        benchmark_file = self.output_dir / "benchmarks" / f"coral_benchmark_{timestamp}.json"
        
        benchmark_data = {
            "benchmark_info": {
                "timestamp": timestamp,
                "datetime": datetime.now().isoformat(),
                "runs_analyzed": len(evolution_reports)
            },
            "runs": [],
            "comparative_analysis": {}
        }
        
        # Load and analyze each run
        all_runs = []
        for report_path in evolution_reports:
            if os.path.exists(report_path):
                with open(report_path, 'r') as f:
                    run_data = json.load(f)
                    all_runs.append(run_data)
                    benchmark_data["runs"].append({
                        "file": os.path.basename(report_path),
                        "config_type": run_data.get("config_type", "unknown"),
                        "best_fitness": run_data["results"]["best_fitness"],
                        "total_time": run_data["results"]["total_time"],
                        "generations": run_data["results"]["generations_completed"]
                    })
        
        # Comparative analysis
        if all_runs:
            benchmark_data["comparative_analysis"] = self._compare_runs(all_runs)
        
        # Save benchmark
        with open(benchmark_file, 'w') as f:
            json.dump(benchmark_data, f, indent=2)
        
        # Generate markdown benchmark report
        md_filepath = self._generate_benchmark_markdown(benchmark_data, benchmark_file.stem)
        
        print(f"📈 Benchmark report saved: {benchmark_file}")
        print(f"📋 Benchmark summary: {md_filepath}")
        
        return str(benchmark_file)
    
    def _analyze_evolution_run(self, report_data: Dict) -> Dict:
        """Analyze single evolution run for insights"""
        
        analysis = {
            "performance_summary": {},
            "convergence_analysis": {},
            "parameter_effectiveness": {},
            "recommendations": []
        }
        
        # Performance summary
        results = report_data.get("results", {})
        analysis["performance_summary"] = {
            "final_fitness": results.get("best_fitness", 0.0),
            "fitness_achieved": results.get("best_fitness", 0.0) >= 0.8,
            "efficiency_score": self._calculate_efficiency(report_data),
            "completion_rate": results.get("generations_completed", 0) / report_data.get("parameters", {}).get("generations", 20)
        }
        
        # Convergence analysis
        generations = report_data.get("generation_log", [])
        if generations:
            fitness_progression = [g.get("best_fitness", 0.0) for g in generations]
            analysis["convergence_analysis"] = {
                "improvement_rate": self._calculate_improvement_rate(fitness_progression),
                "convergence_generation": self._find_convergence_point(fitness_progression),
                "final_diversity": generations[-1].get("fitness_std", 0.0) if generations else 0.0
            }
        
        # Parameter effectiveness
        params = report_data.get("parameters", {})
        analysis["parameter_effectiveness"] = {
            "survival_threshold": params.get("survival_threshold", 0.1),
            "ca_steps_fast": params.get("fast_eval_steps", 5),
            "ca_steps_full": params.get("full_eval_steps", 25),
            "population_size": params.get("population_size", 50)
        }
        
        # Recommendations
        analysis["recommendations"] = self._generate_recommendations(analysis, report_data)
        
        return analysis
    
    def _generate_markdown_report(self, report_data: Dict, basename: str) -> str:
        """Generate human-readable markdown report"""
        
        md_file = self.output_dir / "evolution" / f"{basename}_report.md"
        
        results = report_data.get("results", {})
        params = report_data.get("parameters", {})
        analysis = report_data.get("analysis", {})
        
        with open(md_file, 'w') as f:
            f.write(f"""# CORAL Evolution Report
## {report_data['metadata']['datetime']}

### Configuration: {report_data.get('config_type', 'default')}

## 🎯 Results Summary
- **Best Fitness Achieved**: {results.get('best_fitness', 0.0):.3f}
- **Total Runtime**: {results.get('total_time', 0.0):.1f} seconds ({results.get('total_time', 0.0)/60:.1f} minutes)
- **Generations Completed**: {results.get('generations_completed', 0)}/{params.get('generations', 20)}
- **Early Termination**: {'Yes' if results.get('terminated_early', False) else 'No'}

## 📊 Performance Analysis
- **Efficiency Score**: {analysis.get('performance_summary', {}).get('efficiency_score', 0.0):.3f}
- **Target Achieved**: {'✅ Yes' if analysis.get('performance_summary', {}).get('fitness_achieved', False) else '❌ No'}
- **Completion Rate**: {analysis.get('performance_summary', {}).get('completion_rate', 0.0):.1%}

## 🔧 Configuration Used
- **Survival Threshold**: {params.get('survival_threshold', 0.1)}
- **Population Size**: {params.get('population_size', 50)}
- **Fast Eval Steps**: {params.get('fast_eval_steps', 5)}
- **Full Eval Steps**: {params.get('full_eval_steps', 25)}
- **Problem Count**: {params.get('problem_count', 8)}
- **Difficulty Range**: {params.get('difficulty_range', 'adaptive')}

## 📈 Evolution Progress
""")
            
            # Add generation details if available
            generations = report_data.get("generation_log", [])
            if generations:
                f.write("| Generation | Best Fitness | Avg Fitness | Survivors | Active Genomes |\n")
                f.write("|-----------|-------------|------------|-----------|---------------|\n")
                for i, gen in enumerate(generations[:10]):  # Show first 10 generations
                    f.write(f"| {gen.get('generation', i)} | {gen.get('best_fitness', 0.0):.3f} | {gen.get('avg_fitness', 0.0):.3f} | {gen.get('survivors', 0)} | {gen.get('active_genomes', 0)} |\n")
                
                if len(generations) > 10:
                    f.write(f"| ... | ... | ... | ... | ... |\n")
                    last_gen = generations[-1]
                    f.write(f"| {last_gen.get('generation', len(generations)-1)} | {last_gen.get('best_fitness', 0.0):.3f} | {last_gen.get('avg_fitness', 0.0):.3f} | {last_gen.get('survivors', 0)} | {last_gen.get('active_genomes', 0)} |\n")
            
            # Add recommendations
            recommendations = analysis.get("recommendations", [])
            if recommendations:
                f.write("\n## 💡 Recommendations\n")
                for i, rec in enumerate(recommendations, 1):
                    f.write(f"{i}. {rec}\n")
            
            f.write(f"\n---\n*Generated by CORAL Evolution Reporter v2.0*\n")
        
        return str(md_file)
    
    def _generate_benchmark_markdown(self, benchmark_data: Dict, basename: str) -> str:
        """Generate benchmark comparison markdown"""
        
        md_file = self.output_dir / "benchmarks" / f"{basename}_report.md"
        
        with open(md_file, 'w') as f:
            f.write(f"""# CORAL Benchmark Report
## {benchmark_data['benchmark_info']['datetime']}

### Runs Analyzed: {benchmark_data['benchmark_info']['runs_analyzed']}

## 📊 Run Comparison
| Config Type | Best Fitness | Runtime (min) | Generations | Efficiency |
|------------|-------------|---------------|------------|------------|
""")
            
            runs = benchmark_data.get("runs", [])
            for run in runs:
                runtime_min = run.get("total_time", 0) / 60
                efficiency = run.get("best_fitness", 0) / max(runtime_min, 1)
                f.write(f"| {run.get('config_type', 'unknown')} | {run.get('best_fitness', 0.0):.3f} | {runtime_min:.1f} | {run.get('generations', 0)} | {efficiency:.3f} |\n")
            
            # Add comparative analysis
            comp_analysis = benchmark_data.get("comparative_analysis", {})
            if comp_analysis:
                f.write(f"""
## 🏆 Best Performance
- **Highest Fitness**: {comp_analysis.get('best_fitness', 0.0):.3f} ({comp_analysis.get('best_config', 'unknown')} config)
- **Fastest Runtime**: {comp_analysis.get('fastest_time', 0.0):.1f} minutes ({comp_analysis.get('fastest_config', 'unknown')} config)
- **Most Efficient**: {comp_analysis.get('most_efficient_config', 'unknown')} config

## 📈 Insights
- **Average Fitness**: {comp_analysis.get('avg_fitness', 0.0):.3f}
- **Fitness Standard Deviation**: {comp_analysis.get('fitness_std', 0.0):.3f}
- **Configuration Recommendation**: {comp_analysis.get('recommended_config', 'default')}
""")
            
            f.write(f"\n---\n*Generated by CORAL Benchmark Reporter v2.0*\n")
        
        return str(md_file)
    
    def _calculate_efficiency(self, report_data: Dict) -> float:
        """Calculate efficiency score (fitness per minute)"""
        results = report_data.get("results", {})
        fitness = results.get("best_fitness", 0.0)
        time_minutes = results.get("total_time", 1) / 60
        return fitness / max(time_minutes, 1)
    
    def _calculate_improvement_rate(self, fitness_progression: List[float]) -> float:
        """Calculate average fitness improvement per generation"""
        if len(fitness_progression) < 2:
            return 0.0
        
        improvements = []
        for i in range(1, len(fitness_progression)):
            improvement = fitness_progression[i] - fitness_progression[i-1]
            improvements.append(max(0, improvement))  # Only count positive improvements
        
        return np.mean(improvements) if improvements else 0.0
    
    def _find_convergence_point(self, fitness_progression: List[float]) -> Optional[int]:
        """Find generation where fitness converged"""
        if len(fitness_progression) < 5:
            return None
            
        # Look for 5 consecutive generations with < 0.01 improvement
        for i in range(4, len(fitness_progression)):
            window = fitness_progression[i-4:i+1]
            if max(window) - min(window) < 0.01:
                return i - 4
        
        return None
    
    def _compare_runs(self, runs: List[Dict]) -> Dict:
        """Compare multiple evolution runs"""
        
        if not runs:
            return {}
        
        fitness_scores = [r["results"]["best_fitness"] for r in runs]
        runtimes = [r["results"]["total_time"] for r in runs]
        configs = [r.get("config_type", "unknown") for r in runs]
        
        # Find best performers
        best_fitness_idx = np.argmax(fitness_scores)
        fastest_idx = np.argmin(runtimes)
        
        # Calculate efficiency (fitness per minute)
        efficiencies = [f / max(t/60, 1) for f, t in zip(fitness_scores, runtimes)]
        most_efficient_idx = np.argmax(efficiencies)
        
        return {
            "best_fitness": fitness_scores[best_fitness_idx],
            "best_config": configs[best_fitness_idx],
            "fastest_time": runtimes[fastest_idx] / 60,
            "fastest_config": configs[fastest_idx],
            "most_efficient_config": configs[most_efficient_idx],
            "avg_fitness": np.mean(fitness_scores),
            "fitness_std": np.std(fitness_scores),
            "recommended_config": configs[most_efficient_idx]
        }
    
    def _generate_recommendations(self, analysis: Dict, report_data: Dict) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        perf = analysis.get("performance_summary", {})
        conv = analysis.get("convergence_analysis", {})
        params = report_data.get("parameters", {})
        
        # Performance-based recommendations
        if perf.get("final_fitness", 0) < 0.5:
            recommendations.append("Consider increasing survival threshold for better selection pressure")
            recommendations.append("Try optimized configuration with more CA evolution steps")
        
        if perf.get("final_fitness", 0) < 0.8 and params.get("fast_eval_steps", 5) < 10:
            recommendations.append("Increase fast evaluation steps for better feature extraction")
        
        # Convergence-based recommendations  
        convergence_gen = conv.get("convergence_generation")
        total_gens = len(report_data.get("generation_log", []))
        
        if convergence_gen and convergence_gen < total_gens * 0.3:
            recommendations.append("Consider increasing population diversity to prevent premature convergence")
        
        if conv.get("final_diversity", 0) < 0.05:
            recommendations.append("Add diversity preservation mechanisms")
        
        # Configuration-specific recommendations
        if params.get("survival_threshold", 0.1) < 0.3:
            recommendations.append("Increase survival threshold to 0.4 for better selection pressure")
        
        if not recommendations:
            recommendations.append("Current configuration performing well - consider scaling up problem complexity")
        
        return recommendations


# Convenience functions for easy import
def save_evolution_report(report_data: Dict, config_type: str = "default") -> str:
    """Quick function to save evolution report"""
    reporter = EvolutionReporter()
    return reporter.save_evolution_report(report_data, config_type)

def generate_benchmark(evolution_reports: List[str]) -> str:
    """Quick function to generate benchmark from multiple reports"""
    reporter = EvolutionReporter()
    return reporter.generate_benchmark_report(evolution_reports) 