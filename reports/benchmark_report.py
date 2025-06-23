"""
CORAL Benchmark Report Generator

Generates structured benchmark reports following the specified format.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class BenchmarkConfig:
    """Benchmark configuration from YAML"""
    name: str
    model_name: str
    dataset_name: str
    task_description: str
    random_seed: int = 42
    strategies: List[Any] = None  # Strategy configurations
    
    def __post_init__(self):
        if self.strategies is None:
            self.strategies = []

@dataclass
class BenchmarkResult:
    """Individual benchmark result"""
    strategy_name: str
    experiment_results: Dict[str, Any]
    execution_time: float
    cost_estimate: float
    success_rate: float
    best_fitness: float
    avg_fitness: float
    problems_solved: int
    total_problems: int

@dataclass
class BenchmarkSummary:
    """Complete benchmark summary"""
    config: BenchmarkConfig  # Now properly typed!
    results: List[BenchmarkResult]
    baseline_result: Optional[BenchmarkResult]
    timestamp: str
    total_cost: float
    total_time: float

class BenchmarkReportGenerator:
    """Generates structured benchmark reports"""
    
    def __init__(self, benchmark_summary: BenchmarkSummary):
        self.summary = benchmark_summary
        self.baseline = benchmark_summary.baseline_result
        
    def generate_markdown_report(self) -> str:
        """Generate markdown report following the specified format"""
        config = self.summary.config
        baseline = self.baseline
        results = self.summary.results
        
        # Find best result
        best_result = max(results, key=lambda r: r.best_fitness)
        
        # Calculate improvement vs baseline
        if baseline and baseline.best_fitness > 0:
            improvement = ((best_result.best_fitness - baseline.best_fitness) / baseline.best_fitness) * 100
        else:
            improvement = 0.0
        
        report = f"""# 📊 Benchmark Overview

**Model / System:** `{config.model_name}`
**Date:** `{self.summary.timestamp}`
**Task / Dataset:** `{config.dataset_name}`

---

## 1. Executive Summary

* **Headline Metric:** `{best_result.best_fitness:.4f}` (+`{improvement:.1f}%` vs baseline)
* **Key Take‑away:** `{self._generate_key_takeaway(best_result, baseline)}`

---

## 2. Experimental Setup

| Item                 | Details                         |
| -------------------- | ------------------------------- |
| Base Model           | `{config.model_name}`                  |
| Tuning Method        | `CORAL (Cellular Automata + LoRA)`                 |
| Training Budget      | `{self.summary.total_time:.1f}s` / `${self.summary.total_cost:.2f}` |
| Hardware             | `{self._get_hardware_summary()}`                |
| Evaluation Metric(s) | `Fitness Score, Success Rate, Problems Solved`                 |

---

## 3. Results

### 3.1 Aggregate Scores

| Method              | Score              | Δ vs Baseline   |
| ------------------- | ------------------ | --------------- |"""
        
        # Add results table
        for result in sorted(results, key=lambda r: r.best_fitness, reverse=True):
            if result == best_result:
                name = f"**`{result.strategy_name}`**"
                score = f"**`{result.best_fitness:.4f}`**"
            else:
                name = f"`{result.strategy_name}`"
                score = f"`{result.best_fitness:.4f}`"
            
            if baseline and baseline.best_fitness > 0:
                delta = ((result.best_fitness - baseline.best_fitness) / baseline.best_fitness) * 100
                delta_str = f"**`+{delta:.1f}%`**" if result == best_result else f"`{delta:+.1f}%`"
            else:
                delta_str = "–"
            
            report += f"\n| {name:19} | {score:18} | {delta_str:15} |"
        
        if baseline:
            report += f"\n| Baseline            | `{baseline.best_fitness:.4f}`      | –               |"
        
        # Problem-level breakdown
        report += f"""

### 3.2 Strategy Details

| Strategy          | Success Rate | Avg Fitness | Problems Solved | Cost Estimate |
| ----------------- | ------------ | ----------- | --------------- | ------------- |"""
        
        for result in results:
            success_pct = result.success_rate * 100
            solved_fraction = f"{result.problems_solved}/{result.total_problems}"
            report += f"\n| `{result.strategy_name:15}` | {success_pct:10.1f}% | {result.avg_fitness:11.4f} | {solved_fraction:15} | ${result.cost_estimate:11.2f} |"
        
        report += f"""

---

## 4. Analysis & Insights

* **Strengths:** {self._analyze_strengths()}
* **Weaknesses / Trade‑offs:** {self._analyze_weaknesses()}
* **Cost Efficiency:** {self._analyze_cost_efficiency()}

---

## 5. Reproducibility Checklist

* **Code Repository:** `{self._get_repo_url()}`
* **Random Seed(s):** `{self.summary.config.random_seed}`
* **Data Splits:** `{self._get_data_splits()}`
* **Config File:** `{self._get_config_path()}`
* **Environment:** `{self._get_environment_info()}`

---

## 6. Future Work

{self._generate_future_work()}

---

## 7. Raw Results

```json
{self._get_raw_results_json()}
```
"""
        
        return report
    
    def _generate_key_takeaway(self, best_result: BenchmarkResult, baseline: Optional[BenchmarkResult]) -> str:
        """Generate one-sentence summary"""
        if baseline:
            if best_result.best_fitness > baseline.best_fitness:
                return f"{best_result.strategy_name} strategy achieves superior performance with {best_result.problems_solved} problems solved"
            else:
                return f"Baseline strategy remains competitive despite {best_result.strategy_name} optimizations"
        else:
            return f"{best_result.strategy_name} strategy demonstrates strong performance with {best_result.success_rate*100:.1f}% success rate"
    
    def _get_hardware_summary(self) -> str:
        """Get hardware summary from configurations"""
        gpus = set()
        for result in self.summary.results:
            # Extract GPU from result or config
            try:
                if hasattr(self.summary.config.strategies[0], 'name'):
                    # Object format
                    strategy = next(s for s in self.summary.config.strategies if s.name == result.strategy_name)
                    gpus.add(strategy.config.runtime.gpu)
                else:
                    # Dict format - fallback
                    strategy = next(s for s in self.summary.config.strategies if s.get('name') == result.strategy_name)
                    gpus.add(strategy.get('config', {}).get('runtime', {}).get('gpu', 'A10G'))
            except (StopIteration, AttributeError):
                # Default fallback
                gpus.add('A10G')
        return ", ".join(sorted(gpus))
    
    def _analyze_strengths(self) -> str:
        """Analyze benchmark strengths"""
        best_result = max(self.summary.results, key=lambda r: r.best_fitness)
        most_reliable = max(self.summary.results, key=lambda r: r.success_rate)
        most_efficient = min(self.summary.results, key=lambda r: r.cost_estimate)
        
        strengths = []
        strengths.append(f"`{best_result.strategy_name}` achieves highest fitness ({best_result.best_fitness:.4f})")
        if most_reliable != best_result:
            strengths.append(f"`{most_reliable.strategy_name}` provides highest reliability ({most_reliable.success_rate*100:.1f}%)")
        if most_efficient != best_result:
            strengths.append(f"`{most_efficient.strategy_name}` offers best cost efficiency (${most_efficient.cost_estimate:.2f})")
        
        return ", ".join(strengths[:2])  # Limit to 2 main strengths
    
    def _analyze_weaknesses(self) -> str:
        """Analyze benchmark weaknesses"""
        worst_result = min(self.summary.results, key=lambda r: r.best_fitness)
        least_reliable = min(self.summary.results, key=lambda r: r.success_rate)
        most_expensive = max(self.summary.results, key=lambda r: r.cost_estimate)
        
        weaknesses = []
        if worst_result.best_fitness < max(r.best_fitness for r in self.summary.results) * 0.8:
            weaknesses.append(f"`{worst_result.strategy_name}` shows lower performance ({worst_result.best_fitness:.4f})")
        if least_reliable.success_rate < 0.9:
            weaknesses.append(f"`{least_reliable.strategy_name}` has reliability concerns ({least_reliable.success_rate*100:.1f}%)")
        if most_expensive.cost_estimate > min(r.cost_estimate for r in self.summary.results) * 2:
            weaknesses.append(f"`{most_expensive.strategy_name}` incurs higher costs (${most_expensive.cost_estimate:.2f})")
        
        return ", ".join(weaknesses[:2]) if weaknesses else "No significant weaknesses identified"
    
    def _analyze_cost_efficiency(self) -> str:
        """Analyze cost efficiency"""
        # Calculate performance per dollar
        efficiency_scores = []
        for result in self.summary.results:
            if result.cost_estimate > 0:
                efficiency = result.best_fitness / result.cost_estimate
                efficiency_scores.append((result.strategy_name, efficiency))
        
        if efficiency_scores:
            best_efficiency = max(efficiency_scores, key=lambda x: x[1])
            return f"`{best_efficiency[0]}` provides best performance per dollar ({best_efficiency[1]:.3f} fitness/$)"
        else:
            return "Cost efficiency analysis unavailable"
    
    def _get_repo_url(self) -> str:
        """Get repository URL"""
        return "https://github.com/user/coral-evolution"
    
    def _get_data_splits(self) -> str:
        """Get data split description"""
        return "QuixBugs train/test split (standard)"
    
    def _get_config_path(self) -> str:
        """Get configuration file path"""
        return f"configs/{self.summary.config.name}.yaml"
    
    def _get_environment_info(self) -> str:
        """Get environment information"""
        return "Python 3.11, Modal, PyTorch 2.1.0, Transformers 4.46.0"
    
    def _generate_future_work(self) -> str:
        """Generate future work suggestions"""
        suggestions = [
            "1. Explore larger population sizes and longer evolution runs",
            "2. Investigate hybrid CA rule combinations for feature generation",
            "3. Test additional LoRA target modules and rank configurations", 
            "4. Benchmark on broader code repair datasets (HumanEval, MBPP)",
            "5. Implement adaptive batch sizing based on GPU memory utilization"
        ]
        return "\n".join(suggestions)
    
    def _get_raw_results_json(self) -> str:
        """Get raw results as formatted JSON"""
        raw_data = {
            "benchmark_config": self.summary.config.__dict__,
            "results": [
                {
                    "strategy": result.strategy_name,
                    "best_fitness": result.best_fitness,
                    "avg_fitness": result.avg_fitness,
                    "success_rate": result.success_rate,
                    "execution_time": result.execution_time,
                    "cost_estimate": result.cost_estimate,
                    "problems_solved": result.problems_solved
                }
                for result in self.summary.results
            ],
            "summary": {
                "total_cost": self.summary.total_cost,
                "total_time": self.summary.total_time,
                "timestamp": self.summary.timestamp
            }
        }
        return json.dumps(raw_data, indent=2)
    
    def generate_json_report(self) -> Dict[str, Any]:
        """Generate JSON report"""
        return {
            "benchmark_overview": {
                "model": self.summary.config.model_name,
                "date": self.summary.timestamp,
                "dataset": self.summary.config.dataset_name,
                "task": self.summary.config.task_description
            },
            "executive_summary": {
                "best_score": max(r.best_fitness for r in self.summary.results),
                "best_strategy": max(self.summary.results, key=lambda r: r.best_fitness).strategy_name,
                "improvement_vs_baseline": self._calculate_improvement(),
                "key_takeaway": self._generate_key_takeaway(
                    max(self.summary.results, key=lambda r: r.best_fitness),
                    self.baseline
                )
            },
            "experimental_setup": {
                "base_model": self.summary.config.model_name,
                "tuning_method": "CORAL (Cellular Automata + LoRA)",
                "training_budget": {
                    "time_seconds": self.summary.total_time,
                    "cost_dollars": self.summary.total_cost
                },
                "hardware": self._get_hardware_summary(),
                "evaluation_metrics": ["fitness_score", "success_rate", "problems_solved"]
            },
            "results": [
                {
                    "strategy": result.strategy_name,
                    "score": result.best_fitness,
                    "delta_vs_baseline": self._calculate_delta_vs_baseline(result),
                    "success_rate": result.success_rate,
                    "avg_fitness": result.avg_fitness,
                    "problems_solved": result.problems_solved,
                    "cost_estimate": result.cost_estimate,
                    "execution_time": result.execution_time
                }
                for result in self.summary.results
            ],
            "analysis": {
                "strengths": self._analyze_strengths(),
                "weaknesses": self._analyze_weaknesses(),
                "cost_efficiency": self._analyze_cost_efficiency()
            },
            "reproducibility": {
                "random_seed": self.summary.config.random_seed,
                "config_file": self._get_config_path(),
                "environment": self._get_environment_info()
            },
            "raw_results": json.loads(self._get_raw_results_json())
        }
    
    def _calculate_improvement(self) -> float:
        """Calculate improvement vs baseline"""
        if not self.baseline:
            return 0.0
        best_result = max(self.summary.results, key=lambda r: r.best_fitness)
        if self.baseline.best_fitness > 0:
            return ((best_result.best_fitness - self.baseline.best_fitness) / self.baseline.best_fitness) * 100
        return 0.0
    
    def _calculate_delta_vs_baseline(self, result: BenchmarkResult) -> float:
        """Calculate delta vs baseline for specific result"""
        if not self.baseline or self.baseline.best_fitness == 0:
            return 0.0
        return ((result.best_fitness - self.baseline.best_fitness) / self.baseline.best_fitness) * 100
    
    def save_reports(self, output_dir: str = "results/benchmarks") -> Dict[str, str]:
        """Save both markdown and JSON reports"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{self.summary.config.name}_{timestamp}"
        
        # Save markdown report
        markdown_content = self.generate_markdown_report()
        markdown_path = os.path.join(output_dir, f"{base_filename}.md")
        with open(markdown_path, 'w') as f:
            f.write(markdown_content)
        
        # Save JSON report
        json_content = self.generate_json_report()
        json_path = os.path.join(output_dir, f"{base_filename}.json")
        with open(json_path, 'w') as f:
            json.dump(json_content, f, indent=2)
        
        return {
            "markdown": markdown_path,
            "json": json_path
        } 