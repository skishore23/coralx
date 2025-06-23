"""
Application layer for experiment orchestration.
Coordinates between domain logic and infrastructure services.
"""
from typing import Dict, Any, List
from coral.domain.experiment import ExperimentConfig, ExperimentResults, create_experiment_config, create_initial_population
from coral.ports.interfaces import Executor, DatasetProvider, ModelRunner, FitnessFn
from coral.application.evolution_engine import EvolutionEngine, CoralConfig
import time


class ExperimentOrchestrator:
    """
    Application service for orchestrating experiments.
    Coordinates between domain logic and infrastructure.
    """
    
    def __init__(self, 
                 executor: Executor,
                 fitness_fn: FitnessFn,
                 model_factory: callable,
                 dataset: DatasetProvider):
        self.executor = executor
        self.fitness_fn = fitness_fn
        self.model_factory = model_factory
        self.dataset = dataset
    
    def run_experiment(self, experiment_config: Dict[str, Any]) -> ExperimentResults:
        """
        Run a complete evolution experiment.
        Pure application logic - no infrastructure concerns.
        """
        print(f"🚀 Starting CORAL evolution experiment")
        start_time = time.time()
        
        try:
            # Create structured configs using domain logic
            exp_config = create_experiment_config(experiment_config)
            coral_config = self._create_coral_config(experiment_config)
            
            # Extract experiment name - FAIL-FAST if missing
            if 'experiment' not in experiment_config or 'name' not in experiment_config['experiment']:
                raise ValueError("FAIL-FAST: Experiment name missing from configuration")
            
            experiment_name = experiment_config['experiment']['name']
            print(f"📋 Experiment: {experiment_name}")
            print(f"🧬 Population: {exp_config.population_size}")
            print(f"🔄 Generations: {exp_config.generations}")
            
            # Create evolution engine
            engine = EvolutionEngine(
                cfg=coral_config,
                fitness_fn=self.fitness_fn,
                executor=self.executor,
                model_factory=self.model_factory,
                dataset=self.dataset
            )
            
            # Create initial population using domain logic with balanced cache-clone strategy
            diversity_strength = 0.4  # Target 3-8x cache efficiency
            print(f"🎯 Using diversity strength: {diversity_strength:.1f} (balanced cache-clone strategy)")
            init_pop = create_initial_population(exp_config, diversity_strength)
            print(f"✅ Created initial population: {init_pop.size()} genomes")
            
            # Run evolution
            print(f"🚀 Starting {exp_config.generations} generations...")
            winners = engine.run(init_pop)
            
            end_time = time.time()
            
            # Create results using domain logic
            from coral.domain.experiment import create_experiment_result
            results = create_experiment_result(
                population=winners,
                start_time=start_time,
                end_time=end_time,
                generations_completed=exp_config.generations,
                success=True
            )
            
            print(f"🏆 Evolution completed in {results.experiment_time:.2f}s")
            print(f"📊 Final population: {results.final_population.size()} genomes")
            print(f"🎯 Best fitness: {results.best_fitness:.4f}")
            
            return results
            
        except Exception as e:
            end_time = time.time()
            print(f"❌ Evolution experiment failed: {e}")
            
            from coral.domain.experiment import create_experiment_result
            from coral.domain.neat import Population
            
            return create_experiment_result(
                population=Population(tuple()),
                start_time=start_time,
                end_time=end_time,
                generations_completed=0,
                success=False,
                error_message=str(e)
            )
    
    def _create_coral_config(self, experiment_config: Dict[str, Any]) -> CoralConfig:
        """Create CoralConfig from experiment configuration."""
        from coral.config.loader import create_config_from_dict
        return create_config_from_dict(experiment_config)


class BenchmarkOrchestrator:
    """
    Application service for benchmarking experiments.
    Coordinates baseline and evolved model comparisons.
    """
    
    def __init__(self, 
                 executor: Executor,
                 dataset: DatasetProvider):
        self.executor = executor
        self.dataset = dataset
    
    def run_baseline_benchmark(self, problems: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run baseline benchmark using vanilla models."""
        print(f"📊 Running baseline benchmark on {len(problems)} problems...")
        
        baseline_scores = []
        start_time = time.time()
        
        for i, problem in enumerate(problems, 1):
            print(f"🔸 Baseline problem {i}/{len(problems)}: {problem['name']}")
            
            try:
                # Generate baseline solution
                baseline_code = self._generate_baseline_solution(problem)
                
                # Load real test cases
                test_cases = self._load_test_cases(problem['name'])
                
                # Evaluate with domain logic
                from coral.domain.quixbugs_evaluation import evaluate_quixbugs_code
                result = evaluate_quixbugs_code(baseline_code, problem, test_cases)
                
                baseline_scores.append({
                    "problem": problem['name'],
                    "bugfix": result.bugfix,
                    "style": result.style,
                    "security": result.security,
                    "runtime": result.runtime,
                    "overall": (result.bugfix + result.style + result.security + result.runtime) / 4.0
                })
                
            except Exception as e:
                print(f"❌ Baseline failed on {problem['name']}: {e}")
                baseline_scores.append({
                    "problem": problem['name'],
                    "bugfix": 0.0, "style": 0.0, "security": 0.0, "runtime": 0.0, "overall": 0.0,
                    "error": str(e)
                })
        
        benchmark_time = time.time() - start_time
        avg_scores = self._calculate_average_scores(baseline_scores)
        
        print(f"✅ Baseline benchmark completed in {benchmark_time:.2f}s")
        print(f"📈 Average baseline scores: {avg_scores}")
        
        return {
            "type": "baseline",
            "problem_count": len(problems),
            "individual_scores": baseline_scores,
            "average_scores": avg_scores,
            "benchmark_time": benchmark_time
        }
    
    def run_evolved_benchmark(self, problems: List[Dict[str, Any]], experiment_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run evolved model benchmark."""
        print(f"🧬 Running evolved benchmark on {len(problems)} problems...")
        
        evolved_scores = []
        start_time = time.time()
        
        for i, problem in enumerate(problems, 1):
            print(f"🔸 Evolved problem {i}/{len(problems)}: {problem['name']}")
            
            try:
                # Generate evolved solution
                evolved_code = self._generate_evolved_solution(problem, experiment_config)
                
                # Load real test cases
                test_cases = self._load_test_cases(problem['name'])
                
                # Evaluate with domain logic
                from coral.domain.quixbugs_evaluation import evaluate_quixbugs_code
                result = evaluate_quixbugs_code(evolved_code, problem, test_cases)
                
                evolved_scores.append({
                    "problem": problem['name'],
                    "bugfix": result.bugfix,
                    "style": result.style,
                    "security": result.security,
                    "runtime": result.runtime,
                    "overall": (result.bugfix + result.style + result.security + result.runtime) / 4.0
                })
                
            except Exception as e:
                print(f"❌ Evolved failed on {problem['name']}: {e}")
                evolved_scores.append({
                    "problem": problem['name'],
                    "bugfix": 0.0, "style": 0.0, "security": 0.0, "runtime": 0.0, "overall": 0.0,
                    "error": str(e)
                })
        
        benchmark_time = time.time() - start_time
        avg_scores = self._calculate_average_scores(evolved_scores)
        
        print(f"✅ Evolved benchmark completed in {benchmark_time:.2f}s")
        print(f"📈 Average evolved scores: {avg_scores}")
        
        return {
            "type": "evolved",
            "problem_count": len(problems),
            "individual_scores": evolved_scores,
            "average_scores": avg_scores,
            "benchmark_time": benchmark_time,
            "experiment_config": experiment_config
        }
    
    def _generate_baseline_solution(self, problem: Dict[str, Any]) -> str:
        """Generate baseline solution using infrastructure services."""
        # This would call the appropriate infrastructure service
        raise NotImplementedError("Must be implemented by infrastructure layer")
    
    def _generate_evolved_solution(self, problem: Dict[str, Any], experiment_config: Dict[str, Any]) -> str:
        """Generate evolved solution using infrastructure services."""
        # This would call the appropriate infrastructure service
        raise NotImplementedError("Must be implemented by infrastructure layer")
    
    def _load_test_cases(self, problem_name: str) -> str:
        """Load test cases using infrastructure services."""
        # This would call the appropriate infrastructure service
        raise NotImplementedError("Must be implemented by infrastructure layer")
    
    def _calculate_average_scores(self, scores: List[Dict[str, Any]]) -> Dict[str, float]:
        """Pure function to calculate average scores."""
        if not scores:
            return {"bugfix": 0.0, "style": 0.0, "security": 0.0, "runtime": 0.0, "overall": 0.0}
        
        totals = {"bugfix": 0.0, "style": 0.0, "security": 0.0, "runtime": 0.0, "overall": 0.0}
        count = 0
        
        for score in scores:
            if "error" not in score:  # Skip failed evaluations
                for key in totals.keys():
                    totals[key] += score.get(key, 0.0)
                count += 1
        
        if count == 0:
            return totals
        
        return {key: total / count for key, total in totals.items()} 