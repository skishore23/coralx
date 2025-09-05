"""
Modal infrastructure package.
Contains Modal-specific service implementations.
"""

from .codellama_service import (
    generate_with_codellama_modal,
    generate_baseline_solution_modal,
    generate_evolved_solution_modal
)

from .experiment_service import (
    run_evolution_experiment_modal,
    evaluate_genome_modal,
    load_real_test_cases_modal
)

from .dataset_service import (
    cache_quixbugs_dataset_modal,
    load_quixbugs_problems_modal
)

__all__ = [
    'generate_with_codellama_modal',
    'generate_baseline_solution_modal',
    'generate_evolved_solution_modal',
    'run_evolution_experiment_modal',
    'evaluate_genome_modal',
    'load_real_test_cases_modal',
    'cache_quixbugs_dataset_modal',
    'load_quixbugs_problems_modal'
]
