# CoralX - Functional CORAL Evolution System
# Enhanced with CORAL-X features: clone-cache, threshold gates, multi-objective optimization

"""
CORAL-X: Real-world Evolutionary Algorithm System
Combines Cellular Automata with LoRA fine-tuning
"""

__version__ = "0.1.0"

# Make key components available at package level
try:
    from .coral.application.evolution_engine import EvolutionEngine, CoralConfig
    from .coral.config.loader import load_config, create_default_config
    from .coral.domain.genome import Genome, MultiObjectiveScores
    from .coral.domain.neat import Population
    from .infra.modal_executor import LocalExecutor, ThreadExecutor
    from .plugins.quixbugs_codellama.plugin import QuixBugsCodeLlamaRealPlugin
    
    __all__ = [
        'EvolutionEngine', 'CoralConfig',
        'load_config', 'create_default_config',
        'Genome', 'MultiObjectiveScores', 'Population',
        'LocalExecutor', 'ThreadExecutor',
        'QuixBugsCodeLlamaRealPlugin'
    ]
except ImportError as e:
    # Package can still be imported even if some components fail
    print(f"⚠️  Some CORAL-X components failed to import: {e}")
    __all__ = []

__author__ = "CoralX Team" 