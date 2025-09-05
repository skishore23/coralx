#!/usr/bin/env python3
"""
M1 Test Script - End-to-End Tiny Run
Tests the complete M1 pipeline with minimal configuration
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.common.config_loader import load_config
from core.application.evolution_orchestrator import EvolutionOrchestrator
from core.application.services import create_evolution_services
from core.services.jsonl_logger import JSONLLogger
from core.services.reproducibility import ReproducibilityManager
from plugins.quixbugs_mini.plugin import QuixBugsMiniPlugin
from plugins.fakenews_mini.plugin import FakeNewsMiniPlugin


def test_m1_pipeline():
    """Test the complete M1 pipeline."""
    print("🧬 CORAL-X M1 Test - End-to-End Tiny Run")
    print("=" * 50)

    # Load M1 configuration
    config_path = project_root / "config" / "examples" / "tiny_run.yaml"
    print(f"Loading M1 config: {config_path}")

    try:
        config = load_config(config_path)
        print("✅ Config loaded successfully")
        print(f"   • Generations: {config.execution.generations}")
        print(f"   • Population size: {config.execution.population_size}")
        print(f"   • Selection mode: {config.execution.selection_mode}")
        print(f"   • Executor: {config.infra.executor}")
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        assert False, f"Config loading failed: {e}"

    # Initialize reproducibility manager
    repro_manager = ReproducibilityManager(Path("./artifacts/tiny_run"))
    print("✅ Reproducibility manager initialized")

    # Create repro.lock
    try:
        repro_info = repro_manager.create_repro_lock(
            experiment_id="m1_test_001",
            seed=config.seed,
            config=config.model_dump(),
            datasets={"quixbugs_mini": "3_problems_mock"},
            checkpoints={"base_model": "mock_model_for_m1"}
        )
        print(f"✅ repro.lock created: {repro_info.experiment_id}")
    except Exception as e:
        print(f"❌ repro.lock creation failed: {e}")
        assert False, f"repro.lock creation failed: {e}"

    # Initialize JSONL logger
    jsonl_logger = JSONLLogger(Path("./artifacts/tiny_run/evolution.jsonl"))
    print("✅ JSONL logger initialized")

    # Test plugin loading
    print("\n🔌 Testing Plugin Loading")
    print("-" * 30)

    try:
        # Test QuixBugs Mini plugin
        quixbugs_plugin = QuixBugsMiniPlugin(config.model_dump())
        print("✅ QuixBugs Mini plugin loaded")

        # Test dataset
        dataset = quixbugs_plugin.dataset()
        problems = list(dataset.problems())
        print(f"   • Dataset problems: {len(problems)}")
        for problem in problems:
            print(f"     - {problem['name']}: {problem['expected_behavior']}")

        # Test model factory
        model_factory = quixbugs_plugin.model_factory()
        print("✅ Model factory created")

        # Test fitness function
        fitness_fn = quixbugs_plugin.fitness_fn()
        print("✅ Fitness function created")

    except Exception as e:
        print(f"❌ Plugin loading failed: {e}")
        assert False, f"Plugin loading failed: {e}"

    # Test FakeNews Mini plugin
    try:
        fakenews_plugin = FakeNewsMiniPlugin(config.model_dump())
        print("✅ FakeNews Mini plugin loaded")

        # Test dataset
        fakenews_dataset = fakenews_plugin.dataset()
        fakenews_problems = list(fakenews_dataset.problems())
        print(f"   • FakeNews samples: {len(fakenews_problems)}")

    except Exception as e:
        print(f"❌ FakeNews plugin loading failed: {e}")
        return False

    # Test evolution services
    print("\n⚙️ Testing Evolution Services")
    print("-" * 30)

    try:
        services = create_evolution_services(config)
        print("✅ Evolution services created")
        print(f"   • Population manager: {type(services.population_manager).__name__}")
        print(f"   • Genetic operations: {type(services.genetic_operations).__name__}")
        print(f"   • Progress tracker: {type(services.progress_tracker).__name__}")
        print(f"   • Executor: {type(services.executor).__name__}")

    except Exception as e:
        print(f"❌ Evolution services creation failed: {e}")
        assert False, f"Evolution services creation failed: {e}"

    # Test evolution orchestrator
    print("\n🎯 Testing Evolution Orchestrator")
    print("-" * 30)

    try:
        # Create EvolutionServices object
        from core.application.evolution_orchestrator import EvolutionServices
        evolution_services = EvolutionServices(
            population_manager=services.population_manager,
            genetic_operations=services.genetic_operations,
            progress_tracker=services.progress_tracker,
            fitness_fn=services.fitness_fn,
            executor=services.executor,
            config=config
        )

        orchestrator = EvolutionOrchestrator(evolution_services)
        print("✅ Evolution orchestrator created")

        # Test population initialization (async method)
        print("   Testing population initialization...")
        import asyncio
        initial_population = asyncio.run(orchestrator._initialize_population())
        print(f"   ✅ Initial population created: {initial_population.size()} genomes")

        # Test population validation
        services.population_manager.validate_population(initial_population)
        print("   ✅ Population validation passed")

        # Test diversity metrics
        diversity = services.population_manager.calculate_diversity_metrics(initial_population)
        print(f"   ✅ Diversity metrics calculated: {diversity}")

    except Exception as e:
        print(f"❌ Evolution orchestrator test failed: {e}")
        assert False, f"Evolution orchestrator test failed: {e}"

    # Test tournament selection
    print("\n🏆 Testing Tournament Selection")
    print("-" * 30)

    try:
        from core.domain.neat import tournament_select
        from random import Random

        # Create test population with mock fitness scores
        test_genomes = []
        for i in range(8):
            from core.domain.genome import Genome
            from core.domain.ca import CASeed
            from core.domain.mapping import LoRAConfig
            import numpy as np

            # Create mock genome
            seed = CASeed(
                grid=np.random.randint(0, 2, (8, 8)),
                rule=30 + i,
                steps=10
            )
            lora_cfg = LoRAConfig(
                r=4 + i,
                alpha=8.0 + i,
                dropout=0.1,
                target_modules=("q_proj", "v_proj")
            )
            genome = Genome(
                seed=seed,
                lora_cfg=lora_cfg,
                id=f"test_genome_{i}",
                fitness=0.5 + i * 0.1  # Mock fitness scores
            )
            test_genomes.append(genome)

        from core.domain.neat import Population
        test_population = Population(tuple(test_genomes))

        # Test tournament selection
        rng = Random(42)  # Deterministic for testing
        survivors = tournament_select(test_population, k=4, tournament_size=3, rng=rng)
        print(f"   ✅ Tournament selection completed: {survivors.size()} survivors")

        # Verify deterministic behavior
        survivors2 = tournament_select(test_population, k=4, tournament_size=3, rng=Random(42))
        assert survivors.genomes == survivors2.genomes, "Tournament selection not deterministic!"
        print("   ✅ Tournament selection is deterministic")

    except Exception as e:
        print(f"❌ Tournament selection test failed: {e}")
        assert False, f"Tournament selection test failed: {e}"

    # Test JSONL logging
    print("\n📝 Testing JSONL Logging")
    print("-" * 30)

    try:
        # Log experiment start
        jsonl_logger.log_experiment_start(config.model_dump())
        print("   ✅ Experiment start logged")

        # Log some test candidates
        for i, genome in enumerate(test_genomes[:3]):
            jsonl_logger.log_candidate(
                genome=genome,
                generation=0,
                evaluation_time=0.1 + i * 0.05,
                additional_data={"test_run": True}
            )
        print("   ✅ Test candidates logged")

        # Log generation summary
        jsonl_logger.log_generation_summary(
            generation=0,
            population_size=8,
            best_fitness=0.8,
            avg_fitness=0.6,
            diversity_metrics=diversity,
            selection_info={"method": "tournament", "tournament_size": 3}
        )
        print("   ✅ Generation summary logged")

        # Get log stats
        stats = jsonl_logger.get_log_stats()
        print(f"   ✅ Log stats: {stats}")

    except Exception as e:
        print(f"❌ JSONL logging test failed: {e}")
        assert False, f"JSONL logging test failed: {e}"

    print("\n🎉 M1 Test Pipeline Completed Successfully!")
    print("=" * 50)
    print("✅ All components working correctly")
    print("✅ Ready for M1 end-to-end run")
    print("✅ Reproducibility ensured with repro.lock")
    print("✅ JSONL logging enabled")
    print("✅ Tournament selection implemented")
    print("✅ Mini plugins working")


if __name__ == "__main__":
    test_m1_pipeline()
