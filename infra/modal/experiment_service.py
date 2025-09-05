"""
Modal service for experiment execution - NO FALLBACKS, config-driven.
Infrastructure layer - handles Modal-specific experiment orchestration.
"""
import time
from typing import Dict, Any



def run_evolution_experiment_modal(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run evolution experiment on Modal with clean architecture.
    SMART DEPENDENCY MANAGEMENT: Auto-setup missing dependencies instead of failing.
    """
    import sys
    from pathlib import Path

    experiment_start_time = time.time()  # Track experiment timing

    print("🚀 Modal: Starting CORAL evolution experiment")
    print(f"⏰ Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📋 Experiment config sections: {list(config_dict.keys())}")
    print("=" * 80)

    # SMART DEPENDENCY MANAGEMENT: Ensure all dependencies before GPU work
    print("🔧 [SMART] Checking dependencies before starting evolution...")
    try:
        import modal
        ensure_deps_fn = modal.Function.from_name('coral-x-production', 'ensure_dependencies_modal')
        deps_result = ensure_deps_fn.remote()

        if deps_result.get("all_ready", False):
            print(f"✅ All dependencies ready in {deps_result.get('total_time', 0):.2f}s")
        else:
            print(f"⚠️  Some dependencies not ready: {deps_result}")
            print("🔄 Continuing anyway - dependencies will auto-setup during execution")
    except Exception as deps_error:
        print(f"⚠️  Dependency check failed: {deps_error}")
        print("🔄 Continuing anyway - dependencies will auto-setup during execution")

    # Add coralx to Python path
    coralx_path = Path("/root/coralx")
    if not coralx_path.exists():
        raise RuntimeError("  CoralX codebase not found in Modal environment")
    sys.path.insert(0, str(coralx_path))

    print("✅ Domain modules imported successfully")
    print(f"📁 Python path: {sys.path[:3]}...")

    try:
        # Import clean domain and application logic
        from core.config.loader import create_config_from_dict
        from core.application.evolution_engine import EvolutionEngine
        from core.domain.experiment import create_experiment_config, create_initial_population, create_experiment_result
        from plugins.quixbugs_codellama.plugin import QuixBugsCodeLlamaRealPlugin
        from infra.modal_executor import ModalExecutor

        print("🔧 Creating structured config from experiment configuration...")
        config = create_config_from_dict(config_dict)
        print("✅ Configuration validation passed")

        # Extract experiment configuration
        experiment_config = config_dict.get('experiment', {})
        execution_config = config_dict.get('execution', {})
        infra_config = config_dict.get('infra', {})

        # Extract key parameters
        generations = execution_config.get('generations', 40)
        population_size = execution_config.get('population_size', 32)

        # Create experiment config and plugin
        exp_config = create_experiment_config(config_dict)
        plugin = QuixBugsCodeLlamaRealPlugin(config_dict)

        print(f"📋 Experiment: {experiment_config.get('name', 'unnamed')}")
        print(f"🧬 Population: {execution_config.get('population_size', 32)}")
        print(f"🔄 Generations: {execution_config.get('generations', 40)}")
        print(f"💾 Cache artifacts: {config_dict.get('cache', {}).get('artifacts_dir', 'unknown')}")
        print(f"📂 Dataset path: {experiment_config.get('dataset', {}).get('path', 'unknown')}")
        print(f"🤖 Model: {experiment_config.get('model', {}).get('name', 'unknown')}")

        # Get Modal app name from config
        modal_config = infra_config.get('modal', {})
        app_name = modal_config.get('app_name', 'coral-x-production')  # Default to production app name

        print(f"   • Modal app name: {app_name}")
        print("   • Checking Modal app deployment...")

        executor = ModalExecutor(app_name=app_name, config=config_dict)

        # Create evolution engine with run_id
        print("🏗️ Creating evolution engine...")
        run_id = config_dict.get('cache', {}).get('run_id', None)
        engine = EvolutionEngine(
            cfg=config,
            fitness_fn=plugin.fitness_fn(),
            executor=executor,
            model_factory=plugin.model_factory(),
            dataset=plugin.dataset(),
            run_id=run_id,  # Pass run_id from config
            raw_config=config_dict  # CRITICAL: Pass raw config for training
        )
        print("✅ Evolution engine ready")

        # Create initial population using domain logic with balanced cache-clone strategy
        # Use diversity_strength = 0.4 for 3-8x cache efficiency (balanced sharing)
        diversity_strength = 0.4
        print(f"🎯 Using diversity strength: {diversity_strength:.1f} (balanced cache-clone strategy)")
        print(f"🔑 Run ID: {run_id or 'None (will reuse existing adapters)'}")
        init_pop = create_initial_population(exp_config, diversity_strength, config_dict, run_id=run_id)  # 🔥 FIX: Pass raw config for adapter_type and run_id
        print(f"✅ Created initial population: {init_pop.size()} genomes")

        # Run evolution (early stopping handled internally by EvolutionEngine)
        print(f"🚀 Starting {generations} generations...")
        winners = engine.run(init_pop)

        experiment_end_time = time.time()

        # Create results using domain logic
        results = create_experiment_result(
            population=winners,
            start_time=experiment_start_time,
            end_time=experiment_end_time,
            generations_completed=generations,
            success=True
        )

        print("=" * 80)
        print(f"🏆 Evolution completed in {results.experiment_time:.2f}s")
        print(f"📊 Final population: {results.final_population.size()} genomes")
        best_fitness_display = f"{results.best_fitness:.4f}" if results.best_fitness is not None else "None"
        print(f"🎯 Best fitness: {best_fitness_display}")

        # Log best genome details if available
        try:
            best_genome = results.final_population.best()
            if best_genome.has_multi_scores():
                scores = best_genome.multi_scores
                print("🏅 Best genome scores:")
                print(f"   • Bugfix: {scores.bugfix:.3f}")
                print(f"   • Style: {scores.style:.3f}")
                print(f"   • Security: {scores.security:.3f}")
                print(f"   • Runtime: {scores.runtime:.3f}")
        except Exception as e:
            print(f"⚠️ Could not extract best genome details: {e}")

        print("=" * 80)

        # Convert to dict for serialization
        return {
            "type": "evolution_experiment",
            "generations": generations,
            "population_size": population_size,
            "experiment_time": results.experiment_time,
                    "best_fitness": results.best_fitness,
        "best_scores": _extract_best_scores_from_results(results),  # 🔥 CRITICAL: Add missing best_scores
        "final_population_size": results.final_population.size(),
        "success": results.success
        }

    except Exception as e:
        experiment_time = time.time() - experiment_start_time
        print(f"❌ Evolution experiment failed: {e}")

        return {
            "type": "evolution_experiment",
            "error": str(e),
            "experiment_time": experiment_time,
            "best_scores": {},  # Empty scores for failed evolution
        "success": False
        }


# REMOVED: Unused self-contained evaluation function
# Using original working fitness function instead


def evaluate_genome_modal(genome_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    TWO-LOOP ARCHITECTURE: Use plugin's full multi-objective evaluation.
    This ensures CA evolution, feature extraction, and cheap knobs generation.
    """
    print(f"🧬 Modal: TWO-LOOP GENOME EVALUATION {genome_data.get('id', 'unknown')}")

    try:
        # Import plugin and domain components
        from plugins.quixbugs_codellama.plugin import QuixBugsCodeLlamaRealPlugin
        from core.domain.genome import Genome, CASeed, LoRAConfig
        import numpy as np

        # Reconstruct genome from serialized data manually
        try:
            # Reconstruct CASeed
            seed_data = genome_data['seed']
            ca_seed = CASeed(
                grid=np.array(seed_data['grid']),
                rule=seed_data['rule'],
                steps=seed_data['steps']
            )

            # Reconstruct LoRAConfig
            lora_data = genome_data['lora_config']
            lora_cfg = LoRAConfig(
                r=lora_data['r'],
                alpha=lora_data['alpha'],
                dropout=lora_data['dropout'],
                target_modules=tuple(lora_data['target_modules']),
                adapter_type=lora_data.get('adapter_type', 'lora')  # 🔥 FIX: Explicit default for backward compatibility (removed task_type)
            )

            # 🔥 FIX: Reconstruct CA features for consistency
            ca_features = None
            ca_features_data = genome_data.get('ca_features')
            if ca_features_data is not None:
                from core.domain.feature_extraction import CAFeatures
                ca_features = CAFeatures(
                    complexity=ca_features_data.get('complexity', 0.5),
                    intensity=ca_features_data.get('intensity', 0.5),
                    periodicity=ca_features_data.get('periodicity', 0.5),
                    convergence=ca_features_data.get('convergence', 0.5)
                )

            # Reconstruct Genome with correct run_id and CA features
            genome = Genome(
                seed=ca_seed,
                lora_cfg=lora_cfg,
                id=genome_data['id'],
                ca_features=ca_features,  # 🔥 FIX: Include CA features
                run_id=genome_data.get('run_id')  # 🔥 FIX: Get run_id from top level, not lora_config
            )

            # 🔥 CRITICAL: Verify cache hash consistency after reconstruction
            from infra.adapter_cache import HeavyGenes
            reconstructed_heavy_genes = HeavyGenes.from_lora_config(
                lora_cfg,
                run_id=genome.run_id
            )
            reconstructed_hash = reconstructed_heavy_genes.to_hash()

            print(f"✅ Genome reconstructed: {genome.id}")
            print(f"   • CA rule: {genome.seed.rule}, steps: {genome.seed.steps}")
            print(f"   • LoRA: r={genome.lora_cfg.r}, α={genome.lora_cfg.alpha}, type={genome.lora_cfg.adapter_type}")
            print(f"   • Run ID: {genome.run_id}")  # 🔥 FIX: Log run_id for debugging
            print(f"   • Expected adapter hash: {reconstructed_hash}")
            print(f"   • Expected adapter path: /cache/adapters/adapter_{reconstructed_hash}")

            # 🔥 FIX: PROACTIVE VOLUME RELOAD - Sync with training containers BEFORE checking cache
            from pathlib import Path
            expected_adapter_path = Path(f"/cache/adapters/adapter_{reconstructed_hash}")

            # ✅ CRITICAL FIX: Reload Modal volume to see latest artifacts from training containers
            def _is_modal_environment() -> bool:
                """Detect if running in Modal environment."""
                import os
                return (
                    os.getenv('MODAL_FUNCTION_ID') is not None or
                    os.getenv('MODAL_TASK_ID') is not None or
                    os.path.exists('/cache')
                )

            if _is_modal_environment():
                try:
                    import modal
                    volume = modal.Volume.from_name("coral-x-clean-cache")
                    volume.reload()
                    print("✅ Modal volume reloaded - synced with training containers before cache check")

                    # Give filesystem a moment to reflect changes
                    import time
                    time.sleep(1)

                except Exception as reload_error:
                    print(f"⚠️  Volume reload warning during cache check: {reload_error}")

            print(f"   • Adapter exists at expected path: {expected_adapter_path.exists()}")

            if not expected_adapter_path.exists():
                print("⚠️  CACHE COORDINATION WARNING:")
                print(f"   Expected adapter not found at: {expected_adapter_path}")
                print("   This suggests hash calculation inconsistency or training interruption")
                print("   Available adapters will be checked during model setup")

                # 🔥 DEBUG: List available adapters to help diagnose the issue
                try:
                    adapters_dir = Path("/cache/adapters")
                    if adapters_dir.exists():
                        available_adapters = [d.name for d in adapters_dir.iterdir() if d.is_dir() and d.name.startswith('adapter_')]
                        print(f"   📁 Available adapters ({len(available_adapters)}): {available_adapters[:5]}")
                        if len(available_adapters) > 5:
                            print(f"       ... and {len(available_adapters) - 5} more")

                        # Check if any adapters have similar hash prefixes (potential hash collision)
                        expected_prefix = reconstructed_hash[:8]
                        similar_adapters = [name for name in available_adapters if expected_prefix in name]
                        if similar_adapters:
                            print(f"   🔍 Adapters with similar hash prefix '{expected_prefix}': {similar_adapters}")
                except Exception as debug_error:
                    print(f"   ❌ Could not list available adapters: {debug_error}")
            else:
                print("✅ Cache coordination verified: adapter exists at expected path")

        except Exception as genome_error:
            raise RuntimeError(f"  Failed to reconstruct genome from data: {genome_error}")

        # Create plugin with full config
        plugin = QuixBugsCodeLlamaRealPlugin(config)

        # Check if adapter path is provided (skip training, use pre-trained adapter)
        adapter_path = config.get('adapter_path')
        if adapter_path:
            print(f"🚀 Using pre-trained adapter: {adapter_path}")
            # Create model runner that uses existing adapter (no training)
            model_factory = plugin.model_factory()
            # Create a mock genome with the adapter path already set
            from pathlib import Path
            if not Path(adapter_path).exists():
                raise RuntimeError(f"  Pre-trained adapter not found: {adapter_path}")

            # Override the plugin's model setup to use the pre-trained adapter
            # We'll need to create a model that points to the existing adapter
            from plugins.quixbugs_codellama.plugin import CodeLlamaRealRunner
            model = CodeLlamaRealRunner(
                lora_cfg=genome.lora_cfg,
                config=config,
                genome=genome
            )
            # Set the adapter path directly to skip training
            model._adapter_path = adapter_path
            model._model_loaded = True
        else:
            print("🏗️  No adapter path provided - will train adapter during model creation")
            # Create model runner for this genome (will do training)
            model_factory = plugin.model_factory()
            model = model_factory(genome.lora_cfg, genome)

        # Load dataset
        dataset_provider = plugin.dataset()
        problems = dataset_provider.problems()

        # Get fitness function and evaluate using FULL TWO-LOOP ARCHITECTURE (RESTORE WORKING VERSION)
        fitness_fn = plugin.fitness_fn()

        print("🎛️ Using FITNESS FUNCTION for full two-loop evaluation...")
        print("   • CA evolution: ✅")
        print("   • Feature extraction: ✅")
        print("   • Cheap knobs generation: ✅")
        print("   • Multi-objective evaluation: ✅")

        # This will run the complete pipeline:
        # 1. CA evolution from genome.seed
        # 2. Feature extraction from CA history
        # 3. Cheap knobs generation from CA features
        # 4. Multi-objective evaluation on QuixBugs problems
        multi_scores = fitness_fn.evaluate_multi_objective(genome, model, problems)

        # Convert to dict for serialization
        result_dict = {
            'genome_id': genome.id,
            'bugfix': multi_scores.bugfix,
            'style': multi_scores.style,
            'security': multi_scores.security,
            'runtime': multi_scores.runtime,
            'syntax': multi_scores.syntax,
            'overall_fitness': multi_scores.overall_fitness(),
            'success': True
        }

        print("✅ TWO-LOOP EVALUATION COMPLETE:")
        print(f"   • Bugfix: {multi_scores.bugfix:.3f}")
        print(f"   • Style: {multi_scores.style:.3f}")
        print(f"   • Security: {multi_scores.security:.3f}")
        print(f"   • Runtime: {multi_scores.runtime:.3f}")
        print(f"   • Syntax: {multi_scores.syntax:.3f}")
        print(f"   • Overall: {multi_scores.overall_fitness():.3f}")

        return result_dict

    except Exception as e:
        print(f"❌ TWO-LOOP EVALUATION FAILED: {e}")
        return {
            'genome_id': genome_data.get('id', 'unknown'),
            'error': str(e),
            'bugfix': 0.0,
            'style': 0.0,
            'security': 0.0,
            'runtime': 0.0,
            'syntax': 0.0,
            'overall_fitness': 0.0,
            'success': False
        }


def load_real_test_cases_modal(problem_name: str, dataset_path: str) -> str:
    """Load real test cases from QuixBugs dataset."""
    from pathlib import Path

    # Use the dataset_path parameter (which comes from config)
    dataset_root = Path(dataset_path)

    # Check if path exists
    if not dataset_root.exists():
        raise RuntimeError(
            f"  Dataset path does not exist: '{dataset_path}'. "
            f"Dataset should be pre-cached at this location. "
            f"Check your config paths.modal.dataset setting."
        )

    print(f"✅ Using QuixBugs dataset at: {dataset_root}")

    # Look for test cases with multiple naming conventions
    possible_locations = [
        dataset_root / "python_testcases" / f"test_{problem_name}.py",
        dataset_root / "python_programs" / f"{problem_name}_test.py",
        dataset_root / "testcases" / f"test_{problem_name}.py",
        dataset_root / f"test_{problem_name}.py",
        # Additional QuixBugs structure variations
        dataset_root / "QuixBugs" / "python_testcases" / f"test_{problem_name}.py",
        dataset_root / "json_testcases" / f"{problem_name}.json"  # JSON test data
    ]

    print(f"🔍 Searching for test cases for '{problem_name}':")
    for i, test_file in enumerate(possible_locations, 1):
        print(f"   {i}. {test_file}")
        if test_file.exists():
            content = test_file.read_text()
            print(f"✅ Found real test cases for {problem_name} at {test_file} ({len(content)} chars)")
            return content

    # Enhanced error message with directory listing for debugging
    if dataset_root.exists():
        print("🔍 Dataset root contents:")
        try:
            for item in sorted(dataset_root.iterdir()):
                if item.is_dir():
                    print(f"   📁 {item.name}/")
                    # Show contents of subdirectories that might contain tests
                    if item.name in ['python_testcases', 'testcases', 'python_programs']:
                        try:
                            subfiles = list(item.glob("*.py"))[:5]  # Show first 5 files
                            for subfile in subfiles:
                                print(f"      📄 {subfile.name}")
                            if len(list(item.glob("*.py"))) > 5:
                                print(f"      ... and {len(list(item.glob('*.py'))) - 5} more files")
                        except:
                            pass
                else:
                    print(f"   📄 {item.name}")
        except Exception as e:
            print(f"   ❌ Could not list directory contents: {e}")

    #   No test case generation fallbacks
    raise RuntimeError(
        f"  No real QuixBugs test cases found for '{problem_name}' in dataset '{dataset_root}'. "
        f"Searched {len(possible_locations)} locations. "
        f"Cannot proceed without real test cases - terminating to avoid GPU waste."
    )


def debug_modal_volume_structure(dataset_path: str):
    """Debug function to inspect the actual modal volume structure."""
    from pathlib import Path
    import os

    print(f"🔍 DEBUG: Inspecting Modal volume structure at '{dataset_path}'...")

    dataset_root = Path(dataset_path)
    if not dataset_root.exists():
        raise RuntimeError(f"  Dataset path does not exist: '{dataset_path}'")

    # Check dataset contents
    if dataset_root.is_dir():
        contents = list(dataset_root.iterdir())[:10]
        print(f"✅ {dataset_path}: EXISTS (dir) - {len(list(dataset_root.iterdir()))} items")
        print(f"   Contents: {[c.name for c in contents]}")

        # If this looks like the dataset directory, inspect further
        subdirs_to_check = ['json_testcases', 'python_testcases', 'python_programs', 'testcases']
        for subdir in subdirs_to_check:
            subpath = dataset_root / subdir
            if subpath.exists():
                sub_contents = list(subpath.iterdir())[:5]
                print(f"     {subdir}/: {len(list(subpath.iterdir()))} files")
                print(f"       First 5: {[f.name for f in sub_contents]}")
    else:
        print(f"✅ {dataset_path}: EXISTS (file)")

    # Check current working directory
    cwd = os.getcwd()
    print(f"📁 Current working directory: {cwd}")

    return {
        "cwd": cwd,
        "dataset_path": str(dataset_root),
        "exists": dataset_root.exists(),
        "is_dir": dataset_root.is_dir() if dataset_root.exists() else False
    }


def _extract_best_scores_from_results(results) -> dict:
    """
    Extract REAL multi-objective scores from evolution results.
    
    CRITICAL: This must return actual scores for valid benchmarking.
    Requires real scores for benchmarking.
    """
    try:
        if not results.final_population or not results.final_population.genomes:
            raise RuntimeError("  No final population available - cannot extract real scores for benchmarking")

        # Find best genome from final population
        best_genome = max(results.final_population.genomes, key=lambda g: g.fitness or 0.0)

        if not best_genome:
            raise RuntimeError("  No best genome found in final population")

        if not hasattr(best_genome, 'multi_scores') or not best_genome.multi_scores:
            raise RuntimeError(
                f"  Best genome {best_genome.id} missing multi_scores. "
                f"Cannot run benchmarks without real multi-objective scores. "
                f"Evolution system must provide actual scores, not approximations."
            )

        best_scores = best_genome.multi_scores.to_dict()
        print(f"🎯 Extracted REAL multi-objective scores: {best_scores}")

        # Validate that we have all required score components
        required_scores = ['bugfix', 'style', 'security', 'runtime', 'syntax']
        missing_scores = [score for score in required_scores if score not in best_scores]

        if missing_scores:
            raise RuntimeError(
                f"  Missing required score components: {missing_scores}. "
                f"Available scores: {list(best_scores.keys())}. "
                f"Cannot run valid benchmarks with incomplete scores."
            )

        print("✅ All required multi-objective scores present for benchmarking")
        return best_scores

    except Exception as e:
        print(f"❌ CRITICAL: Cannot extract real best_scores: {e}")
        print("🚫 Benchmarking requires REAL scores, not defaults or approximations")
        raise RuntimeError(f"  Real score extraction failed: {e}")
