"""Simple CLI for CORAL-X evolution experiments."""

import argparse
import sys
from pathlib import Path


def main() -> None:
    """Simple CLI entry point."""
    parser = argparse.ArgumentParser(description="CORAL-X Evolution Framework")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run evolution experiment")
    run_parser.add_argument(
        "--config", type=Path, required=True, help="Config file path"
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config, plugin resolution, and deterministic population creation without running evolution",
    )
    prove_parser = subparsers.add_parser(
        "prove", help="Run GSM8K LoRA evolution with fixed/random controls"
    )
    prove_parser.add_argument(
        "--config", type=Path, required=True, help="Config file path"
    )
    prove_parser.add_argument(
        "--random-trials",
        type=int,
        default=None,
        help="Number of random LoRA candidates to evaluate after evolution",
    )
    prove_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Proof report JSON output path",
    )

    args = parser.parse_args()

    if args.command == "run":
        config_path = args.config

        if not config_path.exists():
            print(f"❌ Config file not found: {config_path}")
            sys.exit(1)

        print(f"🔧 Loading config: {config_path}")

        try:
            # Load config
            from core.common.config_loader import load_config

            config = load_config(Path(config_path))

            print(f"🔧 Starting experiment: {config.experiment.name}")

            if args.dry_run:
                _run_dry_validation(config)
                print("✅ Dry run validation completed")
                return

            # Run evolution
            from core.application.evolution_orchestrator import EvolutionOrchestrator
            from core.application.services import create_evolution_services

            services = create_evolution_services(config)
            orchestrator = EvolutionOrchestrator(services)

            import asyncio

            result = asyncio.run(orchestrator.run_evolution())

            if result.status == "completed":
                print("\n🎉 EVOLUTION COMPLETED SUCCESSFULLY!")
                print(f"⏱️  Total Time: {result.total_time:.2f}s")
                print(f"🔄 Generations Completed: {result.generations_completed}")

                # Display detailed results
                if result.best_genome:
                    print("\n🏆 BEST GENOME RESULTS:")
                    print(f"   ID: {result.best_genome.id}")
                    print(f"   Fitness: {result.best_genome.fitness:.4f}")

                    if (
                        hasattr(result.best_genome, "lora_cfg")
                        and result.best_genome.lora_cfg
                    ):
                        print("   LoRA Config:")
                        print(f"     Rank: {result.best_genome.lora_cfg.r}")
                        print(f"     Alpha: {result.best_genome.lora_cfg.alpha}")
                        print(f"     Dropout: {result.best_genome.lora_cfg.dropout}")
                        print(
                            f"     Target Modules: {', '.join(result.best_genome.lora_cfg.target_modules)}"
                        )

                    if hasattr(result.best_genome, "seed") and result.best_genome.seed:
                        print("   CA Seed:")
                        print(f"     Grid Size: {result.best_genome.seed.grid.shape}")
                        print(f"     Rule: {result.best_genome.seed.rule}")
                        print(f"     Steps: {result.best_genome.seed.steps}")

                # Display population statistics
                if result.final_population and result.final_population.size() > 0:
                    print("\n📊 POPULATION STATISTICS:")
                    print(f"   Final Population Size: {result.final_population.size()}")

                    # Get fitness statistics
                    evaluated_genomes = [
                        g for g in result.final_population.genomes if g.is_evaluated()
                    ]
                    if evaluated_genomes:
                        fitness_scores = [
                            g.fitness
                            for g in evaluated_genomes
                            if g.fitness is not None
                        ]
                        if fitness_scores:
                            print(
                                f"   Fitness Range: {min(fitness_scores):.4f} - {max(fitness_scores):.4f}"
                            )
                            print(
                                f"   Average Fitness: {sum(fitness_scores) / len(fitness_scores):.4f}"
                            )
                            print(
                                f"   Evaluated Genomes: {len(evaluated_genomes)}/{result.final_population.size()}"
                            )

                print("\n✨ Evolution experiment completed successfully!")

                # Try to get additional metrics from services
                try:
                    print("\n📈 ADDITIONAL METRICS:")

                    # Get genetic operations summary
                    genetic_summary = (
                        services.genetic_operations.get_generation_summary()
                    )
                    if genetic_summary and "message" not in genetic_summary:
                        print("   Genetic Operations:")
                        print(
                            f"     Total Crossovers: {genetic_summary.get('total_crossovers', 'N/A')}"
                        )
                        print(
                            f"     Total Mutations: {genetic_summary.get('total_mutations', 'N/A')}"
                        )
                        print(
                            f"     Avg Operation Time: {genetic_summary.get('avg_operation_time', 'N/A'):.3f}s"
                        )
                        print(
                            f"     Avg Diversity Strength: {genetic_summary.get('avg_diversity_strength', 'N/A'):.3f}"
                        )

                    # Get progress summary
                    progress_data = services.progress_tracker.get_current_progress()
                    if progress_data and "generation_history" in progress_data:
                        gen_history = progress_data["generation_history"]
                        if gen_history:
                            latest_gen = gen_history[-1]
                            print("   Latest Generation:")
                            print(
                                f"     Population Size: {latest_gen.get('population_size', 'N/A')}"
                            )
                            print(
                                f"     Evaluation Rate: {latest_gen.get('evaluation_rate', 'N/A'):.1%}"
                            )
                            print(
                                f"     Best Fitness: {latest_gen.get('best_fitness', 'N/A'):.4f}"
                            )
                            print(
                                f"     Diversity Score: {latest_gen.get('diversity_score', 'N/A'):.3f}"
                            )

                    # Training metrics are only meaningful for plugins that train adapters.
                    if progress_data and "training_stats" in progress_data:
                        training_stats = progress_data["training_stats"]
                        adapters_trained = training_stats.get("adapters_trained", 0)
                        if training_stats and adapters_trained:
                            print("   LoRA Training:")
                            print(f"     Adapters Trained: {adapters_trained}")
                            print(
                                f"     Training Rate: {training_stats.get('training_rate', 'N/A'):.1%}"
                            )
                            current_adapter = training_stats.get(
                                "current_adapter", "N/A"
                            )
                            if current_adapter != "N/A":
                                print(f"     Current Adapter: {current_adapter}")

                except Exception as e:
                    print(f"   Additional metrics unavailable: {e}")

            else:
                print(f"❌ Evolution failed: {result.status}")
                sys.exit(1)

        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)

    elif args.command == "prove":
        config_path = args.config

        if not config_path.exists():
            print(f"❌ Config file not found: {config_path}")
            sys.exit(1)

        print(f"🔧 Loading config: {config_path}")
        try:
            from core.cli.proof import run_gsm8k_lora_proof
            from core.common.config_loader import load_config

            config = load_config(Path(config_path))
            print(f"🔬 Starting proof run: {config.experiment.name}")
            report = run_gsm8k_lora_proof(
                config,
                random_trials=args.random_trials,
                output_path=args.output,
            )

            evolution_best = report["evolution"]["best_by_fitness"]
            evolution_best_exact = report["evolution"].get("best_by_exact_accuracy")
            random_best = report["random_baseline"]["best"]
            base = report["base_model_baseline"]
            fixed = report["fixed_baseline"]
            exact = report["interpretation"].get("exact_accuracy", {})
            print("\n✅ PROOF RUN COMPLETED")
            print(
                f"   Evolution best fitness: {evolution_best['fitness']:.4f}"
                if evolution_best
                else "   Evolution best fitness: unavailable"
            )
            if evolution_best_exact:
                print(
                    "   Evolution best exact candidate: "
                    f"{_format_optional_float(evolution_best_exact['metrics'].get('exact_accuracy'))} "
                    f"({evolution_best_exact['genome_id']})"
                )
            if exact:
                print(
                    "   Exact accuracy: "
                    f"evolution={_format_optional_float(exact.get('evolution_best'))}, "
                    f"base={_format_optional_float(exact.get('base_model'))}, "
                    f"fixed={_format_optional_float(exact.get('fixed'))}, "
                    f"random_best={_format_optional_float(exact.get('random_best'))}"
                )
            print(f"   Base model fitness: {base['fitness']:.4f}")
            print(f"   Fixed baseline fitness: {fixed['fitness']:.4f}")
            if random_best:
                print(f"   Random best fitness: {random_best['fitness']:.4f}")
                print(
                    "   Evolution - random best: "
                    f"{report['interpretation']['evolution_minus_random_best']:.4f}"
                )
            print(f"   Report: {report['artifacts']['proof_report']}")

        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)

    else:
        parser.print_help()
        sys.exit(1)


def _run_dry_validation(config) -> None:
    """Validate local run wiring without evaluating or downloading models."""
    from core.domain.experiment import (
        create_experiment_config,
        create_initial_population,
    )
    from plugins.registry import create_plugin

    plugin = create_plugin(config)
    print(f"   Plugin: {type(plugin).__name__}")

    experiment_config = create_experiment_config(config.model_dump(mode="json"))
    first_population = create_initial_population(
        experiment_config,
        raw_config=config.model_dump(mode="json"),
        run_id=config.cache.run_id,
    )
    second_population = create_initial_population(
        experiment_config,
        raw_config=config.model_dump(mode="json"),
        run_id=config.cache.run_id,
    )

    first_keys = [genome.get_heavy_genes_key() for genome in first_population.genomes]
    second_keys = [genome.get_heavy_genes_key() for genome in second_population.genomes]
    if first_keys != second_keys:
        raise RuntimeError("Deterministic population check failed")

    config.execution.output_dir.mkdir(parents=True, exist_ok=True)
    config.cache.artifacts_dir.mkdir(parents=True, exist_ok=True)
    print(f"   Population: {first_population.size()} deterministic genomes")
    print(f"   Output dir: {config.execution.output_dir}")
    print(f"   Cache dir: {config.cache.artifacts_dir}")


def _format_optional_float(value) -> str:
    """Format optional float values for CLI summaries."""
    return "n/a" if value is None else f"{float(value):.4f}"


if __name__ == "__main__":
    main()
