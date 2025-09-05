"""Simple CLI for CORAL-X evolution experiments."""

import argparse
import sys
from pathlib import Path

def main():
    """Simple CLI entry point."""
    parser = argparse.ArgumentParser(description="CORAL-X Evolution Framework")

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Run command
    run_parser = subparsers.add_parser('run', help='Run evolution experiment')
    run_parser.add_argument('--config', type=Path, required=True, help='Config file path')

    args = parser.parse_args()

    if args.command == 'run':
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

            # Run evolution
            from core.application.evolution_orchestrator import EvolutionOrchestrator
            from core.application.services import create_evolution_services

            services = create_evolution_services(config)
            orchestrator = EvolutionOrchestrator(services)

            import asyncio
            result = asyncio.run(orchestrator.run_evolution())

            if result.status == 'completed':
                print("\n🎉 EVOLUTION COMPLETED SUCCESSFULLY!")
                print(f"⏱️  Total Time: {result.total_time:.2f}s")
                print(f"🔄 Generations Completed: {result.generations_completed}")

                # Display detailed results
                if result.best_genome:
                    print("\n🏆 BEST GENOME RESULTS:")
                    print(f"   ID: {result.best_genome.id}")
                    print(f"   Fitness: {result.best_genome.fitness:.4f}")

                    if hasattr(result.best_genome, 'lora_cfg') and result.best_genome.lora_cfg:
                        print("   LoRA Config:")
                        print(f"     Rank: {result.best_genome.lora_cfg.r}")
                        print(f"     Alpha: {result.best_genome.lora_cfg.alpha}")
                        print(f"     Dropout: {result.best_genome.lora_cfg.dropout}")
                        print(f"     Target Modules: {', '.join(result.best_genome.lora_cfg.target_modules)}")

                    if hasattr(result.best_genome, 'seed') and result.best_genome.seed:
                        print("   CA Seed:")
                        print(f"     Grid Size: {result.best_genome.seed.grid.shape}")
                        print(f"     Rule: {result.best_genome.seed.rule}")
                        print(f"     Steps: {result.best_genome.seed.steps}")

                # Display population statistics
                if result.final_population and result.final_population.size() > 0:
                    print("\n📊 POPULATION STATISTICS:")
                    print(f"   Final Population Size: {result.final_population.size()}")

                    # Get fitness statistics
                    evaluated_genomes = [g for g in result.final_population.genomes if g.is_evaluated()]
                    if evaluated_genomes:
                        fitness_scores = [g.fitness for g in evaluated_genomes if g.fitness is not None]
                        if fitness_scores:
                            print(f"   Fitness Range: {min(fitness_scores):.4f} - {max(fitness_scores):.4f}")
                            print(f"   Average Fitness: {sum(fitness_scores)/len(fitness_scores):.4f}")
                            print(f"   Evaluated Genomes: {len(evaluated_genomes)}/{result.final_population.size()}")

                print("\n✨ Evolution experiment completed successfully!")

                # Try to get additional metrics from services
                try:
                    print("\n📈 ADDITIONAL METRICS:")

                    # Get genetic operations summary
                    genetic_summary = services.genetic_operations.get_generation_summary()
                    if genetic_summary and 'message' not in genetic_summary:
                        print("   Genetic Operations:")
                        print(f"     Total Crossovers: {genetic_summary.get('total_crossovers', 'N/A')}")
                        print(f"     Total Mutations: {genetic_summary.get('total_mutations', 'N/A')}")
                        print(f"     Avg Operation Time: {genetic_summary.get('avg_operation_time', 'N/A'):.3f}s")
                        print(f"     Avg Diversity Strength: {genetic_summary.get('avg_diversity_strength', 'N/A'):.3f}")

                    # Get progress summary
                    progress_data = services.progress_tracker.get_current_progress()
                    if progress_data and 'generation_history' in progress_data:
                        gen_history = progress_data['generation_history']
                        if gen_history:
                            latest_gen = gen_history[-1]
                            print("   Latest Generation:")
                            print(f"     Population Size: {latest_gen.get('population_size', 'N/A')}")
                            print(f"     Evaluation Rate: {latest_gen.get('evaluation_rate', 'N/A'):.1%}")
                            print(f"     Best Fitness: {latest_gen.get('best_fitness', 'N/A'):.4f}")
                            print(f"     Diversity Score: {latest_gen.get('diversity_score', 'N/A'):.3f}")

                    # Get LoRA training metrics
                    if progress_data and 'training_stats' in progress_data:
                        training_stats = progress_data['training_stats']
                        if training_stats:
                            print("   LoRA Training:")
                            print(f"     Adapters Trained: {training_stats.get('adapters_trained', 'N/A')}")
                            print(f"     Training Rate: {training_stats.get('training_rate', 'N/A'):.1%}")
                            current_adapter = training_stats.get('current_adapter', 'N/A')
                            if current_adapter != 'N/A':
                                print(f"     Current Adapter: {current_adapter}")

                except Exception as e:
                    print(f"   Additional metrics unavailable: {e}")

            else:
                print(f"❌ Evolution failed: {result.status}")
                sys.exit(1)

        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
