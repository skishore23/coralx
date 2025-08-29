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
            
            if result.success:
                print(f"✅ Evolution completed: {result.message}")
            else:
                print(f"❌ Evolution failed: {result.error_message}")
                sys.exit(1)
                
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()