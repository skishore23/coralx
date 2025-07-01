###############################################################################
# Enhanced CoralX CLI - NO FALLBACKS, config-driven
###############################################################################
import sys
import subprocess
import time
from pathlib import Path
from typing import Optional
import numpy as np
from random import Random
import yaml

# Add parent directory to path for imports
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from coralx package structure  
from coral.config.loader import load_config, create_default_config
from coral.application.evolution_engine import EvolutionEngine, CoralConfig
from coral.domain.genome import Genome
from coral.domain.ca import CASeed
from plugins.quixbugs_codellama.plugin import QuixBugsCodeLlamaRealPlugin
from coral.domain.neat import Population
from coral.domain.mapping import LoRAConfig
from infra.modal_executor import create_executor_from_config
from plugins.quixbugs_codellama.plugin import QuixBugsCodeLlamaRealPlugin
from benchmarks.benchmark_runner import run_benchmarks
from reporting.report_generator import render_report
from coral.application.experiment_orchestrator import ExperimentOrchestrator


def get_version() -> str:
    """Get CoralX version from VERSION file."""
    version_file = Path(__file__).parent.parent / "VERSION"
    if not version_file.exists():
        raise FileNotFoundError("FAIL-FAST: VERSION file not found")
    return version_file.read_text().strip()


def check_modal_available() -> bool:
    """Check if Modal CLI is available."""
    try:
        subprocess.run(['modal', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def run_experiment(config_path: str, stream_logs: bool = False) -> None:
    """Main experiment runner with automatic live monitoring for Modal."""
    print("🧬 CoralX - Functional CORAL Evolution System")
    print(f"🔖 Version: {get_version()}")
    print("=" * 50)
    
    # Load configuration - fail if not found
    if not Path(config_path).exists():
        raise FileNotFoundError(f"FAIL-FAST: Config file not found: {config_path}")
    
    config = load_config(config_path)
    print(f"✅ Configuration loaded from: {config_path}")
    
    # Check executor type and run accordingly
    executor_type = config.infra.get('executor', 'local') if hasattr(config, 'infra') else 'local'
    
    if executor_type == 'modal':
        print(f"🚀 Executor: Modal (with live monitoring)")
        _run_modal_experiment_with_live_dashboard(config_path, config)
    elif executor_type == 'queue_modal':
        print(f"🚀 Executor: Queue-based Modal (clean architecture)")
        _run_queue_modal_experiment(config_path, config)
    else:
        print(f"🖥️  Executor: Local")
        _run_local_experiment(config_path, config)


def _run_modal_experiment_with_live_dashboard(config_path: str, config: CoralConfig) -> None:
    """Run Modal experiment with integrated live dashboard."""
    import threading
    import time
    import subprocess
    
    print("📺 Starting Modal experiment with live monitoring...")
    print("🎯 This will show real-time progress automatically!")
    print("=" * 60)
    
    # Start the Modal experiment in background using new optimized Modal app
    import json
    
    # Convert CoralConfig to dict for Modal using plugin
    # This approach is cleaner - plugin handles config management
    # Load raw config for plugin that expects full config dict
    import yaml
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)
    
    plugin = QuixBugsCodeLlamaRealPlugin(raw_config)
    config_dict = plugin.get_modal_config(config)
    
    modal_cmd = [
        'modal', 'run', 'coral_modal_app.py::run_experiment_modal',
        '--config-dict', json.dumps(config_dict)
    ]
    
    # Start Modal experiment process
    print("🚀 Launching Modal experiment...")
    modal_process = subprocess.Popen(
        modal_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    # Give Modal experiment time to start
    time.sleep(3)
    
    # Start live monitor in a separate thread
    def run_live_monitor():
        try:
            monitor_script = Path("scripts/monitor_evolution.py")
            if monitor_script.exists():
                print("📊 Starting live progress monitor...")
                subprocess.run([sys.executable, str(monitor_script)])
        except Exception as e:
            print(f"⚠️  Live monitor failed: {e}")
    
    monitor_thread = threading.Thread(target=run_live_monitor, daemon=True)
    monitor_thread.start()
    
    print("=" * 60)
    print("🔥 EVOLUTION RUNNING WITH LIVE DASHBOARD!")
    print("📺 Watch real-time progress in the monitor above")
    print("📊 Modal logs available with: modal app logs coral-x-production")
    print("⌨️  Press Ctrl+C to stop both experiment and monitor")
    print("=" * 60)
    
    try:
        # Stream Modal experiment output
        for line in modal_process.stdout:
            print(f"[Modal] {line.rstrip()}")
        
        # Wait for Modal experiment to complete
        modal_process.wait()
        
        if modal_process.returncode == 0:
            print("\n✅ Modal experiment completed successfully!")
        else:
            print(f"\n❌ Modal experiment failed with code: {modal_process.returncode}")
            
    except KeyboardInterrupt:
        print("\n⚠️  User interrupted - stopping experiment and monitor...")
        modal_process.terminate()
        try:
            modal_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            modal_process.kill()


def _run_queue_modal_experiment(config_path: str, config: CoralConfig) -> None:
    """Run queue-based Modal experiment."""
    import threading
    import time
    import subprocess
    import json
    import yaml
    
    print("🚀 Starting queue-based Modal experiment...")
    print("⚡ Clean architecture: No function-to-function calls")
    print("🔄 Auto-scaling workers handle all computation")
    print("=" * 60)
    
    # Load raw config for queue-based execution
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)
    
    # Use queue-based Modal app
    modal_cmd = [
        'modal', 'run', 'coral_queue_modal_app.py::run_experiment_modal',
        '--config-dict', json.dumps(raw_config)
    ]
    
    print("🚀 Launching queue-based Modal experiment...")
    print("📋 Queue app: coral-x-queues")
    print("🏗️ Workers will auto-scale based on queue load")
    
    # Run the Modal experiment
    try:
        result = subprocess.run(
            modal_cmd,
            check=True,
            text=True
        )
        
        if result.returncode == 0:
            print("\n✅ Queue-based Modal experiment completed successfully!")
        else:
            print(f"\n❌ Queue-based Modal experiment failed with code: {result.returncode}")
    
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Queue-based Modal experiment failed: {e}")
        raise
    except KeyboardInterrupt:
        print("\n⚠️  User interrupted - stopping experiment...")
        return


def _run_local_experiment(config_path: str, config: CoralConfig) -> None:
    """Run local experiment (original implementation)."""
    # Check fail-fast mode - CoralConfig has system as dict
    system_config = getattr(config, 'system', {})
    if isinstance(system_config, dict) and system_config.get('fail_fast', True):
        print("🚨 FAIL-FAST mode enabled - no fallbacks allowed")
    
    # Load plugin
    plugin = _load_plugin(config)
    print(f"✅ Plugin loaded: {config.experiment.get('target')}")
    
    # Create executor
    executor = create_executor_from_config(config)
    print(f"✅ Executor created: {config.infra.get('executor')}")
    
    # Create evolution engine
    engine = EvolutionEngine(
        cfg=config,
        fitness_fn=plugin.fitness_fn(),
        executor=executor,
        model_factory=plugin.model_factory(),
        dataset=plugin.dataset()
    )
    print("✅ Evolution engine initialized")
    
    # Create initial population
    init_pop = _create_initial_population(config)
    print(f"✅ Initial population created: {init_pop.size()} genomes")
    
    # Run evolution
    print("\n🚀 Starting evolution...")
    try:
        winners = engine.run(init_pop)
        print(f"✅ Evolution completed: {winners.size()} final genomes")
    except KeyboardInterrupt:
        print("\n⚠️  Evolution interrupted by user")
        return
    
    # Run benchmarks
    print("\n📊 Running benchmarks...")
    report = run_benchmarks(
        config=config,
        winners=winners,
        model_factory=plugin.model_factory(),
        dataset=plugin.dataset(),
        fitness_fn=plugin.fitness_fn()
    )
    print("✅ Benchmarks completed")
    
    # Generate reports
    print("\n📝 Generating reports...")
    output_dir = config.execution.get('output_dir')
    if not output_dir:
        raise ValueError("FAIL-FAST: output_dir not specified in config")
    
    render_report(report, output_dir)
    print("✅ Reports generated successfully")
    
    if report.best_genome:
        print(f"\n🏆 Best fitness achieved: {report.best_genome.fitness:.4f}")
        print(f"🧬 Best genome: {report.best_genome.genome_id}")
    
    print("\n🎉 Experiment completed successfully!")


def deploy_modal(app_file: str = "coral_modal_app.py") -> bool:
    """Deploy CoralX application to Modal."""
    print("🚀 Deploying CoralX to Modal...")
    print(f"🔖 Version: {get_version()}")
    print("=" * 50)
    
    if not check_modal_available():
        raise RuntimeError("FAIL-FAST: Modal CLI not available. Install with: pip install modal")
    
    app_path = Path(app_file)
    if not app_path.exists():
        raise FileNotFoundError(f"FAIL-FAST: Modal app file not found: {app_file}")
    
    # Deploy to Modal
    print(f"📦 Deploying {app_file} to Modal...")
    result = subprocess.run([
        'modal', 'deploy', str(app_path)
    ], check=True)
    
    print("✅ CoralX deployed successfully to Modal!")
    print("🌐 Your application is now running on Modal's cloud infrastructure")
    return True


def modal_logs(follow: bool = True, tail: int = 100) -> None:
    """Stream logs from Modal deployment."""
    print("📜 Streaming Modal logs...")
    print(f"🔖 Version: {get_version()}")
    
    if not check_modal_available():
        raise RuntimeError("FAIL-FAST: Modal CLI not available. Install with: pip install modal")
    
    cmd = ['modal', 'logs']
    
    if follow:
        cmd.append('--follow')
    
    if tail > 0:
        cmd.extend(['--tail', str(tail)])
    
    print(f"🔄 Running: {' '.join(cmd)}")
    print("=" * 50)
    
    # Stream logs in real-time
    subprocess.run(cmd)


def modal_apps() -> None:
    """List Modal applications."""
    print("📋 Listing Modal applications...")
    
    if not check_modal_available():
        raise RuntimeError("FAIL-FAST: Modal CLI not available. Install with: pip install modal")
    
    subprocess.run(['modal', 'app', 'list'])


def cache_model(config_path: str) -> None:
    """Pre-download model to Modal cache volume based on config."""
    print("📥 Pre-downloading model to Modal cache...")
    print(f"🔖 Version: {get_version()}")
    print("⏰ This will take several minutes but only needs to be done once.")
    print("=" * 50)
    
    if not check_modal_available():
        raise RuntimeError("FAIL-FAST: Modal CLI not available. Install with: pip install modal")
    
    if not Path(config_path).exists():
        raise FileNotFoundError(f"FAIL-FAST: Config file not found: {config_path}")
    
    # Load config to get model parameters
    config = load_config(config_path)
    
    # Create config JSON for Modal function
    import json
    config_json = json.dumps(config)
    
    # Run the model download function with config
    result = subprocess.run([
        "modal", "run", "coral_modal_app.py::download_model_to_cache",
        "--config", config_json
    ], check=True, capture_output=True, text=True)
    
    print("✅ Model successfully cached!")
    print("🚀 Future experiments will use the cached model and start faster.")
    print("\n📊 Model cache status:")
    print(result.stdout)


def modal_proxy(args: list) -> None:
    """Proxy command to Modal CLI."""
    if not check_modal_available():
        raise RuntimeError("FAIL-FAST: Modal CLI not available. Install with: pip install modal")
    
    cmd = ['modal'] + args
    print(f"🔄 Running: {' '.join(cmd)}")
    subprocess.run(cmd)


def monitor_evolution() -> None:
    """Launch real-time evolution monitor."""
    print("📺 Starting Evolution Monitor...")
    print(f"🔖 Version: {get_version()}")
    
    monitor_script = Path("scripts/monitor_evolution.py")
    if not monitor_script.exists():
        raise FileNotFoundError("FAIL-FAST: Monitor script not found at scripts/monitor_evolution.py")
    
    subprocess.run([sys.executable, str(monitor_script)])


def create_config(config_path: str) -> None:
    """Create a default configuration file."""
    print(f"📝 Creating default configuration at: {config_path}")
    print(f"🔖 Version: {get_version()}")
    
    _create_default_config_file(config_path)
    print("✅ Default configuration created")
    
    # Show the config structure
    with open(config_path) as f:
        config_content = f.read()
    print(f"\n📄 Configuration preview:")
    print("=" * 30)
    print(config_content[:500] + "..." if len(config_content) > 500 else config_content)


def status(config_path: Optional[str] = None) -> None:
    """Show CoralX system status based on config."""
    print("🧬 CoralX System Status")
    print(f"🔖 Version: {get_version()}")
    print("=" * 50)
    
    # Load config if provided
    config = None
    if config_path and Path(config_path).exists():
        config = load_config(config_path)
        print(f"📁 Config loaded from: {config_path}")
    
    # Check dependencies
    print("📦 Dependencies:")
    
    # Get dependencies from config or use defaults
    deps_to_check = []
    if config and 'system' in config and 'dependencies' in config['system']:
        deps_to_check = [(dep, f'import {dep}') for dep in config['system']['dependencies']]
    else:
        deps_to_check = [
            ('numpy', 'import numpy'),
            ('torch', 'import torch'),
            ('transformers', 'import transformers'),
            ('modal', 'import modal'),
            ('yaml', 'import yaml')
        ]
    
    for dep_name, import_cmd in deps_to_check:
        try:
            exec(import_cmd)
            print(f"  ✅ {dep_name}")
        except ImportError:
            print(f"  ❌ {dep_name} (missing)")
    
    # Check Modal status
    print(f"\n🌐 Modal:")
    if check_modal_available():
        print("  ✅ Modal CLI available")
        try:
            result = subprocess.run(['modal', '--version'], capture_output=True, text=True)
            print(f"  📋 Version: {result.stdout.strip()}")
        except:
            print("  ⚠️  Could not get Modal version")
    else:
        print("  ❌ Modal CLI not available")


def _load_plugin(config):
    """Load the appropriate plugin based on configuration."""
    target = config.experiment.get('target')
    if not target:
        raise ValueError("FAIL-FAST: No plugin target specified in config")
    
    if target == 'quixbugs_codellama':
        return QuixBugsCodeLlamaRealPlugin(config.experiment)
    else:
        raise ValueError(f"FAIL-FAST: Unknown plugin target: {target}")


def _create_initial_population(config) -> Population:
    """Create initial population based on config parameters."""
    execution_config = config.execution
    evo_config = config.evo
    
    # Get population parameters from config
    population_size = execution_config.get('population_size')
    if not population_size:
        raise ValueError("FAIL-FAST: population_size not specified in config")
    
    # Check if cache-friendly groups are specified
    cache_groups = execution_config.get('cache_friendly_groups')
    
    if cache_groups is not None:
        print(f"🔧 Using cache-friendly population (legacy mode)")
        print(f"   📦 Cache groups: {cache_groups}")
        print(f"   ⚠️  Note: This bypasses CA→LoRA mapping innovation")
        return _create_cache_friendly_population(config, cache_groups)
    else:
        print(f"🧬 Using CORAL-X innovation path")
        print(f"   🎯 CA Evolution → Feature Extraction → LoRA Mapping")
        return _create_coralx_population(config)


def _create_cache_friendly_population(config, cache_groups: int) -> Population:
    """Create population with pre-assigned cache groups (legacy mode)."""
    execution_config = config.execution
    evo_config = config.evo
    population_size = execution_config['population_size']
    
    # Get CA parameters from config
    ca_config = evo_config.get('ca', {})
    grid_size = ca_config.get('grid_size', [8, 8])
    rule_range = ca_config.get('rule_range', [1, 255])
    steps_range = ca_config.get('steps_range', [5, 20])
    initial_density = ca_config.get('initial_density', 0.3)
    
    # Get random seed from config
    seed_value = config.get('seed', 42)
    rng = Random(seed_value)
    
    # Create cache-friendly LoRA configs
    lora_configs = _create_cache_friendly_lora_configs(evo_config, cache_groups, rng)
    
    # Calculate genomes per cache group
    genomes_per_group = max(1, population_size // len(lora_configs))
    
    genomes = []
    for i in range(population_size):
        # Create unique genome ID
        genome_id = f"gen0_genome{i:04d}"
        
        # Create CA seed from config parameters
        grid = np.random.rand(*grid_size) < initial_density
        grid = grid.astype(int)
        rule = rng.randint(*rule_range)
        steps = rng.randint(*steps_range)
        
        ca_seed = CASeed(grid=grid, rule=rule, steps=steps)
        
        # Assign LoRA config from cache groups
        group_index = i // genomes_per_group
        if group_index >= len(lora_configs):
            group_index = group_index % len(lora_configs)
        
        lora_config = lora_configs[group_index]
        
        genome = Genome(seed=ca_seed, lora_cfg=lora_config, id=genome_id)
        genomes.append(genome)
    
    print(f"🔄 Cache optimization: {population_size} genomes → {len(lora_configs)} cache groups")
    
    return Population(tuple(genomes))


def _create_coralx_population(config) -> Population:
    """Create population using full CORAL-X innovation (CA → Features → LoRA)."""
    execution_config = config.execution
    evo_config = config.evo
    population_size = execution_config['population_size']
    
    # Get CA parameters from config
    ca_config = evo_config.get('ca', {})
    grid_size = ca_config.get('grid_size', [8, 8])
    rule_range = ca_config.get('rule_range', [1, 255])
    steps_range = ca_config.get('steps_range', [5, 20])
    initial_density = ca_config.get('initial_density', 0.3)
    
    # Get random seed from config
    seed_value = config.get('seed', 42)
    rng = Random(seed_value)
    
    genomes = []
    print(f"   📊 Generating {population_size} diverse genomes...")
    
    for i in range(population_size):
        # Create unique genome ID
        genome_id = f"gen0_genome{i:04d}"
        
        # Create diverse CA seed from config parameters
        np.random.seed(seed_value + i)  # Ensure diversity
        grid = np.random.rand(*grid_size) < initial_density
        grid = grid.astype(int)
        rule = rng.randint(*rule_range)
        steps = rng.randint(*steps_range)
        
        ca_seed = CASeed(grid=grid, rule=rule, steps=steps)
        
        # Run CA evolution and extract features
        from coral.domain.ca import evolve
        from coral.domain.feature_extraction import extract_features
        from coral.domain.mapping import map_features_to_lora_config
        
        ca_history = evolve(ca_seed)
        features = extract_features(ca_history)
        
        # Map features to LoRA configuration using full innovation
        lora_config = map_features_to_lora_config(features, config, 1.0, 0)  # Default diversity and index
        
        genome = Genome(seed=ca_seed, lora_cfg=lora_config, id=genome_id)
        genomes.append(genome)
        
        # Show progress for large populations
        if (i + 1) % 8 == 0 or (i + 1) == population_size:
            print(f"      Generated {i+1}/{population_size} genomes...")
    
    # Show diversity statistics
    unique_lora_configs = len(set(g.get_heavy_genes_key() for g in genomes))
    diversity_ratio = unique_lora_configs / population_size
    
    print(f"   🎯 Population diversity:")
    print(f"      • Unique LoRA configs: {unique_lora_configs}/{population_size}")
    print(f"      • Diversity ratio: {diversity_ratio:.1%}")
    
    if diversity_ratio < 0.5:
        print(f"      ⚠️  Consider increasing feature extraction sensitivity")
    else:
        print(f"      ✅ Good population diversity achieved")
    
    return Population(tuple(genomes))


def _create_cache_friendly_lora_configs(evo_config, cache_groups: int, rng: Random) -> list:
    """Create discrete LoRA config sets for cache optimization."""
    from coral.domain.mapping import LoRAConfig
    
    # Get LoRA parameters from config - using discrete candidates
    rank_candidates = evo_config.get('rank_candidates')
    alpha_candidates = evo_config.get('alpha_candidates')
    dropout_candidates = evo_config.get('dropout_candidates')
    target_modules = evo_config.get('target_modules')
    
    if not all([rank_candidates, alpha_candidates, dropout_candidates, target_modules]):
        raise ValueError("FAIL-FAST: LoRA parameters not fully specified in config")
    
    # Create strategic combinations for cache efficiency using discrete candidates
    lora_configs = []
    
    for i in range(min(cache_groups, len(rank_candidates))):
        # Select from discrete candidates instead of continuous ranges
        rank = rank_candidates[i % len(rank_candidates)]
        alpha = alpha_candidates[i % len(alpha_candidates)]
        dropout = dropout_candidates[i % len(dropout_candidates)]
        
        config = LoRAConfig(
            r=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=tuple(target_modules)
        )
        lora_configs.append(config)
    
    # Shuffle for randomness while maintaining cache groups
    rng.shuffle(lora_configs)
    
    return lora_configs


def _create_default_config_file(config_path: str) -> None:
    """Create a default configuration file."""
    default_config = create_default_config()
    
    # Ensure parent directory exists
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Write config to file
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, indent=2, default_flow_style=False)


def main():
    """Enhanced CLI entry point with subcommands."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="CoralX - Enhanced CORAL Evolution System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  coral run --config coral_x_modal_config.yaml          # Run experiment
  coral deploy                                          # Deploy to Modal
  coral cache-model --config coral_x_modal_config.yaml # Pre-download model to cache
  coral logs --follow                                   # Stream Modal logs
  coral status --config coral_x_modal_config.yaml      # Show system status
  coral create-config my_config.yaml                    # Create new config
  coral modal logs --tail 50                           # Proxy to modal CLI

Version: {get_version()}
"""
    )
    
    # Create subparsers
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run CoralX experiment')
    run_parser.add_argument(
        '--config', 
        required=True,
        help='Path to configuration file (required)'
    )
    run_parser.add_argument(
        '--stream-logs',
        action='store_true',
        help='Stream logs in real-time'
    )
    
    # Deploy command
    deploy_parser = subparsers.add_parser('deploy', help='Deploy to Modal')
    deploy_parser.add_argument(
        '--app-file',
        default='coral_modal_app.py',
        help='Modal app file to deploy (default: coral_modal_app.py)'
    )
    
    # Cache model command
    cache_parser = subparsers.add_parser('cache-model', help='Pre-download model to Modal cache')
    cache_parser.add_argument(
        '--config',
        required=True,
        help='Configuration file with model parameters (required)'
    )
    
    # Logs command
    logs_parser = subparsers.add_parser('logs', help='Stream Modal logs')
    logs_parser.add_argument(
        '--follow',
        action='store_true',
        default=True,
        help='Follow logs in real-time (default: true)'
    )
    logs_parser.add_argument(
        '--tail',
        type=int,
        default=100,
        help='Number of lines to tail (default: 100)'
    )
    
    # Apps command
    subparsers.add_parser('apps', help='List Modal applications')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show system status')
    status_parser.add_argument(
        '--config',
        help='Configuration file to check (optional)'
    )
    
    # Create config command
    config_parser = subparsers.add_parser('create-config', help='Create default configuration')
    config_parser.add_argument(
        'config_path',
        help='Path for new configuration file'
    )
    
    # Monitor command
    subparsers.add_parser('monitor', help='Monitor evolution progress in real-time')
    
    # Version command
    subparsers.add_parser('version', help='Show version')
    
    # Modal proxy command
    modal_parser = subparsers.add_parser('modal', help='Proxy to Modal CLI')
    modal_parser.add_argument(
        'modal_args',
        nargs=argparse.REMAINDER,
        help='Modal CLI arguments'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        # Dispatch commands
        if args.command == 'run':
            run_experiment(args.config, args.stream_logs)
        elif args.command == 'deploy':
            deploy_modal(args.app_file)
        elif args.command == 'cache-model':
            cache_model(args.config)
        elif args.command == 'logs':
            modal_logs(args.follow, args.tail)
        elif args.command == 'apps':
            modal_apps()
        elif args.command == 'status':
            status(getattr(args, 'config', None))
        elif args.command == 'create-config':
            create_config(args.config_path)
        elif args.command == 'monitor':
            monitor_evolution()
        elif args.command == 'version':
            print(f"CoralX version {get_version()}")
        elif args.command == 'modal':
            modal_proxy(args.modal_args)
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
    
    except Exception as e:
        if "FAIL-FAST:" in str(e):
            print(f"🚨 {e}")
            sys.exit(1)
        else:
            print(f"❌ Error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main() 