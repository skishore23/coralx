 #!/usr/bin/env python3
"""
CORAL-X Evolution Comprehensive Monitor
Explores all reporting paths and constructs real-time evolution picture
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import yaml

class CoralEvolutionMonitor:
    """Comprehensive monitor for CORAL-X evolution progress"""
    
    def __init__(self, config_path: str = "coral_x_codellama_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.modal_volume = self.config['infra']['modal']['volume_name']
        self.temp_dir = Path("./temp_monitor")
        self.temp_dir.mkdir(exist_ok=True)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration file"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _modal_get(self, remote_path: str, local_path: str) -> bool:
        """Download file from Modal volume"""
        try:
            cmd = f"modal volume get {self.modal_volume} {remote_path} {local_path}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return result.returncode == 0
        except Exception as e:
            print(f"⚠️  Modal get failed: {e}")
            return False
    
    def _modal_ls(self, path: str = "") -> List[Dict[str, str]]:
        """List Modal volume contents with timestamps"""
        try:
            cmd = f"modal volume ls {self.modal_volume} {path}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                # Parse the table output with timestamps
                lines = result.stdout.strip().split('\n')
                files = []
                for line in lines:
                    if '│' in line and not line.startswith('┏') and not line.startswith('┡') and not line.startswith('└'):
                        parts = [p.strip() for p in line.split('│')]
                        if len(parts) >= 4 and parts[1] and parts[1] != 'Filename':
                            files.append({
                                'name': parts[1],
                                'type': parts[2],
                                'modified': parts[3],
                                'size': parts[4] if len(parts) > 4 else 'Unknown'
                            })
                return files
            return []
        except Exception as e:
            print(f"⚠️  Modal ls failed: {e}")
            return []
    
    def get_modal_volume_status(self) -> Dict[str, Any]:
        """Get current run Modal volume status with timestamps"""
        
        status = {
            "adapters": {"count": 0, "latest": None, "latest_time": None, "current_run": []},
            "models": {"cached": False, "count": 0},
            "dataset": {"cached": False, "count": 0},
            "emergent_behavior": {"active": False, "latest_update": None, "stats": {}},
            "run_id": self.config['cache']['run_id']
        }
        
        # Check adapters - focus on current run
        adapter_files = self._modal_ls("adapters")
        current_run_adapters = [f for f in adapter_files if f['name'].startswith("adapter_")]
        status["adapters"]["count"] = len(current_run_adapters)
        status["adapters"]["current_run"] = current_run_adapters[-5:]  # Last 5 adapters
        
        if current_run_adapters:
            # Find latest adapter by name/timestamp
            latest = max(current_run_adapters, key=lambda x: x['modified'])
            status["adapters"]["latest"] = latest['name']
            status["adapters"]["latest_time"] = latest['modified']
        
        # Check emergent behavior - current run focus
        emergent_files = self._modal_ls("emergent_behavior")
        progress_file = next((f for f in emergent_files if f['name'] == "progress_log.json"), None)
        
        if progress_file:
            status["emergent_behavior"]["active"] = True
            status["emergent_behavior"]["last_modified"] = progress_file['modified']
            
            # Download and analyze current progress
            local_progress = self.temp_dir / "progress_log.json"
            if self._modal_get("emergent_behavior/progress_log.json", str(local_progress)):
                try:
                    with open(local_progress) as f:
                        progress_data = json.load(f)
                    status["emergent_behavior"]["latest_update"] = progress_data.get("last_updated")
                    status["emergent_behavior"]["stats"] = progress_data.get("generation_stats", {})
                    status["emergent_behavior"]["total_behaviors"] = progress_data.get("total_behaviors", 0)
                except Exception as e:
                    pass  # Silent fail for quick status
        
        # Quick check models and dataset
        model_files = self._modal_ls("models")
        status["models"]["cached"] = len(model_files) > 0
        status["models"]["count"] = len(model_files)
        
        dataset_files = self._modal_ls("quixbugs_dataset")
        status["dataset"]["cached"] = len(dataset_files) > 0
        status["dataset"]["count"] = len(dataset_files)
        
        return status
    
    def get_local_status(self) -> Dict[str, Any]:
        """Get local file system status"""
        
        local_cache = Path(self.config['infra']['cache_volume']['local_path'])
        genetic_tracking = Path(self.config['execution']['genetic_tracking_dir'])
        output_dir = Path(self.config['execution']['output_dir'])
        
        status = {
            "local_cache": {"exists": local_cache.exists(), "size": 0, "subdirs": []},
            "genetic_tracking": {"exists": genetic_tracking.exists(), "files": []},
            "output_dir": {"exists": output_dir.exists(), "latest_files": []}
        }
        
        # Analyze local cache
        if local_cache.exists():
            try:
                status["local_cache"]["subdirs"] = [d.name for d in local_cache.iterdir() if d.is_dir()]
                # Get total size (approximate)
                total_size = sum(f.stat().st_size for f in local_cache.rglob('*') if f.is_file())
                status["local_cache"]["size"] = total_size
            except Exception as e:
                print(f"⚠️  Error analyzing local cache: {e}")
        
        # Analyze genetic tracking
        if genetic_tracking.exists():
            try:
                status["genetic_tracking"]["files"] = [f.name for f in genetic_tracking.iterdir() if f.is_file()]
            except Exception as e:
                print(f"⚠️  Error analyzing genetic tracking: {e}")
        
        # Analyze output directory
        if output_dir.exists():
            try:
                all_files = list(output_dir.rglob('*'))
                status["output_dir"]["latest_files"] = [
                    str(f.relative_to(output_dir)) for f in sorted(all_files, key=lambda x: x.stat().st_mtime)[-10:]
                    if f.is_file()
                ]
            except Exception as e:
                print(f"⚠️  Error analyzing output dir: {e}")
        
        return status
    
    def get_evolution_progress(self) -> Dict[str, Any]:
        """Calculate evolution progress from available data"""
        
        config_evo = self.config['execution']
        progress = {
            "configured": {
                "generations": config_evo['generations'],
                "population_size": config_evo['population_size'],
                "selection_mode": config_evo['selection_mode']
            },
            "current": {
                "generation": config_evo.get('current_generation', 0),
                "completion_pct": 0.0
            },
            "genetic_operations": {
                "survival_rate": config_evo['survival_rate'],
                "crossover_rate": config_evo['crossover_rate']
            }
        }
        
        # Calculate completion percentage
        if progress["configured"]["generations"] > 0:
            progress["current"]["completion_pct"] = (
                progress["current"]["generation"] / progress["configured"]["generations"] * 100
            )
        
        return progress
    
    def get_adapter_analysis(self) -> Dict[str, Any]:
        """Analyze adapter configuration and status"""
        
        evo_config = self.config['evo']
        analysis = {
            "type": self.config.get('adapter_type', 'lora'),
            "parameter_space": {
                "rank_candidates": evo_config['rank_candidates'],
                "alpha_candidates": evo_config['alpha_candidates'],
                "dropout_candidates": evo_config['dropout_candidates'],
                "target_modules": evo_config['target_modules'],
                "total_combinations": len(evo_config['rank_candidates']) * 
                                    len(evo_config['alpha_candidates']) * 
                                    len(evo_config['dropout_candidates'])
            },
            "training_config": self.config['training']
        }
        
        return analysis
    
    def get_fitness_analysis(self) -> Dict[str, Any]:
        """Analyze fitness configuration"""
        
        eval_config = self.config['evaluation']
        analysis = {
            "objectives": eval_config['fitness_weights'],
            "thresholds": self.config['threshold'],
            "adaptive_testing": eval_config.get('adaptive_testing', {})
        }
        
        return analysis
    
    def print_comprehensive_report(self):
        """Print comprehensive evolution status report"""
        print("🎯 CORAL-X EVOLUTION COMPREHENSIVE STATUS REPORT")
        print("=" * 80)
        print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔧 Config: {self.config_path}")
        print(f"🏗️  Experiment: {self.config['experiment']['name']}")
        
        # Modal Volume Status
        print(f"\n🌐 MODAL VOLUME STATUS ({self.modal_volume})")
        print("-" * 50)
        print("🔍 Analyzing Modal Volume Status...")
        modal_status = self.get_modal_volume_status()
        
        print(f"📦 Adapters: {modal_status['adapters']['count']} cached")
        if modal_status['adapters']['latest']:
            print(f"   Latest: {modal_status['adapters']['latest']}")
        
        print(f"🧠 Models: {'✅ Cached' if modal_status['models']['cached'] else '❌ Not cached'}")
        print(f"📚 Dataset: {'✅ Cached' if modal_status['dataset']['cached'] else '❌ Not cached'}")
        
        # Emergent Behavior Status
        print(f"\n🌟 EMERGENT BEHAVIOR TRACKING")
        print("-" * 50)
        emergent_status = modal_status['emergent_behavior']
        if emergent_status['latest_update']:
            print(f"📊 Last Update: {emergent_status['latest_update']}")
            
            if 'stats' in emergent_status:
                stats = emergent_status['stats']
                total_behaviors = sum(gen_stats.get('behaviors_detected', 0) for gen_stats in stats.values())
                total_evaluations = sum(gen_stats.get('total_evaluations', 0) for gen_stats in stats.values())
                
                print(f"🎯 Total Behaviors: {total_behaviors}")
                print(f"📈 Total Evaluations: {total_evaluations}")
                if total_evaluations > 0:
                    print(f"📊 Detection Rate: {total_behaviors/total_evaluations*100:.1f}%")
                
                # Show per-generation stats
                for gen, gen_stats in stats.items():
                    behaviors = gen_stats.get('behaviors_detected', 0)
                    evaluations = gen_stats.get('total_evaluations', 0)
                    if evaluations > 0:
                        rate = behaviors/evaluations*100
                        print(f"   Gen {gen}: {behaviors}/{evaluations} ({rate:.1f}%)")
        else:
            print("❌ No emergent behavior data found")
        
        # Local Status
        print(f"\n💾 LOCAL FILE SYSTEM STATUS")
        print("-" * 50)
        print("🔍 Analyzing Local File Status...")
        local_status = self.get_local_status()
        
        cache_status = local_status['local_cache']
        print(f"🗂️  Local Cache: {'✅ Exists' if cache_status['exists'] else '❌ Missing'}")
        if cache_status['exists']:
            size_mb = cache_status['size'] / (1024*1024)
            print(f"   Size: {size_mb:.1f} MB")
            print(f"   Subdirs: {', '.join(cache_status['subdirs'][:5])}")
        
        genetic_status = local_status['genetic_tracking']
        print(f"🧬 Genetic Tracking: {'✅ Active' if genetic_status['exists'] else '❌ Missing'}")
        if genetic_status['files']:
            print(f"   Files: {len(genetic_status['files'])} tracking files")
        
        # Evolution Progress
        print(f"\n🧬 EVOLUTION PROGRESS")
        print("-" * 50)
        print("🔍 Calculating Evolution Progress...")
        progress = self.get_evolution_progress()
        
        current_gen = progress['current']['generation']
        total_gens = progress['configured']['generations']
        completion = progress['current']['completion_pct']
        
        print(f"📊 Generation: {current_gen}/{total_gens} ({completion:.1f}% complete)")
        print(f"👥 Population: {progress['configured']['population_size']}")
        print(f"🎯 Selection: {progress['configured']['selection_mode']}")
        print(f"🧬 Survival Rate: {progress['genetic_operations']['survival_rate']*100:.0f}%")
        print(f"🔄 Crossover Rate: {progress['genetic_operations']['crossover_rate']*100:.0f}%")
        
        # Adapter Analysis
        print(f"\n🔧 ADAPTER CONFIGURATION")
        print("-" * 50)
        print("🔍 Analyzing Adapter Configuration...")
        adapter_analysis = self.get_adapter_analysis()
        
        print(f"🎛️  Type: {adapter_analysis['type'].upper()}")
        space = adapter_analysis['parameter_space']
        print(f"📊 Parameter Space: {space['total_combinations']:,} combinations")
        print(f"   • Ranks: {space['rank_candidates']}")
        print(f"   • Alphas: {space['alpha_candidates']}")
        print(f"   • Dropouts: {space['dropout_candidates']}")
        print(f"   • Modules: {', '.join(space['target_modules'])}")
        
        # Fitness Analysis
        print(f"\n🎯 FITNESS OBJECTIVES")
        print("-" * 50)
        print("🔍 Analyzing Fitness Configuration...")
        fitness_analysis = self.get_fitness_analysis()
        
        objectives = fitness_analysis['objectives']
        print("📊 Objective Weights:")
        for obj, weight in objectives.items():
            print(f"   • {obj.capitalize()}: {weight*100:.0f}%")
        
        # Training Configuration
        print(f"\n🏋️  TRAINING CONFIGURATION")
        print("-" * 50)
        training = adapter_analysis['training_config']
        print(f"⚡ Learning Rate: {training['learning_rate']}")
        print(f"📦 Batch Size: {training['batch_size']}")
        print(f"🔄 Epochs: {training['epochs']}")
        print(f"🎯 Gradient Accumulation: {training['gradient_accumulation_steps']}")
        
        # Cheap Knobs (Creativity System)
        print(f"\n🎨 CREATIVITY SYSTEM (CHEAP KNOBS)")
        print("-" * 50)
        cheap_knobs = self.config['cheap_knobs']
        print(f"🌡️  Temperature: {cheap_knobs['temperature_range']}")
        print(f"🎯 Top-K: {cheap_knobs['top_k_range']}")
        print(f"🎪 Top-P: {cheap_knobs['top_p_range']}")
        print(f"🔁 Repetition Penalty: {cheap_knobs['repetition_penalty_range']}")
        print(f"📝 Max Tokens: {cheap_knobs['max_tokens_range']}")
        
        # Resource Configuration
        print(f"\n💻 RESOURCE CONFIGURATION")
        print("-" * 50)
        modal_config = self.config['infra']['modal']
        print(f"🏗️  App: {modal_config['app_name']}")
        print(f"💾 Volume: {modal_config['volume_name']}")
        
        functions = modal_config['functions']
        print("🔧 Function Resources:")
        for func_name, func_config in functions.items():
            gpu = func_config.get('gpu', 'CPU')
            memory = func_config.get('memory', 0)
            timeout = func_config.get('timeout', 0)
            print(f"   • {func_name}: {gpu}, {memory}MB RAM, {timeout}s timeout")
        
        print(f"\n✅ COMPREHENSIVE REPORT COMPLETE")
        print("=" * 80)
    
    def command_center(self, refresh_interval: int = 10):
        """Display command center style dashboard for current run"""
        
        def clear_screen():
            os.system('cls' if os.name == 'nt' else 'clear')
        
        def get_current_stats():
            """Get key metrics for current run"""
            modal_status = self.get_modal_volume_status()
            local_status = self.get_local_status()
            
            # Calculate current generation from emergent behavior stats
            current_gen = 0
            total_evaluations = 0
            total_behaviors = 0
            latest_activity = "No activity"
            
            if modal_status['emergent_behavior']['stats']:
                stats = modal_status['emergent_behavior']['stats']
                if stats:
                    current_gen = max(int(gen) for gen in stats.keys())
                    total_evaluations = sum(gen_stats.get('total_evaluations', 0) for gen_stats in stats.values())
                    total_behaviors = sum(gen_stats.get('behaviors_detected', 0) for gen_stats in stats.values())
                    latest_activity = modal_status['emergent_behavior'].get('latest_update', 'Unknown')
            
            return {
                'run_id': modal_status['run_id'],
                'current_generation': current_gen,
                'total_generations': self.config['execution']['generations'],
                'population_size': self.config['execution']['population_size'],
                'adapters_trained': modal_status['adapters']['count'],
                'latest_adapter': modal_status['adapters']['latest'],
                'latest_adapter_time': modal_status['adapters'].get('latest_time', 'Unknown'),
                'total_evaluations': total_evaluations,
                'emergent_behaviors': total_behaviors,
                'detection_rate': (total_behaviors / max(1, total_evaluations)) * 100,
                'latest_activity': latest_activity,
                'adapter_type': self.config.get('adapter_type', 'lora').upper(),
                'models_cached': modal_status['models']['count'],
                'dataset_cached': modal_status['dataset']['count'],
                'emergent_active': modal_status['emergent_behavior']['active']
            }
        
        def print_dashboard(stats):
            """Print compact command center dashboard"""
            print("🎯 CORAL-X EVOLUTION COMMAND CENTER")
            print("=" * 60)
            print(f"🏗️  RUN ID: {stats['run_id']} | 📅 {datetime.now().strftime('%H:%M:%S')}")
            print("=" * 60)
            
            # Evolution Progress
            progress_pct = (stats['current_generation'] / max(1, stats['total_generations'])) * 100
            progress_bar = "█" * int(progress_pct / 5) + "░" * (20 - int(progress_pct / 5))
            
            print(f"🧬 EVOLUTION PROGRESS")
            print(f"   Generation: {stats['current_generation']:2d}/{stats['total_generations']:2d} ({progress_pct:5.1f}%) [{progress_bar}]")
            print(f"   Population: {stats['population_size']:2d} genomes | Type: {stats['adapter_type']}")
            
            # Training Status
            print(f"\n🏋️  TRAINING STATUS")
            print(f"   Adapters Trained: {stats['adapters_trained']:3d}")
            if stats['latest_adapter']:
                adapter_short = stats['latest_adapter'][:20] + "..." if len(stats['latest_adapter']) > 20 else stats['latest_adapter']
                print(f"   Latest: {adapter_short}")
                print(f"   Time: {stats['latest_adapter_time']}")
            else:
                print(f"   Latest: No adapters found")
            
            # Emergent Behavior Tracking
            print(f"\n🌟 EMERGENT BEHAVIOR")
            status_icon = "🟢" if stats['emergent_active'] else "🔴"
            print(f"   Status: {status_icon} {'ACTIVE' if stats['emergent_active'] else 'INACTIVE'}")
            print(f"   Evaluations: {stats['total_evaluations']:4d}")
            print(f"   Behaviors: {stats['emergent_behaviors']:3d} ({stats['detection_rate']:5.1f}%)")
            print(f"   Last Update: {stats['latest_activity']}")
            
            # Infrastructure Status
            print(f"\n💻 INFRASTRUCTURE")
            model_status = "🟢" if stats['models_cached'] > 0 else "🔴"
            dataset_status = "🟢" if stats['dataset_cached'] > 0 else "🔴"
            print(f"   Models: {model_status} {stats['models_cached']} cached")
            print(f"   Dataset: {dataset_status} {stats['dataset_cached']} files")
            
            # Key Metrics Summary
            print(f"\n📊 KEY METRICS")
            efficiency = stats['emergent_behaviors'] / max(1, stats['adapters_trained'])
            print(f"   Behaviors/Adapter: {efficiency:.2f}")
            if stats['current_generation'] > 0:
                gen_rate = stats['total_evaluations'] / max(1, stats['current_generation'])
                print(f"   Evaluations/Gen: {gen_rate:.1f}")
            print(f"   Cache Hit Rate: Calculating...")
            
            print("=" * 60)
            print(f"🔄 Auto-refresh: {refresh_interval}s | Press Ctrl+C to exit")
        
        # Start command center
        try:
            while True:
                clear_screen()
                stats = get_current_stats()
                print_dashboard(stats)
                time.sleep(refresh_interval)
        except KeyboardInterrupt:
            print(f"\n🛑 Command center stopped")
    
    def monitor_live(self, interval: int = 30):
        """Monitor evolution progress live"""
        print(f"🔴 LIVE MONITORING (updates every {interval}s)")
        print("Press Ctrl+C to stop")
        print("-" * 50)
        
        try:
            while True:
                # Get quick status
                modal_status = self.get_modal_volume_status()
                emergent_status = modal_status['emergent_behavior']
                
                timestamp = datetime.now().strftime('%H:%M:%S')
                print(f"\r[{timestamp}] Adapters: {modal_status['adapters']['count']} | "
                      f"Emergent: {emergent_status.get('latest_update', 'No data')}", end="")
                
                time.sleep(interval)
        except KeyboardInterrupt:
            print(f"\n🛑 Live monitoring stopped")

def main():
    """Main function"""
    monitor = CoralEvolutionMonitor()
    
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--command-center" or arg == "-c":
            refresh_rate = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            monitor.command_center(refresh_rate)
        elif arg == "--live" or arg == "-l":
            monitor.monitor_live()
        elif arg == "--full" or arg == "-f":
            monitor.print_comprehensive_report()
        elif arg == "--help" or arg == "-h":
            print("🎯 CORAL-X Evolution Monitor")
            print("Usage:")
            print("  python coral_evolution_monitor.py [OPTIONS]")
            print()
            print("Options:")
            print("  -c, --command-center [SECONDS]  Real-time command center (default: 10s)")
            print("  -l, --live                      Simple live monitoring")  
            print("  -f, --full                      Full comprehensive report")
            print("  -h, --help                      Show this help")
            print()
            print("Examples:")
            print("  python coral_evolution_monitor.py              # Command center (default)")
            print("  python coral_evolution_monitor.py -c 5         # Command center, 5s refresh")
            print("  python coral_evolution_monitor.py -f           # Full report")
        else:
            print(f"Unknown option: {arg}")
            print("Use --help for usage information")
    else:
        # Default to command center
        monitor.command_center()

if __name__ == "__main__":
    main()