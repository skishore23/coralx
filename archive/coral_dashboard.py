#!/usr/bin/env python3
"""
CORAL-X Comprehensive Evolution Dashboard
Complete analysis of current run with all metrics, reports, and data
"""

import subprocess
import json
import os
import yaml
import glob
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

class CoralDashboard:
    """Comprehensive dashboard for CORAL-X evolution analysis"""
    
    def __init__(self, config_path: str = "coral_x_codellama_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.run_id = self.config['cache']['run_id']
        self.temp_dir = Path("./temp_dashboard")
        self.temp_dir.mkdir(exist_ok=True)
        
        # Extract all paths from config
        self.paths = self._extract_paths()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration file"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _extract_paths(self) -> Dict[str, str]:
        """Extract all report and data paths from config"""
        return {
            # Modal volume paths
            'adapters': '/cache/adapters',
            'models': '/cache/models', 
            'dataset': '/cache/quixbugs_dataset',
            'emergent_behavior': self.config['emergent_tracking']['output_dir'],
            
            # Local paths
            'output_dir': self.config['execution']['output_dir'],
            'genetic_tracking': self.config['execution']['genetic_tracking_dir'],
            'local_cache': self.config['infra']['cache_volume']['local_path'],
            
            # Volume info
            'volume_name': self.config['infra']['modal']['volume_name'],
            'app_name': self.config['infra']['modal']['app_name']
        }
    
    def run_cmd(self, cmd: str) -> str:
        """Run command and return output"""
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return result.stdout.strip() if result.returncode == 0 else ""
        except:
            return ""
    
    def get_quick_metrics(self) -> Dict[str, Any]:
        """Get key metrics quickly"""
        # Count adapters
        adapter_count = int(self.run_cmd(
            f"modal volume ls {self.paths['volume_name']} adapters | grep 'adapter_' | "
            f"cut -d'│' -f2 | sed 's/adapters\\///g' | sed 's/\\.json//g' | sort -u | wc -l"
        ) or "0")
        
        # Get emergent behavior data
        self.run_cmd(f"modal volume get {self.paths['volume_name']} emergent_behavior/progress_log.json ./temp_dashboard/progress.json 2>/dev/null")
        
        emergent_data = {'active': False, 'evaluations': 0, 'behaviors': 0, 'generation': 0}
        progress_file = self.temp_dir / "progress.json"
        if progress_file.exists():
            try:
                with open(progress_file) as f:
                    data = json.load(f)
                
                emergent_data['active'] = True
                emergent_data['behaviors'] = data.get('total_behaviors', 0)
                
                stats = data.get('generation_stats', {})
                emergent_data['evaluations'] = sum(gen.get('total_evaluations', 0) for gen in stats.values())
                emergent_data['generation'] = max(int(gen) for gen in stats.keys()) if stats else 0
            except:
                pass
        
        # Infrastructure counts
        model_count = int(self.run_cmd(f"modal volume ls {self.paths['volume_name']} models 2>/dev/null | grep -c '│.*file'") or "0")
        dataset_count = int(self.run_cmd(f"modal volume ls {self.paths['volume_name']} quixbugs_dataset 2>/dev/null | grep -c '│.*file'") or "0")
        
        return {
            'adapters': adapter_count,
            'emergent': emergent_data,
            'models': model_count,
            'dataset': dataset_count,
            'total_generations': self.config['execution']['generations'],
            'population_size': self.config['execution']['population_size']
        }
    
    def get_detailed_analysis(self) -> Dict[str, Any]:
        """Get comprehensive detailed analysis"""
        analysis = {
            'run_info': {
                'run_id': self.run_id,
                'experiment_name': self.config['experiment']['name'],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'evolution': {},
            'adapters': {},
            'emergent': {},
            'infrastructure': {},
            'configuration': {}
        }
        
        # Evolution progress
        metrics = self.get_quick_metrics()
        progress_pct = (metrics['emergent']['generation'] / metrics['total_generations']) * 100 if metrics['total_generations'] > 0 else 0
        
        analysis['evolution'] = {
            'current_generation': metrics['emergent']['generation'],
            'total_generations': metrics['total_generations'],
            'progress_percentage': progress_pct,
            'population_size': metrics['population_size'],
            'total_evaluations': metrics['emergent']['evaluations']
        }
        
        # Adapter analysis
        latest_adapter = self.run_cmd(
            f"modal volume ls {self.paths['volume_name']} adapters | grep 'adapter_' | "
            f"tail -1 | cut -d'│' -f2,3 | sed 's/adapters\\///g' | sed 's/\\.json//g'"
        )
        
        analysis['adapters'] = {
            'count': metrics['adapters'],
            'type': self.config.get('adapter_type', 'lora').upper(),
            'latest': latest_adapter,
            'parameter_space': {
                'ranks': self.config['evo']['rank_candidates'],
                'alphas': self.config['evo']['alpha_candidates'],
                'dropouts': self.config['evo']['dropout_candidates'],
                'total_combinations': len(self.config['evo']['rank_candidates']) * 
                                    len(self.config['evo']['alpha_candidates']) * 
                                    len(self.config['evo']['dropout_candidates'])
            }
        }
        
        # Emergent behavior analysis
        analysis['emergent'] = {
            'active': metrics['emergent']['active'],
            'total_behaviors': metrics['emergent']['behaviors'],
            'total_evaluations': metrics['emergent']['evaluations'],
            'detection_rate': (metrics['emergent']['behaviors'] / max(1, metrics['emergent']['evaluations'])) * 100
        }
        
        # Infrastructure
        analysis['infrastructure'] = {
            'models_cached': metrics['models'],
            'dataset_files': metrics['dataset'],
            'modal_app': self.paths['app_name'],
            'volume': self.paths['volume_name']
        }
        
        # Configuration highlights
        analysis['configuration'] = {
            'fitness_weights': self.config['evaluation']['fitness_weights'],
            'training_problems': self.config['experiment']['dataset']['training_problems'],
            'cheap_knobs_ranges': self.config['cheap_knobs'],
            'thresholds': self.config['threshold'],
            'genetic_ops': {
                'survival_rate': self.config['execution']['survival_rate'],
                'crossover_rate': self.config['execution']['crossover_rate'],
                'selection_mode': self.config['execution']['selection_mode']
            }
        }
        
        return analysis
    
    def print_dashboard(self, detailed: bool = False):
        """Print dashboard - simple or detailed"""
        
        if detailed:
            analysis = self.get_detailed_analysis()
            self._print_detailed_dashboard(analysis)
        else:
            metrics = self.get_quick_metrics()
            self._print_simple_dashboard(metrics)
    
    def _print_simple_dashboard(self, metrics: Dict[str, Any]):
        """Print simple overview dashboard"""
        print("🎯 CORAL-X EVOLUTION DASHBOARD")
        print("=" * 60)
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 🏗️ Run: {self.run_id}")
        print()
        
        # Progress
        emergent = metrics['emergent']
        progress_pct = (emergent['generation'] / metrics['total_generations']) * 100 if metrics['total_generations'] > 0 else 0
        progress_bar = "█" * int(progress_pct / 10) + "░" * (10 - int(progress_pct / 10))
        
        print(f"🧬 EVOLUTION: Gen {emergent['generation']}/{metrics['total_generations']} ({progress_pct:.1f}%) [{progress_bar}]")
        print(f"🎛️  ADAPTERS: {metrics['adapters']} trained (DoRA)")
        
        print(f"\n🌟 EMERGENT BEHAVIOR:")
        status_icon = "🟢" if emergent['active'] else "🔴"
        print(f"   Status: {status_icon} {'ACTIVE' if emergent['active'] else 'INACTIVE'}")
        print(f"   Evaluations: {emergent['evaluations']}")
        print(f"   Behaviors: {emergent['behaviors']} ({emergent['behaviors']/max(1,emergent['evaluations'])*100:.1f}%)")
        
        print(f"\n💻 INFRASTRUCTURE:")
        model_icon = "🟢" if metrics['models'] > 0 else "🔴"
        dataset_icon = "🟢" if metrics['dataset'] > 0 else "🔴"
        print(f"   Models: {model_icon} {metrics['models']} cached")
        print(f"   Dataset: {dataset_icon} {metrics['dataset']} files")
        
        print(f"\n📊 EFFICIENCY:")
        if metrics['adapters'] > 0:
            efficiency = emergent['behaviors'] / metrics['adapters']
            print(f"   Behaviors/Adapter: {efficiency:.3f}")
        if emergent['generation'] > 0:
            evals_per_gen = emergent['evaluations'] / emergent['generation']
            print(f"   Evaluations/Gen: {evals_per_gen:.1f}")
        
        print("\n" + "=" * 60)
    
    def _print_detailed_dashboard(self, analysis: Dict[str, Any]):
        """Print comprehensive detailed dashboard"""
        print("🎯 CORAL-X COMPREHENSIVE EVOLUTION DASHBOARD")
        print("=" * 80)
        info = analysis['run_info']
        print(f"📅 Generated: {info['timestamp']}")
        print(f"🏗️  Run ID: {info['run_id']}")
        print(f"🧪 Experiment: {info['experiment_name']}")
        print()
        
        # Evolution Progress
        print("🧬 EVOLUTION PROGRESS")
        print("-" * 40)
        evo = analysis['evolution']
        progress_bar = "█" * int(evo['progress_percentage'] / 5) + "░" * (20 - int(evo['progress_percentage'] / 5))
        print(f"Generation: {evo['current_generation']}/{evo['total_generations']} ({evo['progress_percentage']:.1f}%) [{progress_bar}]")
        print(f"Population: {evo['population_size']} genomes")
        print(f"Total Evaluations: {evo['total_evaluations']}")
        print()
        
        # Adapter Analysis
        print("🎛️  DORA ADAPTER ANALYSIS")
        print("-" * 40)
        adapters = analysis['adapters']
        print(f"Adapters Trained: {adapters['count']}")
        print(f"Adapter Type: {adapters['type']}")
        if adapters['latest']:
            print(f"Latest: {adapters['latest']}")
        
        space = adapters['parameter_space']
        print(f"\nParameter Space:")
        print(f"  • Ranks: {space['ranks']}")
        print(f"  • Alphas: {space['alphas']}")
        print(f"  • Dropouts: {space['dropouts']}")
        print(f"  • Total Combinations: {space['total_combinations']:,}")
        print()
        
        # Emergent Behavior
        print("🌟 EMERGENT BEHAVIOR TRACKING")
        print("-" * 40)
        emergent = analysis['emergent']
        status_icon = "🟢" if emergent['active'] else "🔴"
        print(f"Status: {status_icon} {'ACTIVE' if emergent['active'] else 'INACTIVE'}")
        print(f"Total Evaluations: {emergent['total_evaluations']}")
        print(f"Total Behaviors: {emergent['total_behaviors']}")
        print(f"Detection Rate: {emergent['detection_rate']:.1f}%")
        print()
        
        # Test Configuration
        print("🧪 TEST CONFIGURATION")
        print("-" * 40)
        config = analysis['configuration']
        problems = config['training_problems']
        print(f"Training Problems: {len(problems)}")
        print(f"Problems: {', '.join(problems[:6])}{'...' if len(problems) > 6 else ''}")
        
        print(f"\nFitness Weights:")
        for obj, weight in config['fitness_weights'].items():
            print(f"  • {obj.capitalize()}: {weight*100:.0f}%")
        
        print(f"\nGenetic Operations:")
        genetic = config['genetic_ops']
        print(f"  • Survival Rate: {genetic['survival_rate']*100:.0f}%")
        print(f"  • Crossover Rate: {genetic['crossover_rate']*100:.0f}%")
        print(f"  • Selection: {genetic['selection_mode']}")
        print()
        
        # Hyperparameters
        print("🎛️  HYPERPARAMETER CONFIGURATION")
        print("-" * 40)
        knobs = config['cheap_knobs_ranges']
        print("Cheap Knobs (CA-derived):")
        print(f"  • Temperature: {knobs['temperature_range']}")
        print(f"  • Top-K: {knobs['top_k_range']}")
        print(f"  • Top-P: {knobs['top_p_range']}")
        print(f"  • Rep. Penalty: {knobs['repetition_penalty_range']}")
        print(f"  • Max Tokens: {knobs['max_tokens_range']}")
        
        thresholds = config['thresholds']
        print(f"\nThreshold Evolution ({thresholds['schedule']}):")
        for metric in ['bugfix', 'style', 'security', 'runtime', 'syntax']:
            base = thresholds['base_thresholds'][metric]
            max_val = thresholds['max_thresholds'][metric]
            print(f"  • {metric.capitalize()}: {base} → {max_val}")
        print()
        
        # Infrastructure
        print("💻 INFRASTRUCTURE STATUS")
        print("-" * 40)
        infra = analysis['infrastructure']
        model_icon = "🟢" if infra['models_cached'] > 0 else "🔴"
        dataset_icon = "🟢" if infra['dataset_files'] > 0 else "🔴"
        print(f"Models: {model_icon} {infra['models_cached']} cached")
        print(f"Dataset: {dataset_icon} {infra['dataset_files']} files")
        print(f"Modal App: {infra['modal_app']}")
        print(f"Volume: {infra['volume']}")
        print()
        
        # Key Metrics
        print("📊 KEY METRICS")
        print("-" * 40)
        if adapters['count'] > 0:
            efficiency = emergent['total_behaviors'] / adapters['count']
            print(f"Behaviors per Adapter: {efficiency:.3f}")
        
        if evo['current_generation'] > 0:
            evals_per_gen = evo['total_evaluations'] / evo['current_generation']
            print(f"Evaluations per Generation: {evals_per_gen:.1f}")
            
            # Estimate completion
            gens_remaining = evo['total_generations'] - evo['current_generation']
            print(f"Generations Remaining: {gens_remaining}")
        
        print("✅ COMPREHENSIVE DASHBOARD COMPLETE")
        print("=" * 80)
        
        # Cleanup temp files
        try:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
        except:
            pass

def main():
    """Main function"""
    import sys
    
    dashboard = CoralDashboard()
    
    if len(sys.argv) > 1 and (sys.argv[1] == "--detailed" or sys.argv[1] == "-d"):
        dashboard.print_dashboard(detailed=True)
    else:
        dashboard.print_dashboard(detailed=False)

if __name__ == "__main__":
    main() 