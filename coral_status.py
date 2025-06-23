#!/usr/bin/env python3
"""
CORAL-X Simple Status - One Command Shows Everything
Direct Modal CLI access for accurate real-time status
"""

import subprocess
import json
import os
from datetime import datetime

def run_cmd(cmd):
    """Run command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else ""
    except:
        return ""



def get_emergent_data():
    """Get real emergent behavior data"""
    # Download latest progress
    run_cmd("modal volume get coral-x-clean-cache emergent_behavior/progress_log.json ./temp_status.json 2>/dev/null")
    
    if os.path.exists('./temp_status.json'):
        try:
            with open('./temp_status.json') as f:
                data = json.load(f)
            os.remove('./temp_status.json')
            
            stats = data.get('generation_stats', {})
            total_evals = sum(gen.get('total_evaluations', 0) for gen in stats.values())
            total_behaviors = sum(gen.get('behaviors_detected', 0) for gen in stats.values())
            current_gen = max(int(gen) for gen in stats.keys()) if stats else 0
            last_update = data.get('last_updated', 'Unknown')
            
            return {
                'generation': current_gen,
                'evaluations': total_evals,
                'behaviors': total_behaviors,
                'rate': (total_behaviors / max(1, total_evals)) * 100,
                'last_update': last_update,
                'active': True
            }
        except:
            pass
    
    return {
        'generation': 0,
        'evaluations': 0,
        'behaviors': 0,
        'rate': 0.0,
        'last_update': 'No data',
        'active': False
    }

def main():
    """Show complete CORAL-X status in one display"""
    
    print("🎯 CORAL-X EVOLUTION STATUS")
    print("=" * 50)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Count unique adapters (simpler approach)
    adapter_count_cmd = "modal volume ls coral-x-clean-cache adapters | grep 'adapter_' | cut -d'│' -f2 | sed 's/adapters\\///g' | sed 's/\\.json//g' | sort -u | wc -l"
    adapter_count = int(run_cmd(adapter_count_cmd) or "0")
    
    # Get latest adapter name
    latest_cmd = "modal volume ls coral-x-clean-cache adapters | grep 'adapter_' | tail -1 | cut -d'│' -f2 | sed 's/adapters\\///g' | sed 's/\\.json//g'"
    latest_adapter = run_cmd(latest_cmd) or "None"
    
    # Simple count for models and dataset  
    model_count = int(run_cmd("modal volume ls coral-x-clean-cache models 2>/dev/null | grep -c '│.*file'") or "0")
    dataset_count = int(run_cmd("modal volume ls coral-x-clean-cache quixbugs_dataset 2>/dev/null | grep -c '│.*file'") or "0")
    
    # Get emergent behavior data
    emergent = get_emergent_data()
    
    # Calculate progress
    total_gens = 20
    progress_pct = (emergent['generation'] / total_gens) * 100 if total_gens > 0 else 0
    progress_bar = "█" * int(progress_pct / 10) + "░" * (10 - int(progress_pct / 10))
    
    # Display everything
    print(f"🧬 EVOLUTION: Gen {emergent['generation']}/{total_gens} ({progress_pct:.1f}%) [{progress_bar}]")
    print(f"🎛️  ADAPTERS: {adapter_count} trained (DoRA)")
    if latest_adapter != "None":
        short_name = latest_adapter[:30] + "..." if len(latest_adapter) > 30 else latest_adapter
        print(f"   Latest: {short_name}")
    
    print(f"\n🌟 EMERGENT BEHAVIOR:")
    status_icon = "🟢" if emergent['active'] else "🔴"
    print(f"   Status: {status_icon} {'ACTIVE' if emergent['active'] else 'INACTIVE'}")
    print(f"   Evaluations: {emergent['evaluations']}")
    print(f"   Behaviors: {emergent['behaviors']} ({emergent['rate']:.1f}%)")
    print(f"   Last Update: {emergent['last_update']}")
    
    print(f"\n💻 INFRASTRUCTURE:")
    model_icon = "🟢" if model_count > 0 else "🔴"
    dataset_icon = "🟢" if dataset_count > 0 else "🔴"
    print(f"   Models: {model_icon} {model_count} cached")
    print(f"   Dataset: {dataset_icon} {dataset_count} files")
    
    # Key metrics
    print(f"\n📊 METRICS:")
    if adapter_count > 0:
        efficiency = emergent['behaviors'] / adapter_count
        print(f"   Behaviors/Adapter: {efficiency:.3f}")
    if emergent['generation'] > 0:
        evals_per_gen = emergent['evaluations'] / emergent['generation']
        print(f"   Evaluations/Gen: {evals_per_gen:.1f}")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main() 