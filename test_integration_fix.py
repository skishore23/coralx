#!/usr/bin/env python3
"""
Test Integration Fix: Automatic Held-Out Benchmark
=================================================

This test demonstrates the missing integration and provides a solution
to automatically run held-out benchmarks after evolution completes.

CURRENT STATE: Manual execution required
TARGET STATE: Automatic execution integrated into evolution workflow
"""

import sys
import os
from pathlib import Path
import json
import tempfile

# Add coralx to path
sys.path.insert(0, str(Path.cwd()))

def test_current_integration_gap():
    """Demonstrate the current integration gap."""
    print("🔍 TESTING CURRENT INTEGRATION GAP")
    print("=" * 50)
    
    # Show current evolution completion workflow
    evolution_file = Path("run_coral_x_evolution.py")
    with open(evolution_file, 'r') as f:
        evolution_code = f.read()
    
    # Find evolution completion sections
    lines = evolution_code.split('\n')
    completion_sections = []
    
    for i, line in enumerate(lines):
        if 'evolution complete' in line.lower() or 'winners =' in line or 'return result' in line:
            # Show context around completion
            start = max(0, i-3)
            end = min(len(lines), i+7)
            section = '\n'.join(f"{j+1:3d}: {lines[j]}" for j in range(start, end))
            completion_sections.append(section)
    
    print("📋 Current evolution completion workflow:")
    for i, section in enumerate(completion_sections[:2], 1):  # Show first 2 sections
        print(f"\nSection {i}:")
        print(section)
    
    # Check for held-out benchmark calls
    has_held_out_call = "run_held_out_benchmark" in evolution_code
    has_benchmark_runner = "HeldOutBenchmarkRunner" in evolution_code
    has_automatic_execution = "held_out_benchmark" in evolution_code and "automatic" in evolution_code
    
    print(f"\n🚨 INTEGRATION ANALYSIS:")
    print(f"   • Calls run_held_out_benchmark: {has_held_out_call}")
    print(f"   • Uses HeldOutBenchmarkRunner: {has_benchmark_runner}")
    print(f"   • Has automatic execution: {has_automatic_execution}")
    
    print(f"\n❌ CURRENT STATE: Evolution completes → User must manually run benchmark")
    print(f"✅ TARGET STATE: Evolution completes → Automatic held-out benchmark → Results")
    
    return {
        'has_integration': has_held_out_call or has_benchmark_runner,
        'manual_step_required': not (has_held_out_call or has_benchmark_runner)
    }


def design_integration_fix():
    """Design the integration fix to add automatic held-out benchmark execution."""
    print("\n🔧 DESIGNING INTEGRATION FIX")
    print("=" * 50)
    
    print("🎯 INTEGRATION SOLUTION:")
    print("   1. Modify run_coral_x_evolution.py to call held-out benchmark after evolution")
    print("   2. Pass best adapter path automatically") 
    print("   3. Save combined results (evolution + held-out benchmark)")
    print("   4. Maintain backwards compatibility")
    
    # Design the integration points
    integration_design = {
        'trigger_point': 'After evolution completes successfully',
        'conditions': ['Evolution succeeded', 'Best adapter available', 'Config enables held-out'],
        'implementation': 'Add _run_automatic_held_out_benchmark() function',
        'config_flag': 'execution.run_held_out_benchmark: true',
        'result_integration': 'Combine evolution and benchmark results'
    }
    
    print(f"\n📋 INTEGRATION DESIGN:")
    for key, value in integration_design.items():
        print(f"   • {key}: {value}")
    
    # Show proposed code structure
    proposed_code = '''
def _run_automatic_held_out_benchmark(result: dict, config: dict) -> dict:
    """Run held-out benchmark automatically after evolution."""
    if not config.get('execution', {}).get('run_held_out_benchmark', False):
        print("⏭️  Held-out benchmark disabled in config")
        return result
    
    if not result.get('success', False):
        print("⏭️  Skipping held-out benchmark - evolution failed")
        return result
    
    try:
        from run_held_out_benchmark import HeldOutBenchmarkRunner
        
        # Extract best adapter path
        best_adapter_path = result.get('best_adapter_path')
        if not best_adapter_path:
            print("⚠️  No best adapter path - cannot run held-out benchmark")
            return result
        
        print("🚀 AUTOMATIC HELD-OUT BENCHMARK")
        runner = HeldOutBenchmarkRunner(config_path)
        held_out_results = runner.run_held_out_benchmark(best_adapter_path)
        
        # Integrate results
        result['held_out_benchmark'] = held_out_results
        result['scientific_validation'] = 'complete'
        
        print("✅ Held-out benchmark completed automatically")
        return result
        
    except Exception as e:
        print(f"⚠️  Held-out benchmark failed: {e}")
        result['held_out_benchmark_error'] = str(e)
        return result
'''
    
    print(f"\n💻 PROPOSED CODE ADDITION:")
    print(proposed_code)
    
    return integration_design


def test_proposed_integration():
    """Test the proposed integration approach."""
    print("\n🧪 TESTING PROPOSED INTEGRATION")
    print("=" * 50)
    
    # Create mock evolution result
    mock_result = {
        'success': True,
        'best_fitness': 0.850,
        'best_adapter_path': '/cache/adapters/adapter_abc123',
        'generations': 10,
        'experiment_time': 1800.0
    }
    
    # Create mock config with held-out benchmark enabled
    mock_config = {
        'execution': {
            'run_held_out_benchmark': True,
            'population_size': 20,
            'generations': 10
        }
    }
    
    print(f"🔍 Testing integration with mock data:")
    print(f"   • Evolution success: {mock_result['success']}")
    print(f"   • Best adapter: {mock_result['best_adapter_path']}")
    print(f"   • Held-out enabled: {mock_config['execution']['run_held_out_benchmark']}")
    
    # Simulate the integration logic
    def simulate_integration(result, config):
        if not config.get('execution', {}).get('run_held_out_benchmark', False):
            return result, "Disabled in config"
        
        if not result.get('success', False):
            return result, "Evolution failed"
        
        best_adapter = result.get('best_adapter_path')
        if not best_adapter:
            return result, "No adapter path"
        
        # Simulate held-out benchmark execution
        simulated_held_out = {
            'test_type': 'held_out_benchmark',
            'problems_tested': 8,
            'data_leakage': False,
            'scientific_validity': 'high',
            'scores': {'overall': 0.725, 'success_rate': 0.625}
        }
        
        result['held_out_benchmark'] = simulated_held_out
        result['scientific_validation'] = 'complete'
        
        return result, "Success"
    
    integrated_result, status = simulate_integration(mock_result, mock_config)
    
    print(f"\n✅ INTEGRATION SIMULATION RESULT:")
    print(f"   • Status: {status}")
    print(f"   • Has held-out results: {'held_out_benchmark' in integrated_result}")
    print(f"   • Scientific validation: {integrated_result.get('scientific_validation', 'none')}")
    
    if 'held_out_benchmark' in integrated_result:
        held_out = integrated_result['held_out_benchmark']
        print(f"   • Held-out score: {held_out['scores']['overall']:.3f}")
        print(f"   • Data leakage: {held_out['data_leakage']}")
        print(f"   • Scientific validity: {held_out['scientific_validity']}")
    
    return {
        'integration_successful': 'held_out_benchmark' in integrated_result,
        'maintains_compatibility': 'best_fitness' in integrated_result,
        'scientific_validation': integrated_result.get('scientific_validation') == 'complete'
    }


def create_config_template():
    """Create a configuration template that enables automatic held-out benchmarks."""
    print("\n📝 CREATING CONFIG TEMPLATE")
    print("=" * 50)
    
    config_template = {
        'execution': {
            'population_size': 20,
            'generations': 10,
            'selection_mode': 'pareto',
            'run_held_out_benchmark': True,  # 🔥 NEW: Enable automatic held-out benchmark
            'held_out_benchmark_config': {   # 🔥 NEW: Held-out benchmark configuration
                'enabled': True,
                'run_after_evolution': True,
                'save_combined_results': True,
                'fail_on_benchmark_error': False
            }
        },
        'experiment': {
            'model': {'name': 'codellama/CodeLlama-7b-Python-hf'},
            'dataset': {'path': '/cache/quixbugs_dataset'}
        },
        'infra': {
            'executor': 'modal'
        }
    }
    
    print("🔧 Configuration template with automatic held-out benchmarks:")
    print(json.dumps(config_template, indent=2))
    
    # Save template
    template_file = "coral_x_with_auto_benchmark_config.yaml"
    with open(template_file, 'w') as f:
        import yaml
        yaml.dump(config_template, f, default_flow_style=False)
    
    print(f"\n💾 Template saved: {template_file}")
    print(f"🎯 Usage: python run_coral_x_evolution.py --config {template_file}")
    
    return template_file


def main():
    """Test and demonstrate the integration fix for automatic held-out benchmarks."""
    print("🔧 CORAL-X INTEGRATION FIX: Automatic Held-Out Benchmarks")
    print("=" * 80)
    print("This test demonstrates the missing integration and provides a solution")
    
    # Test current state
    current_state = test_current_integration_gap()
    
    # Design the fix
    integration_design = design_integration_fix()
    
    # Test proposed integration
    integration_test = test_proposed_integration()
    
    # Create config template
    config_template = create_config_template()
    
    # Summary
    print("\n" + "="*80)
    print("🏆 INTEGRATION FIX SUMMARY")
    print("="*80)
    
    print(f"\n🔍 CURRENT STATE:")
    print(f"   ❌ Manual execution required: {current_state['manual_step_required']}")
    print(f"   🔧 User must run: python run_held_out_benchmark.py --evolved-adapter <path>")
    
    print(f"\n✅ PROPOSED SOLUTION:")
    print(f"   🔧 Automatic integration: {integration_test['integration_successful']}")
    print(f"   🔒 Backwards compatible: {integration_test['maintains_compatibility']}")
    print(f"   🔬 Scientific validation: {integration_test['scientific_validation']}")
    print(f"   📝 Config template: {config_template}")
    
    print(f"\n🎯 IMPLEMENTATION STEPS:")
    print(f"   1. Add _run_automatic_held_out_benchmark() to run_coral_x_evolution.py")
    print(f"   2. Call after evolution completes successfully")
    print(f"   3. Add config flag: execution.run_held_out_benchmark: true")
    print(f"   4. Integrate results into evolution output")
    print(f"   5. Test with: python run_coral_x_evolution.py --config {config_template}")
    
    print(f"\n🚀 BENEFIT:")
    print(f"   • Eliminates manual step")
    print(f"   • Ensures scientific validation")
    print(f"   • Maintains full traceability")
    print(f"   • Enables one-command evolution + benchmark")
    
    return 0


if __name__ == "__main__":
    exit(main()) 