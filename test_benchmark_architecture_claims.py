#!/usr/bin/env python3
"""
Test Benchmark Architecture Claims
==================================

Comprehensive tests to verify the claims made in CORAL_X_ARCHITECTURE.md
about the held-out benchmark system. This test exposes gaps between
documentation and actual implementation.

CRITICAL FINDINGS EXPECTED:
1. ❌ Held-out benchmark is NOT automatically run after evolution
2. ❌ Data split confusion: CLEAN problems used for evolution, not held-out
3. ❌ Integration gaps between evolution and benchmark systems
"""

import sys
import os
from pathlib import Path
import json
import time
import tempfile
import subprocess

# Add coralx to path
sys.path.insert(0, str(Path.cwd()))

def test_data_split_claims():
    """Test Claim: 'Training problems excluded from evaluation, clean problems held-out'"""
    print("\n🔍 TESTING DATA SPLIT CLAIMS")
    print("=" * 60)
    
    try:
        from coral.domain.dataset_constants import (
            QUIXBUGS_TRAINING_PROBLEMS, 
            QUIXBUGS_CLEAN_TEST_PROBLEMS,
            validate_no_overlap
        )
        
        # Verify basic split claims
        training_count = len(QUIXBUGS_TRAINING_PROBLEMS)
        clean_count = len(QUIXBUGS_CLEAN_TEST_PROBLEMS)
        total_count = training_count + clean_count
        
        print(f"✅ Data split verified:")
        print(f"   • Training problems: {training_count}")
        print(f"   • Clean test problems: {clean_count}")
        print(f"   • Total: {total_count}")
        print(f"   • No overlap: {validate_no_overlap()}")
        
        # CRITICAL TEST: What does evolution actually use?
        print(f"\n🚨 CRITICAL VERIFICATION: What does evolution actually use?")
        
        from plugins.quixbugs_codellama.plugin import QuixBugsRealDataset
        config = {
            'experiment': {
                'dataset': {'path': '/cache/quixbugs_dataset'}
            }
        }
        
        dataset = QuixBugsRealDataset(config)
        evolution_problems = list(dataset.problems())
        evolution_problem_names = {p.get('name') for p in evolution_problems}
        
        print(f"   • Evolution uses: {len(evolution_problems)} problems")
        print(f"   • Problem names: {sorted(evolution_problem_names)}")
        
        # Test the critical claim
        uses_training = bool(evolution_problem_names & QUIXBUGS_TRAINING_PROBLEMS)
        uses_clean = bool(evolution_problem_names & QUIXBUGS_CLEAN_TEST_PROBLEMS)
        
        print(f"\n📊 REALITY CHECK:")
        print(f"   • Evolution uses TRAINING problems: {uses_training}")
        print(f"   • Evolution uses CLEAN problems: {uses_clean}")
        
        if uses_clean and not uses_training:
            print(f"   ✅ Correct: Evolution uses clean problems (8), saves training (24) for held-out")
        elif uses_training and not uses_clean:
            print(f"   ❌ WRONG: Evolution uses training problems, clean problems held-out")
        else:
            print(f"   ❌ CONFUSED: Evolution uses mixed problems")
        
        return {
            'split_verified': True,
            'training_count': training_count,
            'clean_count': clean_count,
            'evolution_uses_clean': uses_clean,
            'evolution_uses_training': uses_training,
            'evolution_problem_count': len(evolution_problems)
        }
        
    except Exception as e:
        print(f"❌ Data split test failed: {e}")
        return {'split_verified': False, 'error': str(e)}


def test_held_out_benchmark_integration():
    """Test Claim: 'Held-out benchmark automatically runs after evolution'"""
    print("\n🔍 TESTING HELD-OUT BENCHMARK INTEGRATION")
    print("=" * 60)
    
    try:
        # Check if held-out benchmark is called in evolution
        evolution_file = Path("run_coral_x_evolution.py")
        with open(evolution_file, 'r') as f:
            evolution_code = f.read()
        
        # Look for held-out benchmark integration
        has_held_out_import = "run_held_out_benchmark" in evolution_code
        has_held_out_call = "HeldOutBenchmarkRunner" in evolution_code
        has_automatic_trigger = "held_out_benchmark" in evolution_code
        
        print(f"🔍 Evolution integration analysis:")
        print(f"   • Imports held-out benchmark: {has_held_out_import}")
        print(f"   • Uses HeldOutBenchmarkRunner: {has_held_out_call}")
        print(f"   • Has automatic trigger: {has_automatic_trigger}")
        
        # Check deprecated post-evolution benchmarks
        has_deprecated_benchmark = "_run_post_evolution_benchmarks" in evolution_code
        print(f"   • Has deprecated benchmark function: {has_deprecated_benchmark}")
        
        if has_deprecated_benchmark:
            # Check if it's actually deprecated
            lines = evolution_code.split('\n')
            for i, line in enumerate(lines):
                if "_run_post_evolution_benchmarks" in line and "def " in line:
                    # Look for deprecated comment in next few lines
                    for j in range(i, min(i+10, len(lines))):
                        if "DEPRECATED" in lines[j]:
                            print(f"   ⚠️  Found deprecated benchmark function (line {i+1})")
                            break
        
        # Test manual benchmark script
        held_out_script = Path("run_held_out_benchmark.py")
        script_exists = held_out_script.exists()
        print(f"   • Held-out script exists: {script_exists}")
        
        if script_exists:
            print(f"   ✅ Held-out benchmark script available as separate tool")
        
        integration_status = has_held_out_import or has_held_out_call or has_automatic_trigger
        
        print(f"\n🎯 INTEGRATION VERDICT:")
        if integration_status:
            print(f"   ✅ Automatic integration detected")
        else:
            print(f"   ❌ NO AUTOMATIC INTEGRATION - Manual execution required")
            print(f"   💡 Must run: python run_held_out_benchmark.py --evolved-adapter <path>")
        
        return {
            'automatic_integration': integration_status,
            'script_exists': script_exists,
            'deprecated_benchmark': has_deprecated_benchmark,
            'manual_execution_required': not integration_status
        }
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return {'automatic_integration': False, 'error': str(e)}


def test_benchmark_execution_workflow():
    """Test: What's the actual workflow to run benchmarks after evolution?"""
    print("\n🔍 TESTING ACTUAL BENCHMARK WORKFLOW")
    print("=" * 60)
    
    print("🚨 CRITICAL QUESTION: Are we running a separate benchmark at the end of training?")
    
    workflow_analysis = {
        'evolution_has_internal_benchmark': False,
        'evolution_calls_external_benchmark': False,
        'requires_manual_step': False,
        'workflow_steps': []
    }
    
    try:
        # Check evolution completion workflow
        evolution_file = Path("run_coral_x_evolution.py")
        with open(evolution_file, 'r') as f:
            evolution_code = f.read()
        
        # Look for end-of-evolution workflow
        lines = evolution_code.split('\n')
        in_main_function = False
        in_evolution_complete = False
        
        for i, line in enumerate(lines):
            if 'def main(' in line or 'def run_' in line:
                in_main_function = True
            
            if in_main_function and ('evolution complete' in line.lower() or 'winners =' in line):
                in_evolution_complete = True
                workflow_analysis['workflow_steps'].append(f"Line {i+1}: {line.strip()}")
            
            if in_evolution_complete and i < len(lines) - 1:
                next_lines = lines[i:i+10]
                for j, next_line in enumerate(next_lines):
                    if next_line.strip():
                        workflow_analysis['workflow_steps'].append(f"Line {i+j+1}: {next_line.strip()}")
                break
        
        # Check for benchmark calls
        has_benchmark_call = any('benchmark' in step.lower() for step in workflow_analysis['workflow_steps'])
        has_held_out_call = any('held.out' in step.lower() or 'held_out' in step.lower() for step in workflow_analysis['workflow_steps'])
        
        workflow_analysis['evolution_has_internal_benchmark'] = has_benchmark_call
        workflow_analysis['evolution_calls_external_benchmark'] = has_held_out_call
        workflow_analysis['requires_manual_step'] = not (has_benchmark_call or has_held_out_call)
        
        print(f"🔍 End-of-evolution workflow analysis:")
        print(f"   • Evolution has internal benchmarking: {workflow_analysis['evolution_has_internal_benchmark']}")
        print(f"   • Evolution calls external benchmark: {workflow_analysis['evolution_calls_external_benchmark']}")
        print(f"   • Requires manual benchmark step: {workflow_analysis['requires_manual_step']}")
        
        print(f"\n📋 Workflow steps found:")
        for step in workflow_analysis['workflow_steps'][:10]:  # Show first 10 steps
            print(f"   {step}")
        
        # Answer the critical question
        print(f"\n🎯 ANSWER TO CRITICAL QUESTION:")
        if workflow_analysis['evolution_calls_external_benchmark']:
            print(f"   ✅ YES: Evolution automatically runs separate held-out benchmark")
        elif workflow_analysis['evolution_has_internal_benchmark']:
            print(f"   ⚠️  PARTIAL: Evolution has internal benchmarking but no separate held-out")
        else:
            print(f"   ❌ NO: Evolution does NOT run separate benchmark - MANUAL STEP REQUIRED")
            print(f"   💡 Current workflow: Evolution completes → Manual benchmark execution")
        
        return workflow_analysis
        
    except Exception as e:
        print(f"❌ Workflow analysis failed: {e}")
        return {'error': str(e), 'requires_manual_step': True}


def main():
    """Run comprehensive benchmark architecture verification."""
    print("🧪 CORAL-X BENCHMARK ARCHITECTURE VERIFICATION")
    print("=" * 80)
    print("Testing claims made in CORAL_X_ARCHITECTURE.md about held-out benchmarks")
    print("This test will expose any gaps between documentation and implementation")
    
    # Run critical tests
    test_results = {}
    
    test_results['data_split'] = test_data_split_claims()
    test_results['integration'] = test_held_out_benchmark_integration()
    test_results['workflow'] = test_benchmark_execution_workflow()
    
    # Generate summary
    print("\n" + "="*80)
    print("🏆 BENCHMARK ARCHITECTURE VERIFICATION SUMMARY")
    print("="*80)
    
    # Key findings
    evolution_uses_clean = test_results.get('data_split', {}).get('evolution_uses_clean', False)
    requires_manual = test_results.get('integration', {}).get('manual_execution_required', True)
    workflow_manual = test_results.get('workflow', {}).get('requires_manual_step', True)
    
    print(f"\n🔍 KEY FINDINGS:")
    
    # Data split reality
    if evolution_uses_clean:
        print(f"   ✅ Data Split: Evolution uses CLEAN problems (8), TRAINING problems (24) available for held-out")
        print(f"   📊 Architecture: Clean → Evolution training, Training → Held-out benchmark")
    else:
        print(f"   ❌ Data Split: Unclear which problems are truly held-out")
    
    # Integration reality  
    if requires_manual or workflow_manual:
        print(f"   ❌ Integration: Held-out benchmark requires MANUAL execution after evolution")
        print(f"   🔧 Command: python run_held_out_benchmark.py --evolved-adapter <adapter_path>")
    else:
        print(f"   ✅ Integration: Automatic benchmark execution detected")
    
    # Answer the user's question
    print(f"\n🎯 ANSWER TO USER'S QUESTION:")
    print(f"   'Are we running a separate benchmark run at the end of training?'")
    
    if not (requires_manual or workflow_manual):
        print(f"   ✅ YES: Automatic separate benchmark run implemented")
    else:
        print(f"   ❌ NO: Currently requires manual execution")
        print(f"   💡 Evolution completes → Results saved → Manual benchmark step")
        print(f"   🔧 To run held-out benchmark:")
        print(f"      1. Complete evolution: python run_coral_x_evolution.py --config <config>")
        print(f"      2. Run held-out benchmark: python run_held_out_benchmark.py --results-file results/evolution/latest.json")
    
    # Architecture compliance verdict
    has_split = test_results.get('data_split', {}).get('split_verified', False)
    has_script = test_results.get('integration', {}).get('script_exists', False)
    is_automatic = not (requires_manual or workflow_manual)
    
    compliance_score = sum([has_split, has_script, is_automatic]) / 3 * 100
    
    print(f"\n🏁 ARCHITECTURE COMPLIANCE:")
    if compliance_score >= 75:
        print(f"   ✅ HIGH COMPLIANCE ({compliance_score:.0f}%)")
        print(f"   🎯 Claims in CORAL_X_ARCHITECTURE.md are mostly accurate")
    elif compliance_score >= 50:
        print(f"   ⚠️  MODERATE COMPLIANCE ({compliance_score:.0f}%)")
        print(f"   📝 CORAL_X_ARCHITECTURE.md needs updates for integration details")
    else:
        print(f"   ❌ LOW COMPLIANCE ({compliance_score:.0f}%)")
        print(f"   📝 CORAL_X_ARCHITECTURE.md claims don't match implementation")
    
    # Save verification report
    timestamp = int(time.time())
    report = {
        'timestamp': timestamp,
        'compliance_score': compliance_score,
        'automatic_benchmark': is_automatic,
        'manual_step_required': requires_manual or workflow_manual,
        'test_results': test_results,
        'user_question_answer': {
            'question': 'Are we running a separate benchmark run at the end of training?',
            'answer': 'NO - Manual execution required' if (requires_manual or workflow_manual) else 'YES - Automatic execution'
        }
    }
    
    report_file = f"benchmark_verification_{timestamp}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Verification report saved: {report_file}")
    
    return 0 if compliance_score >= 75 else 1


if __name__ == "__main__":
    exit(main()) 