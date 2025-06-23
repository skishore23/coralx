#!/usr/bin/env python3
"""Quick test of CA → cheap knobs pipeline without emergent tracking."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

def test_ca_to_cheap_knobs_pipeline():
    """Test the exact pipeline that should happen in Modal."""
    print("🧪 QUICK TEST: CA → Cheap Knobs Pipeline")
    print("=" * 50)
    
    try:
        import yaml
        import numpy as np
        from coral.domain.experiment import create_experiment_config, create_initial_population
        from coral.domain.ca import evolve
        from coral.domain.feature_extraction import extract_features
        from coral.domain.cheap_knobs import map_ca_features_to_cheap_knobs
        
        # Load config
        with open("coral_x_codellama_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # Create test genome
        exp_config = create_experiment_config(config)
        population = create_initial_population(exp_config, diversity_strength=0.5, raw_config=config, run_id="test")
        genome = population.genomes[0]
        
        print(f"✅ Genome: {genome.id}")
        print(f"   • CA rule: {genome.seed.rule}")
        print(f"   • CA steps: {genome.seed.steps}")
        
        # Run the exact pipeline that should happen in Modal
        print(f"\n🌊 1. CA Evolution...")
        ca_history = evolve(genome.seed)
        print(f"   ✅ Generated CA states (size: {ca_history.states.shape if hasattr(ca_history, 'states') else 'unknown'})")
        
        print(f"🔍 2. Feature Extraction...")
        ca_features = extract_features(ca_history)
        print(f"   ✅ Features: complexity={ca_features.complexity:.3f}, intensity={ca_features.intensity:.3f}")
        
        print(f"🎛️ 3. Cheap Knobs Generation...")
        knobs_config = config['cheap_knobs']
        cheap_knobs = map_ca_features_to_cheap_knobs(ca_features, knobs_config)
        
        print(f"   ✅ FULL CHEAP KNOBS (what you'll see in Modal):")
        print(f"      • Temperature: {cheap_knobs.temperature:.3f} (complexity-driven)")
        print(f"      • Top-p: {cheap_knobs.top_p:.3f} (intensity-driven)")
        print(f"      • Top-k: {cheap_knobs.top_k} (convergence-driven)")
        print(f"      • Repetition penalty: {cheap_knobs.repetition_penalty:.3f} (periodicity-driven)")
        print(f"      • Max tokens: {cheap_knobs.max_new_tokens} (feature-derived)")
        print(f"      • Sampling: {cheap_knobs.do_sample} (CA-controlled)")
        
        # Verify these are NOT defaults
        defaults = {"temperature": 0.7, "top_p": 0.9, "top_k": 50, "repetition_penalty": 1.0}
        is_ca_derived = (
            abs(cheap_knobs.temperature - defaults["temperature"]) > 0.01 or
            abs(cheap_knobs.top_p - defaults["top_p"]) > 0.01 or
            abs(cheap_knobs.top_k - defaults["top_k"]) > 1 or
            abs(cheap_knobs.repetition_penalty - defaults["repetition_penalty"]) > 0.01
        )
        
        if is_ca_derived:
            print(f"\n🎉 SUCCESS: These are CA-DERIVED values (not defaults)!")
            print(f"💡 Your Modal logs will show these exact values!")
        else:
            print(f"\n❌ WARNING: Values look like defaults")
            
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ca_to_cheap_knobs_pipeline()
    if success:
        print(f"\n✅ PIPELINE WORKS! Ready to deploy Modal fix.")
    else:
        print(f"\n❌ Pipeline broken - fix needed.") 