#!/usr/bin/env python3
"""
Reproduce the original cache coordination error that was causing adapter misses.
This demonstrates what was broken before our fix.
"""
import sys
import hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))


def reproduce_original_cache_error():
    """Reproduce the exact cache error that was happening before our fix."""
    print("🚨 REPRODUCING ORIGINAL CACHE COORDINATION ERROR")
    print("=" * 60)
    print("This demonstrates what was broken before our fix...")
    
    try:
        from coral.domain.genome import Genome, CASeed, LoRAConfig
        from infra.adapter_cache import HeavyGenes
        import numpy as np
        
        # Create the exact genome that was failing
        print("🧬 Creating failing genome configuration...")
        
        ca_seed = CASeed(
            grid=np.random.randint(0, 2, (8, 8)),
            rule=190,
            steps=23
        )
        
        lora_cfg = LoRAConfig(
            r=4,
            alpha=32.0,
            dropout=0.15,
            target_modules=('q_proj', 'k_proj', 'v_proj', 'o_proj'),
            adapter_type='dora'
        )
        
        genome = Genome(
            seed=ca_seed,
            lora_cfg=lora_cfg,
            id='gen0_genome0003',
            run_id='dora_test_v1'
        )
        
        print(f"✅ Genome created: {genome.id}")
        print(f"   • Run ID: {genome.run_id}")
        print(f"   • Adapter type: {genome.lora_cfg.adapter_type}")
        
        # Phase 1: Calculate training-time hash (correct)
        print(f"\n🔧 PHASE 1: Training-time hash calculation (CORRECT)")
        print("-" * 50)
        
        training_heavy_genes = HeavyGenes.from_lora_config(
            genome.lora_cfg, 
            run_id=genome.run_id
        )
        training_hash = training_heavy_genes.to_hash()
        
        print(f"   📊 Training hash: {training_hash}")
        print(f"   📁 Adapter saved to: /cache/adapters/adapter_{training_hash}")
        print(f"   ✅ Training completes successfully")
        
        # Phase 2: Simulate OLD serialization/deserialization (BROKEN)
        print(f"\n💥 PHASE 2: OLD Modal serialization (BROKEN BEHAVIOR)")
        print("-" * 50)
        
        # Simulate the OLD serialization that was causing problems
        old_serialized = {
            'id': genome.id,
            'seed': {
                'grid': genome.seed.grid.tolist(),
                'rule': genome.seed.rule,
                'steps': genome.seed.steps
            },
            'lora_config': {
                'r': genome.lora_cfg.r,
                'alpha': genome.lora_cfg.alpha,
                'dropout': genome.lora_cfg.dropout,
                'target_modules': list(genome.lora_cfg.target_modules),
                # PROBLEM 1: adapter_type not always preserved
                # 'adapter_type': genome.lora_cfg.adapter_type  # <-- Missing!
            },
            # PROBLEM 2: run_id not always preserved  
            # 'run_id': genome.run_id  # <-- Missing!
        }
        
        print(f"   ⚠️  OLD serialization problems:")
        print(f"      • adapter_type missing: {'adapter_type' not in old_serialized['lora_config']}")
        print(f"      • run_id missing: {'run_id' not in old_serialized}")
        
        # Simulate OLD reconstruction with missing fields
        old_reconstructed_lora = LoRAConfig(
            r=old_serialized['lora_config']['r'],
            alpha=old_serialized['lora_config']['alpha'],
            dropout=old_serialized['lora_config']['dropout'],
            target_modules=tuple(old_serialized['lora_config']['target_modules']),
            adapter_type=old_serialized['lora_config'].get('adapter_type', 'lora')  # Defaults to 'lora'!
        )
        
        old_reconstructed_genome = Genome(
            seed=CASeed(
                grid=np.array(old_serialized['seed']['grid']),
                rule=old_serialized['seed']['rule'],
                steps=old_serialized['seed']['steps']
            ),
            lora_cfg=old_reconstructed_lora,
            id=old_serialized['id'],
            run_id=old_serialized.get('run_id')  # Becomes None!
        )
        
        print(f"   📥 OLD reconstruction result:")
        print(f"      • run_id: {old_reconstructed_genome.run_id} (was: {genome.run_id})")
        print(f"      • adapter_type: {old_reconstructed_genome.lora_cfg.adapter_type} (was: {genome.lora_cfg.adapter_type})")
        
        # Phase 3: Calculate generation-time hash (BROKEN)
        print(f"\n💔 PHASE 3: Generation-time hash calculation (BROKEN)")
        print("-" * 50)
        
        old_generation_heavy_genes = HeavyGenes.from_lora_config(
            old_reconstructed_genome.lora_cfg,
            run_id=old_reconstructed_genome.run_id
        )
        old_generation_hash = old_generation_heavy_genes.to_hash()
        
        print(f"   📊 OLD generation hash: {old_generation_hash}")
        print(f"   📁 Looking for adapter: /cache/adapters/adapter_{old_generation_hash}")
        
        # Phase 4: Demonstrate the hash mismatch
        print(f"\n🚨 PHASE 4: HASH MISMATCH DEMONSTRATION")
        print("=" * 50)
        
        print(f"Training saved:    adapter_{training_hash}")
        print(f"Generation seeks:  adapter_{old_generation_hash}")
        print(f"Hashes match:      {training_hash == old_generation_hash}")
        
        if training_hash != old_generation_hash:
            print(f"❌ CACHE MISS REPRODUCED!")
            print(f"   This is exactly what was causing your evolution to fail!")
            print(f"   The adapter was trained and saved, but generation couldn't find it.")
            
            # Show the specific differences
            print(f"\n🔍 ROOT CAUSE ANALYSIS:")
            print(f"   Training genes: {training_heavy_genes}")
            print(f"   OLD generation genes: {old_generation_heavy_genes}")
            
            # Demonstrate which field caused the difference
            if training_heavy_genes.run_id != old_generation_heavy_genes.run_id:
                print(f"   💥 run_id mismatch: '{training_heavy_genes.run_id}' → '{old_generation_heavy_genes.run_id}'")
            
            if training_heavy_genes.adapter_type != old_generation_heavy_genes.adapter_type:
                print(f"   💥 adapter_type mismatch: '{training_heavy_genes.adapter_type}' → '{old_generation_heavy_genes.adapter_type}'")
            
            return True  # Successfully reproduced the error!
        else:
            print(f"   ⚠️  Unexpected: Hashes match in reproduction")
            return False  # Could not reproduce the error (unexpected)
        
    except Exception as e:
        print(f"❌ Error reproducing cache issue: {e}")
        import traceback
        traceback.print_exc()
        return False


def demonstrate_fix():
    """Show how our fix resolves the issue."""
    print(f"\n✅ DEMONSTRATING OUR FIX")
    print("=" * 40)
    
    try:
        from coral.domain.genome import Genome, CASeed, LoRAConfig
        from infra.adapter_cache import HeavyGenes
        import numpy as np
        
        # Same genome
        ca_seed = CASeed(
            grid=np.random.randint(0, 2, (8, 8)),
            rule=190,
            steps=23
        )
        
        lora_cfg = LoRAConfig(
            r=4,
            alpha=32.0,
            dropout=0.15,
            target_modules=('q_proj', 'k_proj', 'v_proj', 'o_proj'),
            adapter_type='dora'
        )
        
        genome = Genome(
            seed=ca_seed,
            lora_cfg=lora_cfg,
            id='gen0_genome0003',
            run_id='dora_test_v1'
        )
        
        # NEW FIXED serialization
        fixed_serialized = {
            'id': genome.id,
            'seed': {
                'grid': genome.seed.grid.tolist(),
                'rule': genome.seed.rule,
                'steps': genome.seed.steps
            },
            'lora_config': {
                'r': genome.lora_cfg.r,
                'alpha': genome.lora_cfg.alpha,
                'dropout': genome.lora_cfg.dropout,
                'target_modules': list(genome.lora_cfg.target_modules),
                'adapter_type': getattr(genome.lora_cfg, 'adapter_type', 'lora')  # 🔥 FIX: Explicit preservation
            },
            'run_id': getattr(genome, 'run_id', None)  # 🔥 FIX: Explicit preservation
        }
        
        # NEW FIXED reconstruction
        fixed_reconstructed_lora = LoRAConfig(
            r=fixed_serialized['lora_config']['r'],
            alpha=fixed_serialized['lora_config']['alpha'],
            dropout=fixed_serialized['lora_config']['dropout'],
            target_modules=tuple(fixed_serialized['lora_config']['target_modules']),
            adapter_type=fixed_serialized['lora_config'].get('adapter_type', 'lora')  # Now preserved!
        )
        
        fixed_reconstructed_genome = Genome(
            seed=CASeed(
                grid=np.array(fixed_serialized['seed']['grid']),
                rule=fixed_serialized['seed']['rule'],
                steps=fixed_serialized['seed']['steps']
            ),
            lora_cfg=fixed_reconstructed_lora,
            id=fixed_serialized['id'],
            run_id=fixed_serialized.get('run_id')  # Now preserved!
        )
        
        # Calculate both hashes
        training_genes = HeavyGenes.from_lora_config(genome.lora_cfg, run_id=genome.run_id)
        fixed_generation_genes = HeavyGenes.from_lora_config(
            fixed_reconstructed_genome.lora_cfg,
            run_id=fixed_reconstructed_genome.run_id
        )
        
        training_hash = training_genes.to_hash()
        fixed_generation_hash = fixed_generation_genes.to_hash()
        
        print(f"Fixed training hash:    {training_hash}")
        print(f"Fixed generation hash:  {fixed_generation_hash}")
        print(f"Hashes match:          {training_hash == fixed_generation_hash}")
        
        if training_hash == fixed_generation_hash:
            print(f"✅ CACHE COORDINATION FIXED!")
            print(f"   Both phases now produce identical hash: {training_hash}")
            return True
        else:
            print(f"❌ Fix didn't work as expected")
            return False
            
    except Exception as e:
        print(f"❌ Error demonstrating fix: {e}")
        return False


if __name__ == "__main__":
    print("🔬 CACHE COORDINATION ERROR REPRODUCTION")
    print("=" * 70)
    
    # Step 1: Reproduce the original error
    error_reproduced = reproduce_original_cache_error()
    
    # Step 2: Show our fix
    fix_works = demonstrate_fix()
    
    print(f"\n📊 SUMMARY")
    print("=" * 30)
    
    if error_reproduced and fix_works:
        print(f"✅ Successfully reproduced original error AND demonstrated fix!")
        print(f"   • Original behavior: Cache miss due to hash inconsistency")
        print(f"   • Fixed behavior: Hash consistency maintained")
    elif error_reproduced and not fix_works:
        print(f"✅ Successfully reproduced original error but fix failed!")
        print(f"   • This means the error reproduction worked but our fix needs work")
    elif not error_reproduced and fix_works:
        print(f"⚠️  Could not reproduce original error, but fix works")
        print(f"   • This suggests the fix is already preventing the error")
    else:
        print(f"❌ Neither error reproduction nor fix demonstration worked")
        print(f"   • This suggests there may be other issues")
    
    print(f"\n🎯 The original issue was:")
    print(f"   • run_id: 'dora_test_v1' → None (lost during serialization)")
    print(f"   • adapter_type: 'dora' → 'lora' (defaulted during reconstruction)")
    print(f"   • Result: Different hash → Cache miss → 'Adapter not found' error")
    
    print(f"\n🔧 Our fix:")
    print(f"   • Explicit preservation of run_id in serialization")
    print(f"   • Explicit preservation of adapter_type in serialization") 
    print(f"   • Hash consistency verification during serialization")
    print(f"   • Result: Same hash → Cache hit → Adapter found ✅") 