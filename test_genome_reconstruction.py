#!/usr/bin/env python3
"""Test manual genome reconstruction for Modal compatibility."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

def test_genome_reconstruction():
    """Test that we can manually reconstruct a genome from serialized data."""
    print("🧪 TESTING: Manual Genome Reconstruction")
    print("=" * 50)
    
    try:
        import yaml
        import numpy as np
        from coral.domain.experiment import create_experiment_config, create_initial_population
        from coral.domain.genome import Genome, CASeed, LoRAConfig
        from infra.modal_executor import ModalExecutor
        
        # Load config
        with open("coral_x_codellama_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # Create test genome
        exp_config = create_experiment_config(config)
        population = create_initial_population(exp_config, diversity_strength=0.5, raw_config=config, run_id="test_reconstruction")
        original_genome = population.genomes[0]
        
        print(f"✅ Original genome: {original_genome.id}")
        print(f"   • CA rule: {original_genome.seed.rule}")
        print(f"   • LoRA rank: {original_genome.lora_cfg.r}")
        print(f"   • LoRA alpha: {original_genome.lora_cfg.alpha}")
        
        # Serialize via ModalExecutor
        modal_executor = ModalExecutor(app_name="test", config=config)
        genome_data = modal_executor._serialize_genome(original_genome)
        
        print(f"\n🔄 Serialized genome data:")
        print(f"   • Keys: {list(genome_data.keys())}")
        print(f"   • Seed keys: {list(genome_data['seed'].keys())}")
        print(f"   • LoRA keys: {list(genome_data['lora_config'].keys())}")
        
        # Manually reconstruct genome (same logic as in Modal function)
        print(f"\n🔧 Reconstructing genome manually...")
        
        # Reconstruct CASeed
        seed_data = genome_data['seed']
        ca_seed = CASeed(
            grid=np.array(seed_data['grid']),
            rule=seed_data['rule'],
            steps=seed_data['steps']
        )
        
        # Reconstruct LoRAConfig
        lora_data = genome_data['lora_config']
        lora_cfg = LoRAConfig(
            r=lora_data['r'],
            alpha=lora_data['alpha'],
            dropout=lora_data['dropout'],
            target_modules=tuple(lora_data['target_modules']),
            adapter_type=lora_data.get('adapter_type', 'lora')
        )
        
        # Reconstruct Genome
        reconstructed_genome = Genome(
            seed=ca_seed,
            lora_cfg=lora_cfg,
            id=genome_data['id'],
            run_id=lora_data.get('run_id')
        )
        
        print(f"✅ Reconstructed genome: {reconstructed_genome.id}")
        print(f"   • CA rule: {reconstructed_genome.seed.rule}")
        print(f"   • LoRA rank: {reconstructed_genome.lora_cfg.r}")
        print(f"   • LoRA alpha: {reconstructed_genome.lora_cfg.alpha}")
        
        # Validate reconstruction
        errors = []
        
        if original_genome.id != reconstructed_genome.id:
            errors.append(f"ID mismatch: {original_genome.id} != {reconstructed_genome.id}")
            
        if original_genome.seed.rule != reconstructed_genome.seed.rule:
            errors.append(f"CA rule mismatch: {original_genome.seed.rule} != {reconstructed_genome.seed.rule}")
            
        if original_genome.lora_cfg.r != reconstructed_genome.lora_cfg.r:
            errors.append(f"LoRA rank mismatch: {original_genome.lora_cfg.r} != {reconstructed_genome.lora_cfg.r}")
            
        if original_genome.lora_cfg.alpha != reconstructed_genome.lora_cfg.alpha:
            errors.append(f"LoRA alpha mismatch: {original_genome.lora_cfg.alpha} != {reconstructed_genome.lora_cfg.alpha}")
            
        # Check grid equality
        if not np.array_equal(original_genome.seed.grid, reconstructed_genome.seed.grid):
            errors.append(f"CA grid mismatch")
        
        if errors:
            print(f"\n❌ RECONSTRUCTION ERRORS:")
            for error in errors:
                print(f"   • {error}")
            return False
        else:
            print(f"\n🎉 SUCCESS: Perfect reconstruction!")
            print(f"💡 Manual genome reconstruction works for Modal!")
            return True
        
    except Exception as e:
        print(f"❌ Reconstruction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_genome_reconstruction()
    if success:
        print(f"\n✅ RECONSTRUCTION WORKS! Ready to deploy fixed Modal function.")
    else:
        print(f"\n❌ Reconstruction broken - fix needed.") 