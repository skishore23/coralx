#!/usr/bin/env python3
"""
CORAL-X Cache Group Analyzer
Predict how many unique LoRA adapters will be trained before starting evolution.
"""

import yaml
from pathlib import Path
from collections import defaultdict, Counter
from coral.config.loader import load_config
from coral.domain.genome import create_initial_population
from coral.domain.feature_extraction import extract_features_from_ca_seed
from coral.domain.mapping import map_features_to_lora_config

def analyze_cache_groups(config_path: str = "coral_x_codellama_config.yaml"):
    """
    Analyze the initial population and predict cache groups.
    
    Returns:
        dict: Analysis results with cache statistics
    """
    print("🔍 CORAL-X Cache Group Analyzer")
    print("=" * 60)
    
    # Load configuration
    print(f"📋 Loading config: {config_path}")
    config = load_config(config_path)
    
    # Create initial population
    population_size = config['evolution']['population_size']
    print(f"👥 Creating initial population: {population_size} genomes")
    
    population = create_initial_population(
        population_size=population_size,
        config=config
    )
    
    # Extract features and map to LoRA configs
    print("🧬 Analyzing genome → LoRA mappings...")
    heavy_genes_map = defaultdict(list)
    lora_configs = []
    
    for i, genome in enumerate(population):
        try:
            # Extract CA features
            features = extract_features_from_ca_seed(genome.ca_seed)
            
            # Map to LoRA config  
            lora_config = map_features_to_lora_config(features, config, 1.0, i)  # Use loop index
            lora_configs.append(lora_config)
            
            # Create heavy genes key (for caching)
            heavy_key = (
                lora_config.r,  # 🔥 FIXED: Use .r not .rank
                lora_config.alpha, 
                lora_config.dropout,
                tuple(sorted(lora_config.target_modules))
            )
            
            heavy_genes_map[heavy_key].append(i)
            
        except Exception as e:
            print(f"❌ Error analyzing genome {i}: {e}")
            continue
    
    # Analyze cache groups
    unique_configs = len(heavy_genes_map)
    group_sizes = [len(genomes) for genomes in heavy_genes_map.values()]
    total_genomes = sum(group_sizes)
    cache_efficiency = total_genomes / unique_configs if unique_configs > 0 else 0
    
    # Detailed statistics
    size_distribution = Counter(group_sizes)
    
    print("\n📊 CACHE GROUP ANALYSIS")
    print("=" * 60)
    print(f"👥 Total Genomes: {total_genomes}")
    print(f"🔧 Unique LoRA Configs: {unique_configs}")
    print(f"⚡ Cache Efficiency: {cache_efficiency:.1f}x reuse")
    print(f"📈 Group Sizes: {sorted(group_sizes, reverse=True)}")
    
    print(f"\n📋 SIZE DISTRIBUTION:")
    for size, count in sorted(size_distribution.items(), reverse=True):
        print(f"   • {count} groups of size {size}")
    
    # Estimate training time
    estimated_training_time_per_adapter = 8  # minutes
    total_training_time = unique_configs * estimated_training_time_per_adapter
    
    print(f"\n⏱️  TRAINING TIME ESTIMATE:")
    print(f"   • {unique_configs} adapters × {estimated_training_time_per_adapter} min = {total_training_time} min")
    print(f"   • Total estimated time: {total_training_time // 60}h {total_training_time % 60}m")
    
    # Show example LoRA configurations
    print(f"\n🔧 EXAMPLE LORA CONFIGURATIONS:")
    for i, (heavy_key, genome_indices) in enumerate(list(heavy_genes_map.items())[:5]):
        rank, alpha, dropout, modules = heavy_key
        print(f"   {i+1}. Rank={rank}, Alpha={alpha:.2f}, Dropout={dropout:.3f}")
        print(f"      Modules: {list(modules)}")
        print(f"      Used by genomes: {genome_indices} ({len(genome_indices)} genomes)")
        print()
    
    if len(heavy_genes_map) > 5:
        print(f"   ... and {len(heavy_genes_map) - 5} more configurations")
    
    return {
        'total_genomes': total_genomes,
        'unique_configs': unique_configs,
        'cache_efficiency': cache_efficiency,
        'group_sizes': group_sizes,
        'size_distribution': dict(size_distribution),
        'estimated_training_time_minutes': total_training_time,
        'heavy_genes_map': dict(heavy_genes_map),
        'lora_configs': lora_configs
    }

if __name__ == "__main__":
    import sys
    
    config_file = sys.argv[1] if len(sys.argv) > 1 else "coral_x_codellama_config.yaml"
    
    if not Path(config_file).exists():
        print(f"❌ Config file not found: {config_file}")
        print("Usage: python analyze_cache_groups.py [config_file]")
        sys.exit(1)
    
    try:
        analysis = analyze_cache_groups(config_file)        
        print("\n✅ Analysis complete!")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc() 