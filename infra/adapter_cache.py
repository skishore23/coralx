###############################################################################
# Clone-Cache Helper — Cache LoRA adapters to avoid expensive retraining
# Modal-primary storage with local sync
###############################################################################
import hashlib
import os
from pathlib import Path
from typing import Dict, Tuple, Any, Callable, Optional, List
from dataclasses import dataclass
import json

# Import from coralx package structure
from core.domain.mapping import LoRAConfig


def is_modal_environment() -> bool:
    """Detect if running in Modal environment."""
    return any(var in os.environ for var in ['MODAL_TASK_ID', 'MODAL_ENVIRONMENT']) or os.path.exists('/cache')


def get_storage_paths(config: 'CacheConfig') -> Tuple[str, str]:
    """Get storage paths - Modal volume when in Modal environment."""
    if is_modal_environment():
        return "/cache/adapters", None
    else:
        return str(Path(config.artifacts_dir).resolve()), None


@dataclass(frozen=True)
class HeavyGenes:
    """Heavy genes that require adapter training."""
    rank: int
    alpha: float
    dropout: float
    target_modules: Tuple[str, ...]
    adapter_type: str = "lora"  # "lora" or "dora"
    run_id: Optional[str] = None  # Experiment-specific identifier
    
    # Note: to_key() method removed - using HeavyGenes object directly via hash
    
    def to_hash(self) -> str:
        """Generate hash for file naming - includes adapter type and run_id for separate caching."""
        # Use object attributes directly instead of tuple conversion
        key_str = f"{self.rank}_{self.alpha}_{self.dropout}_{self.target_modules}_{self.adapter_type}_{self.run_id}"
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'rank': self.rank,
            'alpha': self.alpha,
            'dropout': self.dropout,
            'target_modules': list(self.target_modules),
            'adapter_type': self.adapter_type,
            'run_id': self.run_id
        }
    
    @classmethod
    def from_lora_config(cls, lora_cfg: LoRAConfig, run_id: Optional[str] = None) -> 'HeavyGenes':
        """Extract heavy genes from adapter config (LoRA or DoRA) with optional run_id."""
        adapter_type = getattr(lora_cfg, 'adapter_type', 'lora')  # Default to LoRA for backward compatibility
        return cls(
            rank=lora_cfg.r,
            alpha=lora_cfg.alpha,
            dropout=lora_cfg.dropout,
            target_modules=lora_cfg.target_modules,
            adapter_type=adapter_type,
            run_id=run_id
        )


@dataclass(frozen=True)
class CacheConfig:
    """Configuration for adapter cache - Modal-native when possible."""
    artifacts_dir: str
    base_checkpoint: str
    cache_metadata: bool
    cleanup_threshold: int = 25  # REDUCED: Lower threshold for better storage management
    auto_sync: bool = True
    modal_mount: str = "/cache"
    modal_native: bool = True  # Prefer Modal volume for consistency
    run_id: Optional[str] = None  # Experiment-specific identifier for cache separation


def create_cache_config_from_dict(config_dict: Dict[str, Any]) -> CacheConfig:
    """Create CacheConfig from configuration dictionary."""
    if 'cache' not in config_dict:
        raise ValueError("  'cache' section missing from configuration")
    
    cache_config = config_dict['cache']
    
    # Validate required fields
    required_fields = ['artifacts_dir', 'base_checkpoint']
    for field in required_fields:
        if field not in cache_config:
            raise ValueError(f"  '{field}' missing from cache configuration")
    
    # Get volume config for auto-sync
    volume_config = config_dict.get('infra', {}).get('cache_volume', {})
    
    return CacheConfig(
        artifacts_dir=cache_config['artifacts_dir'],
        base_checkpoint=cache_config['base_checkpoint'],
        cache_metadata=cache_config.get('metadata', True),
        cleanup_threshold=cache_config.get('cleanup_threshold', 100),
        auto_sync=volume_config.get('auto_sync', True),
        modal_mount=volume_config.get('modal_mount', '/cache'),
        modal_native=cache_config.get('modal_native', True),
        run_id=cache_config.get('run_id', None)
    )


class AdapterCache:
    """Cache for trained LoRA adapters with Modal-primary storage."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        
        # Determine storage paths based on environment
        self.primary_path, self.secondary_path = get_storage_paths(config)
        self.artifacts_dir = Path(self.primary_path)
        self.artifacts_dir.mkdir(exist_ok=True, parents=True)
        
        # In-memory cache using hash strings as keys
        self._cache: Dict[str, str] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        
        print(f"📁 Adapter cache initialized:")
        print(f"   • Environment: {'Modal' if is_modal_environment() else 'Local'}")
        print(f"   • Primary storage: {self.primary_path}")
        print(f"   • Secondary storage: {self.secondary_path or 'None'}")
        print(f"   • Auto-sync: {self.config.auto_sync}")
        
        # IMMEDIATE: Clean up legacy .pt files on initialization
        self._cleanup_legacy_pt_files()
        
        # Load existing cache metadata
        self._load_cache_metadata()
    
    def get_or_train_adapter(self, 
                           heavy_genes: HeavyGenes,  # 🔥 FIX: Expect specific type, not arbitrary input
                           trainer_fn: Callable[[HeavyGenes, str], str]) -> str:
        """Get adapter path, training if not cached.
        
        Args:
            heavy_genes: HeavyGenes object (use HeavyGenes.from_lora_config() to create)
            trainer_fn: Function to train adapter if not cached
            
        Returns:
            str: Path to adapter
        """
        # Validate input type for clarity
        if not isinstance(heavy_genes, HeavyGenes):
            raise TypeError(
                f"  Expected HeavyGenes object, got {type(heavy_genes)}. "
                f"Use HeavyGenes.from_lora_config() to create from LoRAConfig."
            )
        
        # Use HeavyGenes object directly as cache key instead of tuple conversion
        cache_key = heavy_genes
        
        # Check in-memory cache first using hash key
        cache_hash = heavy_genes.to_hash()
        
        if cache_hash in self._cache:
            adapter_path = self._cache[cache_hash]
            if Path(adapter_path).exists():
                print(f"💾 [CACHE] HIT: Found in-memory cache")
                print(f"   📁 Adapter: {adapter_path}")
                return str(adapter_path)  # 🔥 FIX: Always return string for Modal consistency
            else:
                # Remove stale cache entry
                print(f"⚠️  [CACHE] STALE: In-memory entry exists but file missing")
                del self._cache[cache_hash]
        
        # Check primary storage
        adapter_path = self._get_adapter_path(heavy_genes)
        if adapter_path.exists() and adapter_path.is_dir():
            print(f"💿 [CACHE] HIT: Found in primary storage")
            print(f"   📁 Adapter: {adapter_path}")
            self._cache[cache_hash] = str(adapter_path)
            return str(adapter_path)
        
        # Modal-native only - no syncing complexity
        if is_modal_environment():
            print(f"🎯 Modal-native execution: no syncing needed")
        
        # 🔥 CRITICAL FIX: Try volume reload before declaring cache miss
        if is_modal_environment():
            print(f"🔄 [CACHE] Attempting Modal volume reload before final cache miss decision...")
            try:
                import modal
                volume = modal.Volume.from_name("coral-x-clean-cache")
                volume.reload()
                print(f"✅ Modal volume reloaded - checking cache again...")
                
                # Give filesystem a moment to reflect changes
                import time
                time.sleep(1)
                
                # Check one more time after volume reload
                if adapter_path.exists() and adapter_path.is_dir():
                    print(f"🎯 [CACHE] HIT after volume reload! Adapter found at: {adapter_path}")
                    self._cache[cache_hash] = str(adapter_path)
                    return str(adapter_path)
                else:
                    print(f"❌ [CACHE] Still not found after volume reload - confirmed miss")
                    
            except Exception as reload_error:
                print(f"⚠️  Volume reload warning during cache check: {reload_error}")
        
        # Train new adapter
        print(f"❌ [CACHE] MISS: No cached adapter found")
        print(f"🔧 [TRAINING] Starting new adapter for genes: {heavy_genes}")
        print(f"   📍 Will save to: {adapter_path}")
        trained_path = trainer_fn(heavy_genes, self.config.base_checkpoint)
        
        # Move to cache location if different
        if trained_path != str(adapter_path):
            self._move_adapter_to_cache(trained_path, adapter_path)
        
        # Update cache
        self._cache[cache_hash] = str(adapter_path)
        self._save_metadata(heavy_genes, adapter_path)
        
        # Cleanup old adapters if threshold exceeded
        self._cleanup_if_needed()
        
        return str(adapter_path)
    
    def _get_adapter_path(self, heavy_genes: HeavyGenes) -> Path:
        """Get the expected path for an adapter (directory for LoRA adapters)."""
        dirname = f"adapter_{heavy_genes.to_hash()}"
        return self.artifacts_dir / dirname
    
    def _move_adapter_to_cache(self, source_path: str, target_path: Path) -> None:
        """Move trained adapter to cache location with Modal volume sync."""
        source = Path(source_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"🔄 Moving adapter from {source} to {target_path}")
        
        if source.is_dir():
            import shutil
            if target_path.exists():
                shutil.rmtree(target_path)  # Remove existing target
            shutil.copytree(source, target_path, dirs_exist_ok=True)
            print(f"📁 Adapter directory moved successfully")
            
            # Clean up source if different location
            if source != target_path and source.exists():
                shutil.rmtree(source)
                print(f"🧹 Cleaned up temporary training directory")
        else:
            if target_path.exists():
                target_path.unlink()  # Remove existing target
            source.rename(target_path)
            print(f"📄 Adapter file moved successfully")
        
        # CRITICAL: Verify the target exists and is accessible
        if not target_path.exists():
            raise RuntimeError(f"CRITICAL: Failed to move adapter to {target_path}")
        
        # Modal-specific: Force filesystem sync for volume persistence
        if is_modal_environment():
            try:
                import os
                os.sync()  # Force filesystem sync
                print(f"💾 Modal volume sync completed")
                
                # Double-check adapter directory structure
                if target_path.is_dir():
                    adapter_files = list(target_path.glob("*"))
                    print(f"📂 Adapter contains {len(adapter_files)} files:")
                    for file in adapter_files[:3]:  # Show first 3 files
                        print(f"   • {file.name} ({file.stat().st_size} bytes)")
                    if len(adapter_files) > 3:
                        print(f"   • ... and {len(adapter_files) - 3} more files")
                        
                    # Verify critical adapter files exist
                    expected_files = ['adapter_config.json', 'adapter_model.safetensors']
                    for expected in expected_files:
                        expected_path = target_path / expected
                        if expected_path.exists():
                            print(f"   ✅ {expected} ({expected_path.stat().st_size} bytes)")
                        else:
                            # Check for alternative formats
                            alt_files = list(target_path.glob(f"{expected.split('.')[0]}.*"))
                            if alt_files:
                                print(f"   ✅ {expected} (found as {alt_files[0].name})")
                            else:
                                print(f"   ⚠️  {expected} not found")
                                
            except Exception as e:
                print(f"⚠️  Modal sync warning: {e}")
        
        print(f"✅ Adapter verified at: {target_path}")
    
    def has_adapter(self, cache_hash: str) -> bool:
        """Check if adapter exists in cache."""
        adapter_path = self.artifacts_dir / f"adapter_{cache_hash}"
        return adapter_path.exists() and adapter_path.is_dir()
    
    def get_adapter_path(self, cache_hash: str) -> str:
        """Get adapter path by cache hash."""
        adapter_path = self.artifacts_dir / f"adapter_{cache_hash}"
        return str(adapter_path)
    
    
    def _save_metadata(self, heavy_genes: HeavyGenes, adapter_path: Path) -> None:
        """Save metadata for cached adapter."""
        if not self.config.cache_metadata:
            return
        
        metadata = {
            'heavy_genes': {
                'rank': heavy_genes.rank,
                'alpha': heavy_genes.alpha,
                'dropout': heavy_genes.dropout,
                'target_modules': list(heavy_genes.target_modules),
                'adapter_type': heavy_genes.adapter_type,
                'run_id': heavy_genes.run_id
            },
            'base_checkpoint': self.config.base_checkpoint,
            'adapter_path': str(adapter_path),
            'hash': heavy_genes.to_hash(),
            'environment': 'modal' if is_modal_environment() else 'local'
        }
        
        metadata_path = adapter_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self._metadata[heavy_genes.to_hash()] = metadata
    
    def _load_cache_metadata(self) -> None:
        """Load existing cache metadata from disk."""
        if not self.config.cache_metadata:
            return
        
        for metadata_file in self.artifacts_dir.glob("*.json"):
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
                
                # Reconstruct heavy genes
                genes_data = metadata['heavy_genes']
                adapter_type = genes_data.get('adapter_type', 'lora')  # Default to LoRA for backward compatibility
                run_id = genes_data.get('run_id', None)  # Default to None for backward compatibility
                heavy_genes = HeavyGenes(
                    rank=genes_data['rank'],
                    alpha=genes_data['alpha'],
                    dropout=genes_data['dropout'],
                    target_modules=tuple(genes_data['target_modules']),
                    adapter_type=adapter_type,
                    run_id=run_id
                )
                
                # Check if adapter file exists
                adapter_path = metadata['adapter_path']
                if Path(adapter_path).exists():
                    self._cache[heavy_genes.to_hash()] = adapter_path
                    self._metadata[metadata['hash']] = metadata
                
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"⚠️  Skipping invalid metadata file {metadata_file}: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'total_adapters': len(self._cache),
            'primary_storage': self.primary_path,
            'secondary_storage': self.secondary_path,
            'environment': 'modal' if is_modal_environment() else 'local',
            'base_checkpoint': self.config.base_checkpoint,
            'cache_size_mb': sum(
                sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())
                for path in self._cache.values()
                if Path(path).exists()
            ) / (1024 * 1024)
        }
    
    def clear_cache(self) -> None:
        """Clear the adapter cache."""
        self._cache.clear()
        self._metadata.clear()
        
        # Optionally remove files (commented out for safety)
        # for adapter_file in self.artifacts_dir.glob("adapter_*.pt"):
        #     adapter_file.unlink()
        # for metadata_file in self.artifacts_dir.glob("*.json"):
        #     metadata_file.unlink()
    
    def _cleanup_if_needed(self) -> None:
        """Cleanup old adapters and legacy .pt files if cache size exceeds threshold."""
        # IMMEDIATE: Clean up legacy .pt files that are no longer compatible
        self._cleanup_legacy_pt_files()
        
        if len(self._cache) <= self.config.cleanup_threshold:
            return
        
        print(f"🧹 [CACHE] cleanup: {len(self._cache)} adapters > {self.config.cleanup_threshold} threshold")
        
        # Get all adapter directories with their modification times
        adapter_dirs = []
        for adapter_item in self.artifacts_dir.glob("adapter_*"):
            if adapter_item.exists() and adapter_item.is_dir():
                mod_time = adapter_item.stat().st_mtime
                adapter_dirs.append((mod_time, adapter_item))
        
        # Sort by modification time (oldest first)
        adapter_dirs.sort(key=lambda x: x[0])
        
        # Remove oldest adapters beyond threshold
        dirs_to_remove = len(adapter_dirs) - self.config.cleanup_threshold
        if dirs_to_remove > 0:
            for i in range(dirs_to_remove):
                _, adapter_dir = adapter_dirs[i]
                metadata_file = adapter_dir.with_suffix('.json')
                
                # Remove from in-memory cache
                for key, path in list(self._cache.items()):
                    if Path(path) == adapter_dir:
                        del self._cache[key]
                        break
                
                # Remove from metadata cache
                adapter_hash = adapter_dir.name.replace('adapter_', '')
                if adapter_hash in self._metadata:
                    del self._metadata[adapter_hash]
                
                # Remove directories and metadata
                try:
                    import shutil
                    shutil.rmtree(adapter_dir)
                    if metadata_file.exists():
                        metadata_file.unlink()
                    print(f"   🗑️  [CACHE] Removed old adapter: {adapter_dir.name}")
                except Exception as e:
                    print(f"   ⚠️  Failed to remove {adapter_dir.name}: {e}")
            
            print(f"✅ [CACHE] cleanup completed: removed {dirs_to_remove} old adapters")
    
    def _cleanup_legacy_pt_files(self) -> None:
        """Remove legacy .pt entries that conflict with new directory-based cache format."""
        legacy_entries = list(self.artifacts_dir.glob("adapter_*.pt"))
        if not legacy_entries:
            return
        
        print(f"🧹 [CACHE] Found {len(legacy_entries)} legacy .pt entries to clean up")
        
        removed_count = 0
        for pt_entry in legacy_entries:
            try:
                adapter_hash = pt_entry.stem.replace('adapter_', '')
                corresponding_dir = self.artifacts_dir / f"adapter_{adapter_hash}"
                
                if pt_entry.is_dir():
                    # This is a directory with .pt extension - VERY problematic
                    print(f"   🚨 [CACHE] Directory with .pt extension: {pt_entry.name}")
                    print(f"      This causes major confusion - renaming to proper format...")
                    
                    if corresponding_dir.exists():
                        print(f"      Target directory already exists, removing problematic .pt directory")
                        import shutil
                        shutil.rmtree(pt_entry)
                        removed_count += 1
                    else:
                        # Rename the .pt directory to proper format
                        pt_entry.rename(corresponding_dir)
                        print(f"      Renamed: {pt_entry.name} → {corresponding_dir.name}")
                        removed_count += 1
                
                elif pt_entry.is_file():
                    # This is an actual .pt file
                    if corresponding_dir.exists() and corresponding_dir.is_dir():
                        print(f"   🔄 [CACHE] Removing legacy .pt file (directory exists): {pt_entry.name}")
                        pt_entry.unlink()
                        removed_count += 1
                    else:
                        print(f"   ⚠️  [CACHE] Legacy .pt file without directory: {pt_entry.name}")
                        print(f"      Removing to prevent format conflicts...")
                        pt_entry.unlink()
                        removed_count += 1
            
            except Exception as e:
                print(f"   ❌ Failed to handle legacy entry {pt_entry.name}: {e}")
        
        if removed_count > 0:
            print(f"✅ [CACHE] Cleaned up {removed_count} legacy .pt entries")
            print(f"   💡 LoRA adapters now use consistent directory format")
    
    def force_cleanup_legacy_files(self) -> Dict[str, int]:
        """Force cleanup of all legacy .pt files and return statistics."""
        print(f"🧹 [CACHE] FORCE CLEANUP: Removing all legacy .pt files...")
        
        # Find all .pt files
        legacy_files = list(self.artifacts_dir.glob("adapter_*.pt"))
        legacy_metadata = list(self.artifacts_dir.glob("adapter_*.json"))
        
        # Also check for any .bin files (old PyTorch format)
        legacy_bin_files = list(self.artifacts_dir.glob("adapter_*.bin"))
        
        stats = {
            'pt_files_removed': 0,
            'bin_files_removed': 0,
            'orphaned_metadata_removed': 0,
            'directories_preserved': 0
        }
        
        # Remove .pt files
        for pt_file in legacy_files:
            try:
                pt_file.unlink()
                stats['pt_files_removed'] += 1
                print(f"   🗑️  Removed: {pt_file.name}")
            except Exception as e:
                print(f"   ❌ Failed to remove {pt_file.name}: {e}")
        
        # Remove .bin files
        for bin_file in legacy_bin_files:
            try:
                bin_file.unlink()
                stats['bin_files_removed'] += 1
                print(f"   🗑️  Removed: {bin_file.name}")
            except Exception as e:
                print(f"   ❌ Failed to remove {bin_file.name}: {e}")
        
        # Check for orphaned metadata files
        for metadata_file in legacy_metadata:
            adapter_hash = metadata_file.stem.replace('adapter_', '')
            adapter_dir = self.artifacts_dir / f"adapter_{adapter_hash}"
            
            if not adapter_dir.exists():
                try:
                    metadata_file.unlink()
                    stats['orphaned_metadata_removed'] += 1
                    print(f"   🗑️  Removed orphaned metadata: {metadata_file.name}")
                except Exception as e:
                    print(f"   ❌ Failed to remove {metadata_file.name}: {e}")
        
        # Count preserved directories
        adapter_dirs = list(self.artifacts_dir.glob("adapter_*"))
        stats['directories_preserved'] = len([d for d in adapter_dirs if d.is_dir()])
        
        print(f"✅ [CACHE] Force cleanup completed:")
        print(f"   • .pt files removed: {stats['pt_files_removed']}")
        print(f"   • .bin files removed: {stats['bin_files_removed']}")
        print(f"   • Orphaned metadata removed: {stats['orphaned_metadata_removed']}")
        print(f"   • Adapter directories preserved: {stats['directories_preserved']}")
        
        return stats

    def get_or_create_adapter(self, lora_config: LoRAConfig, genome_id: str = "unknown") -> str:
        """
        Get existing adapter or create new one with comprehensive debugging.
        
        ENHANCED DEBUG MODE: Detailed cache behavior analysis
        """
        heavy_genes = HeavyGenes.from_lora_config(lora_config, run_id=self.config.run_id)
        cache_hash = heavy_genes.to_hash()
        
        # Comprehensive cache debugging
        print(f"🔍 CACHE LOOKUP ANALYSIS")
        print(f"──────────────────────────────────────────────────")
        print(f"   • Genome ID: {genome_id}")
        print(f"   • LoRA Config: r={lora_config.r}, α={lora_config.alpha}, dropout={lora_config.dropout}")
        print(f"   • Heavy Genes: {heavy_genes}")
        print(f"   • Cache Hash: {cache_hash}")
        
        # Check cache status
        if self.has_adapter(cache_hash):
            # Cache HIT - detailed analysis
            adapter_path = self.get_adapter_path(cache_hash)
            print(f"   ✅ [CACHE] HIT - Reusing existing adapter")
            print(f"      • Adapter path: {adapter_path}")
            print(f"      • Cache hash: {cache_hash}")
            
            # Track cache sharing for debugging
            if not hasattr(self, '_cache_usage_tracking'):
                self._cache_usage_tracking = {}
            
            if cache_hash not in self._cache_usage_tracking:
                self._cache_usage_tracking[cache_hash] = {
                    'creation_genome': genome_id,
                    'reuse_count': 0,
                    'reuse_genomes': [],
                    'lora_config': lora_config,
                    'adapter_path': adapter_path
                }
            
            # Track this reuse
            self._cache_usage_tracking[cache_hash]['reuse_count'] += 1
            self._cache_usage_tracking[cache_hash]['reuse_genomes'].append(genome_id)
            
            usage_info = self._cache_usage_tracking[cache_hash]
            print(f"      • Total reuses: {usage_info['reuse_count']}")
            print(f"      • Shared by genomes: {usage_info['reuse_genomes']}")
            print(f"      • Original creator: {usage_info['creation_genome']}")
            
            # Cache efficiency analysis
            if usage_info['reuse_count'] > 5:
                print(f"      ⚠️  HIGH CACHE REUSE - Consider if this is expected")
            elif usage_info['reuse_count'] > 1:
                print(f"      ✅ MODERATE CACHE SHARING - Good efficiency")
            
            return adapter_path
        else:
            # Cache MISS - need to create new adapter
            print(f"   ❌ [CACHE] MISS - Creating new adapter")
            print(f"      • This adapter configuration has not been seen before")
            
            # Check for similar configurations (debugging cache granularity)
            similar_configs = self._find_similar_configs(lora_config)
            if similar_configs:
                print(f"      📊 SIMILAR CONFIGURATIONS FOUND:")
                for i, (similar_hash, similar_config) in enumerate(similar_configs[:3]):
                    print(f"         {i+1}. Hash: {similar_hash[:8]}... Config: r={similar_config.r}, α={similar_config.alpha}, dropout={similar_config.dropout}")
                print(f"      💡 Cache granularity analysis: {len(similar_configs)} similar configs exist")
            else:
                print(f"      🔬 UNIQUE CONFIGURATION - No similar configs found")
            
            # Initialize tracking for new adapter
            if not hasattr(self, '_cache_usage_tracking'):
                self._cache_usage_tracking = {}
            
            self._cache_usage_tracking[cache_hash] = {
                'creation_genome': genome_id,
                'reuse_count': 0,
                'reuse_genomes': [],
                'lora_config': lora_config,
                'adapter_path': None  # Will be set after creation
            }
            
            return None  # Signal that new adapter needs to be created

    def _find_similar_configs(self, target_config: LoRAConfig, similarity_threshold: float = 0.8) -> List[Tuple[str, LoRAConfig]]:
        """Find similar LoRA configurations for cache granularity analysis."""
        if not hasattr(self, '_cache_usage_tracking'):
            return []
        
        similar_configs = []
        
        for cache_hash, info in self._cache_usage_tracking.items():
            cached_config = info['lora_config']
            
            # Calculate similarity score
            similarity = self._calculate_config_similarity(target_config, cached_config)
            
            if similarity >= similarity_threshold:
                similar_configs.append((cache_hash, cached_config))
        
        return similar_configs

    def _calculate_config_similarity(self, config1: LoRAConfig, config2: LoRAConfig) -> float:
        """Calculate similarity between two LoRA configurations."""
        # Normalize parameters for comparison
        rank_sim = 1.0 - abs(config1.r - config2.r) / max(config1.r, config2.r)
        alpha_sim = 1.0 - abs(config1.alpha - config2.alpha) / max(config1.alpha, config2.alpha)
        dropout_sim = 1.0 - abs(config1.dropout - config2.dropout) / max(config1.dropout, config2.dropout, 0.01)
        
        # Weighted average (rank and alpha more important)
        similarity = (0.4 * rank_sim + 0.4 * alpha_sim + 0.2 * dropout_sim)
        
        return similarity

    def print_cache_usage_summary(self):
        """Print comprehensive cache usage summary for debugging."""
        if not hasattr(self, '_cache_usage_tracking'):
            print("📊 CACHE USAGE SUMMARY: No tracking data available")
            return
        
        tracking = self._cache_usage_tracking
        
        print(f"\n📊 COMPREHENSIVE CACHE USAGE SUMMARY")
        print(f"{'=' * 60}")
        print(f"   • Total cache groups: {len(tracking)}")
        
        # Calculate statistics
        total_genomes = sum(info['reuse_count'] + 1 for info in tracking.values())  # +1 for creator
        total_reuses = sum(info['reuse_count'] for info in tracking.values())
        cache_efficiency = total_genomes / len(tracking) if len(tracking) > 0 else 0
        
        print(f"   • Total genomes processed: {total_genomes}")
        print(f"   • Total cache reuses: {total_reuses}")
        print(f"   • Cache efficiency: {cache_efficiency:.1f}x reuse")
        
        # Cache efficiency assessment
        if cache_efficiency > 8:
            print(f"   ⚠️  VERY HIGH CACHE REUSE - May indicate insufficient diversity")
        elif cache_efficiency > 4:
            print(f"   ✅ HIGH CACHE EFFICIENCY - Good balance")
        elif cache_efficiency > 2:
            print(f"   ✅ MODERATE CACHE EFFICIENCY - Reasonable sharing")
        else:
            print(f"   ❌ LOW CACHE EFFICIENCY - Most genomes unique")
        
        # Detailed breakdown
        print(f"\n🔄 CACHE GROUP BREAKDOWN")
        print(f"{'─' * 60}")
        
        # Sort by reuse count
        sorted_groups = sorted(tracking.items(), key=lambda x: x[1]['reuse_count'], reverse=True)
        
        for i, (cache_hash, info) in enumerate(sorted_groups[:10]):  # Show top 10
            config = info['lora_config']
            reuse_count = info['reuse_count']
            total_genomes_in_group = reuse_count + 1  # +1 for creator
            
            print(f"   Group {i+1}: {cache_hash[:8]}... ({total_genomes_in_group} genomes)")
            print(f"      • LoRA: r={config.r}, α={config.alpha}, dropout={config.dropout}")
            print(f"      • Creator: {info['creation_genome']}")
            print(f"      • Reuses: {reuse_count}")
            
            if reuse_count > 0:
                recent_reuses = info['reuse_genomes'][-3:]  # Show last 3 reuses
                print(f"      • Recent users: {', '.join(recent_reuses)}")
            
            # Cache sharing assessment
            if reuse_count > 8:
                print(f"      ❌ EXCESSIVE SHARING - Check if this is expected")
            elif reuse_count > 3:
                print(f"      ⚠️  HIGH SHARING - Monitor diversity")
            elif reuse_count > 0:
                print(f"      ✅ MODERATE SHARING - Good efficiency")
            else:
                print(f"      🔬 UNIQUE ADAPTER - No sharing yet")
        
        # Cache diversity analysis
        print(f"\n🎯 CACHE DIVERSITY ANALYSIS")
        print(f"{'─' * 60}")
        
        # Analyze parameter distributions
        ranks = [info['lora_config'].r for info in tracking.values()]
        alphas = [info['lora_config'].alpha for info in tracking.values()]
        dropouts = [info['lora_config'].dropout for info in tracking.values()]
        
        print(f"   • Rank distribution: {sorted(set(ranks))}")
        print(f"   • Alpha distribution: {sorted(set(alphas))}")
        print(f"   • Dropout distribution: {sorted(set(dropouts))}")
        
        # Diversity metrics
        rank_diversity = len(set(ranks)) / len(ranks) if ranks else 0
        alpha_diversity = len(set(alphas)) / len(alphas) if alphas else 0
        dropout_diversity = len(set(dropouts)) / len(dropouts) if dropouts else 0
        
        print(f"   • Rank diversity: {rank_diversity:.1%}")
        print(f"   • Alpha diversity: {alpha_diversity:.1%}")
        print(f"   • Dropout diversity: {dropout_diversity:.1%}")
        
        overall_diversity = (rank_diversity + alpha_diversity + dropout_diversity) / 3
        print(f"   • Overall parameter diversity: {overall_diversity:.1%}")
        
        if overall_diversity < 0.3:
            print(f"   ❌ LOW DIVERSITY - Parameters too similar")
        elif overall_diversity < 0.6:
            print(f"   ⚠️  MODERATE DIVERSITY - Room for improvement")
        else:
            print(f"   ✅ HIGH DIVERSITY - Good parameter spread")


# Global cache instance (singleton pattern for simplicity)
_cache_instance: Optional[AdapterCache] = None


def get_adapter_cache(config: Optional[CacheConfig] = None) -> AdapterCache:
    """Get global adapter cache instance."""
    global _cache_instance
    
    if _cache_instance is None:
        if config is None:
            raise ValueError(
                "  AdapterCache requires configuration on first access. "
                "No default configuration allowed."
            )
        _cache_instance = AdapterCache(config)
    
    return _cache_instance


def get_or_train_adapter(heavy_genes: HeavyGenes,  # 🔥 FIX: Expect specific type, not arbitrary input
                        trainer_fn: Callable[[HeavyGenes, str], str],
                        cache_config: CacheConfig) -> str:
        """Convenience function for getting or training adapters.
        
        Args:
            heavy_genes: HeavyGenes object (use HeavyGenes.from_lora_config() to create)
            trainer_fn: Function to train adapter if not cached
            cache_config: Cache configuration
            
        Returns:
            str: Path to adapter
        """
        if cache_config is None:
            raise ValueError("  cache_config is required")
        
        cache = get_adapter_cache(cache_config)
        return cache.get_or_train_adapter(heavy_genes, trainer_fn) 