###############################################################################
# Real QuixBugs + CodeLlama Plugin for CORAL-X
# Config-driven, NO FALLBACKS - fail-fast principle
###############################################################################
import json
import os
import numpy as np
from pathlib import Path
from typing import Iterable, Dict, Any, Callable, List, Optional
from dataclasses import dataclass
import time

# Import from clean coralx package structure
from coral.ports.interfaces import DatasetProvider, ModelRunner, FitnessFn
from coral.domain.mapping import LoRAConfig
from coral.domain.genome import Genome, MultiObjectiveScores
from coral.domain.quixbugs_evaluation import evaluate_quixbugs_code, EvaluationResults
from infra.adapter_cache import HeavyGenes, CacheConfig, get_or_train_adapter
from adapters.quixbugs_real import QuixBugsRealAdapter

# Import simple emergent behavior tracking
from coral.domain.emergent_behavior_integration import SimpleEmergentTracker


@dataclass
class CAFeatures:
    """CA features extracted from cellular automata evolution."""
    complexity: float
    intensity: float  
    periodicity: float
    convergence: float


def extract_ca_features(history: List[np.ndarray]) -> CAFeatures:
    """Extract CA features from evolution history."""
    if not history:
        return CAFeatures(complexity=0.0, intensity=0.0, periodicity=0.0, convergence=0.0)
    
    # Convert to numpy array for analysis
    states = np.array(history)
    
    # 1. Complexity - entropy-based measure
    complexity = 0.0
    for state in states:
        flat_state = state.flatten()
        if len(flat_state) > 0:
            unique, counts = np.unique(flat_state, return_counts=True)
            probs = counts / len(flat_state)
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            complexity += entropy
    complexity /= len(states)
    
    # 2. Intensity - average density of living cells
    intensity = np.mean([np.mean(state) for state in states])
    
    # 3. Periodicity - detect repeating patterns
    periodicity = 0.0
    if len(states) > 3:
        for period in range(1, min(10, len(states) // 2)):
            matches = 0
            for i in range(len(states) - period):
                if np.array_equal(states[i], states[i + period]):
                    matches += 1
            periodicity = max(periodicity, matches / (len(states) - period))
    
    # 4. Convergence - stability measure
    convergence = 0.0
    if len(states) > 1:
        differences = []
        for i in range(1, len(states)):
            diff = np.sum(np.abs(states[i] - states[i-1]))
            differences.append(diff)
        convergence = 1.0 / (1.0 + np.mean(differences))
    
    return CAFeatures(
        complexity=float(complexity),
        intensity=float(intensity),
        periodicity=float(periodicity), 
        convergence=float(convergence)
    )


def map_features_to_lora_cfg(ca_features: CAFeatures, config: Dict[str, Any]) -> LoRAConfig:
    """Map CA features to LoRA configuration using config parameters."""
    evo_config = config.get('evo', {})
    
    # Get LoRA candidates from config
    rank_candidates = evo_config.get('rank_candidates')
    alpha_candidates = evo_config.get('alpha_candidates')
    dropout_candidates = evo_config.get('dropout_candidates')
    target_modules = evo_config.get('target_modules')
    
    if not all([rank_candidates, alpha_candidates, dropout_candidates, target_modules]):
        raise ValueError("FAIL-FAST: LoRA parameters not specified in config")
    
    # Map features to LoRA parameters using discrete candidates
    # Use CA features to select parameters from discrete lists
    rank = rank_candidates[int(ca_features.complexity * len(rank_candidates)) % len(rank_candidates)]
    alpha = alpha_candidates[int(ca_features.intensity * len(alpha_candidates)) % len(alpha_candidates)]
    dropout = dropout_candidates[int(ca_features.periodicity * len(dropout_candidates)) % len(dropout_candidates)]
    
    return LoRAConfig(
        r=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=tuple(target_modules)
    )


class QuixBugsRealDataset(DatasetProvider):
    """Real QuixBugs dataset provider using actual dataset."""
    
    def __init__(self, config: Dict[str, Any]):
        # Store config first
        self.config = config
        
        # Access dataset config from experiment section  
        experiment_config = config.get('experiment', {})
        dataset_config = experiment_config.get('dataset', {})
        
        # Use config-driven dataset path - no auto-detection allowed
        self.adapter = QuixBugsRealAdapter(config=self.config)
        
        # 🔥 FIX: Use centralized training problems to prevent data leakage  
        # This MUST match exactly what's used in DoRA/LoRA training
        from coral.domain.dataset_constants import QUIXBUGS_TRAINING_PROBLEMS
        CENTRALIZED_TRAINING_PROBLEMS = QUIXBUGS_TRAINING_PROBLEMS
        
        # Use config-specified problems if available, otherwise use centralized constants
        config_training_problems = set(dataset_config.get('training_problems', []))
        if config_training_problems:
            self.training_problems = config_training_problems
            print(f"📁 QuixBugs dataset loaded with CONFIG-specified training split")
        else:
            self.training_problems = CENTRALIZED_TRAINING_PROBLEMS
            print(f"📁 QuixBugs dataset loaded with CENTRALIZED training split (preventing data leakage)")
        
        print(f"🎯 Training problems: {len(self.training_problems)} problems EXCLUDED from evaluation")
        print(f"⚠️  ANTI-CONTAMINATION: Only clean problems will be used for evaluation")
    
    def problems(self) -> Iterable[Dict[str, Any]]:
        """Yield real QuixBugs problems (excluding training problems)."""
        total_problems = 0
        excluded_problems = 0
        clean_problems = []
        
        for problem in self.adapter.problems():
            total_problems += 1
            problem_name = problem.get('name', 'unknown')
            
            # Only yield problems NOT in training set
            if problem_name not in self.training_problems:
                clean_problems.append(problem_name)
                yield problem
            else:
                excluded_problems += 1
                # Debug: Log excluded contaminated problems
                if excluded_problems <= 5:  # Only log first 5 to avoid spam
                    print(f"   ⚠️  EXCLUDED contaminated problem: {problem_name}")
        
        # Final contamination report
        clean_count = len(clean_problems)
        print(f"\n🛡️  ANTI-CONTAMINATION REPORT:")
        print(f"   • Total problems found: {total_problems}")
        print(f"   • Contaminated problems excluded: {excluded_problems}")
        print(f"   • Clean problems available: {clean_count}")
        print(f"   • Contamination prevented: {excluded_problems/total_problems*100:.1f}%")
        
        if clean_count > 0:
            print(f"\n✅ Clean evaluation problems ({clean_count}):")
            for problem_name in sorted(clean_problems):
                print(f"   • {problem_name}")
        else:
            print(f"\n❌ WARNING: No clean problems available for evaluation!")
            print(f"   All problems are contaminated with training data.")


class CodeLlamaRealRunner(ModelRunner):
    """Real CodeLlama model runner with LoRA - config-driven."""
    
    def __init__(self, lora_cfg: LoRAConfig, config: Dict[str, Any], genome: Genome = None):
        self.lora_cfg = lora_cfg
        self.config = config
        self.genome = genome        # ← Set BEFORE _setup_model() uses it
        # 🔥 FIX: Cache loaded model and tokenizer for reuse across problems
        # Safe because: Each genome gets own ModelRunner instance in Modal container
        # Container dies after evaluation, cleaning up all memory automatically
        self._cached_model = None
        self._cached_tokenizer = None
        self._cached_adapter_path = None
        self._setup_model()
    
    def _setup_model(self):
        """Setup model with real LoRA configuration from config."""
        # Get model configuration
        model_config = self.config.get('experiment', {}).get('model', {})
        cache_config_dict = self.config.get('cache', {})
        
        model_name = model_config.get('name')
        if not model_name:
            raise ValueError("FAIL-FAST: Model name not specified in config")
        
        # Create cache config from config dict (run_id comes from genome, not config)
        cache_config = CacheConfig(
            artifacts_dir=cache_config_dict.get('artifacts_dir', './artifacts'),
            base_checkpoint=cache_config_dict.get('base_checkpoint', model_name),
            cache_metadata=cache_config_dict.get('metadata', True),
            cleanup_threshold=cache_config_dict.get('cleanup_threshold', 100),
            run_id=None  # Not used - run_id comes from genome only
        )
        
        # 🔥 FIX: No fallbacks - config values pass seamlessly
        if not self.genome:
            raise ValueError("FAIL-FAST: Genome is required for model creation to preserve run_id and adapter_type")
        
        if not hasattr(self.genome, 'run_id'):
            raise ValueError("FAIL-FAST: Genome missing run_id field - cannot proceed without experiment identifier")
        
        # Use genome's run_id directly - no fallbacks
        heavy_genes = HeavyGenes.from_lora_config(self.lora_cfg, run_id=self.genome.run_id)
        
        print(f"   🔍 HeavyGenes from genome:")
        print(f"      • Run ID: {self.genome.run_id}")
        print(f"      • Adapter type: {heavy_genes.adapter_type}")
        print(f"      • Rank: {heavy_genes.rank}, Alpha: {heavy_genes.alpha}, Dropout: {heavy_genes.dropout}")
        
        # Define trainer function
        def trainer_fn(genes: HeavyGenes, base_ckpt: str) -> str:
            """Real trainer function using config parameters."""
            from coral.domain.lora_training import train_codellama_lora
            
            # Generate save path (LoRA adapters are directories, not .pt files)
            adapter_dirname = f"adapter_{genes.to_hash()}"
            save_path = str(Path(cache_config.artifacts_dir) / adapter_dirname)
            
            # Get training config from main config
            training_config = self.config.get('training', {})
            
            # Train with config parameters - PASS CONFIG!
            return train_codellama_lora(base_ckpt, genes, save_path, config=self.config)
        
        # Use cache system
        cache_start = time.time()
        self._adapter_path = get_or_train_adapter(
            heavy_genes=heavy_genes,
            trainer_fn=trainer_fn,
            cache_config=cache_config
        )
        cache_time = time.time() - cache_start
        
        print(f"✅ Clone-cache operation completed in {cache_time:.2f}s")
        print(f"🤖 CodeLlama ready with adapter: {self._adapter_path}")
        self._model_loaded = True
    
    def generate(self, prompt: str, max_tokens: int = None, cheap_knobs=None) -> str:
        """Generate code completion using config parameters with optional cheap knobs."""
        if not self._model_loaded:
            raise RuntimeError("FAIL-FAST: Model not loaded")
        
        # Get generation parameters from config
        gen_config = self.config.get('generation', {})
        if max_tokens is None:
            max_tokens = gen_config.get('max_tokens', 512)
        
        # Check if we should use Modal (includes queue_modal)
        infra_config = self.config.get('infra', {})
        executor_type = infra_config.get('executor')
        use_modal = executor_type in ['modal', 'queue_modal']
        
        if use_modal:
            return self._generate_modal(prompt, max_tokens, cheap_knobs)
        else:
            return self._generate_local(prompt, max_tokens, cheap_knobs)
    
    def _generate_modal(self, prompt: str, max_tokens: int, cheap_knobs=None) -> str:
        """Generate using Modal with config parameters and optional cheap knobs."""
        try:
            # 🔍 DEBUGGING: Trace cheap knobs parameter flow
            print(f"🔍 MODAL GENERATION DEBUG:")
            print(f"   • cheap_knobs parameter: {cheap_knobs}")
            print(f"   • cheap_knobs type: {type(cheap_knobs)}")
            if cheap_knobs:
                print(f"   • cheap_knobs attributes: {[attr for attr in dir(cheap_knobs) if not attr.startswith('_')]}")
                print(f"   • temperature: {getattr(cheap_knobs, 'temperature', 'MISSING')}")
                print(f"   • top_p: {getattr(cheap_knobs, 'top_p', 'MISSING')}")
                print(f"   • top_k: {getattr(cheap_knobs, 'top_k', 'MISSING')}")
            else:
                print(f"   ❌ CRITICAL: cheap_knobs is None/False - this violates two-loop architecture!")
                
                # FAIL-FAST: Should not reach Modal with None cheap knobs during evolution
                raise RuntimeError(
                    f"FAIL-FAST: cheap_knobs is None in Modal generation during evolution. "
                    f"Two-loop architecture requires CA-derived parameters. "
                    f"Check parameter passing from fitness function → model.generate() → _generate_modal()"
                )
            
            # Import Modal functions
            import modal
            modal_config = self.config.get('infra', {}).get('modal', {})
            app_name = modal_config.get('app_name')
            
            if not app_name:
                raise ValueError("FAIL-FAST: Modal app_name not specified in config")
            
            # Get Modal function
            generate_fn = modal.Function.from_name(app_name, "generate_code_modal")
            
            # Get model config
            model_config = self.config.get('experiment', {}).get('model', {})
            model_name = model_config.get('name')
            
            # IMPROVED: Extract problem info from prompt directly instead of in Modal
            import re
            
            # Extract function name and buggy code from prompt
            func_match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)', prompt)
            if not func_match:
                raise RuntimeError(f"FAIL-FAST: Could not extract function name from prompt")
            problem_name = func_match.group(1)
            
            # Extract buggy code from prompt (look for code block)
            code_match = re.search(r'```python\s*\n(.*?)```', prompt, re.DOTALL)
            if not code_match:
                code_match = re.search(r'```\s*\n(.*?)```', prompt, re.DOTALL)
            
            if code_match:
                buggy_code = code_match.group(1).strip()
            else:
                # Fallback: use the function definition from the prompt
                buggy_code = func_match.group(0)
            
            print(f"   📝 Extracted: problem='{problem_name}', buggy_code={len(buggy_code)} chars")
            
            # TWO-LOOP ARCHITECTURE: Convert cheap knobs to dict for Modal serialization
            from coral.domain.cheap_knobs import cheap_knobs_to_generation_kwargs
            cheap_knobs_dict = cheap_knobs_to_generation_kwargs(cheap_knobs)
            
            print(f"   🎛️ CONVERTED CHEAP KNOBS:")
            print(f"      • Temperature: {cheap_knobs_dict['temperature']:.3f} (from CA complexity)")
            print(f"      • Top-p: {cheap_knobs_dict['top_p']:.3f} (from CA intensity)")
            print(f"      • Top-k: {cheap_knobs_dict['top_k']} (from CA convergence)")
            print(f"      • Repetition penalty: {cheap_knobs_dict['repetition_penalty']:.3f} (from CA periodicity)")
            print(f"      • Max tokens: {cheap_knobs_dict['max_new_tokens']} (from combined CA features)")
            print(f"      • Sampling: {cheap_knobs_dict['do_sample']} (from CA creativity)")
            
            # Call Modal function with cheap knobs
            result = generate_fn.remote(model_name, self._adapter_path, problem_name, buggy_code, self.config, cheap_knobs_dict)
            return result
            
        except Exception as e:
            raise RuntimeError(f"FAIL-FAST: Modal generation failed: {e}")
    
    def _generate_local(self, prompt: str, max_tokens: int, cheap_knobs=None) -> str:
        """Local CodeLlama generation with cached model for reuse."""
        try:
            import torch
            
            # Get model config
            model_config = self.config.get('experiment', {}).get('model', {})
            base_model = model_config.get('name', 'codellama/CodeLlama-7b-Python-hf')
            
            print(f"   🤖 Local generation: {base_model} + {self._adapter_path}")
            
            # 🔥 FIX: Use cached model if available, otherwise load and cache
            if (self._cached_model is None or 
                self._cached_tokenizer is None or 
                self._cached_adapter_path != self._adapter_path):
                
                print(f"   📥 Loading base model: {base_model}")
                print(f"   🗂️  Using model cache: /cache/models")
                print(f"   ✅ Using cached model (offline mode)")
                
                # Import and load model with adapter
                from coral.domain.lora_training import load_lora_adapter_with_base_model
                
                self._cached_model, self._cached_tokenizer = load_lora_adapter_with_base_model(
                    base_model_name=base_model,
                    adapter_path=self._adapter_path
                )
                self._cached_adapter_path = self._adapter_path
                print(f"   🔗 Loading LoRA adapter: {self._adapter_path}")
                print(f"   🔧 Set padding token to EOS token: </s>")
            else:
                print(f"   ♻️  Reusing cached model and adapter")
            
            model = self._cached_model
            tokenizer = self._cached_tokenizer
            
            # 🔥 FIX: Use cheap knobs if provided (two-loop architecture), otherwise config defaults
            if cheap_knobs is not None:
                temperature = cheap_knobs.temperature
                top_p = cheap_knobs.top_p
                top_k = cheap_knobs.top_k
                max_tokens = cheap_knobs.max_new_tokens
                do_sample = cheap_knobs.do_sample
                print(f"   🎛️ Using CA-derived cheap knobs: T={temperature:.3f}, p={top_p:.3f}, k={top_k}")
            else:
                # Fallback to config defaults
                gen_config = self.config.get('generation', {})
                temperature = gen_config.get('temperature', 0.7)
                top_p = gen_config.get('top_p', 0.9)
                top_k = gen_config.get('top_k', 50)
                do_sample = True
                print(f"   🔧 Using config defaults: T={temperature:.3f}, p={top_p:.3f}, k={top_k}")
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt")
            
            # FIX: Move input tensors to same device as model (GPU)
            device = next(model.parameters()).device
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)
            
            # Generate with CA-derived or config parameters
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the new part (after the input prompt)
            if generated_text.startswith(prompt):
                response = generated_text[len(prompt):].strip()
            else:
                response = generated_text.strip()
            
            print(f"   ✅ Generated {len(response)} characters")
            return response
            
        except ImportError as e:
            raise NotImplementedError(
                f"FAIL-FAST: Local CodeLlama generation requires transformers/torch. "
                f"Missing dependency: {e}. Use Modal executor instead."
            )
        except Exception as e:
            raise RuntimeError(f"FAIL-FAST: Local generation failed: {e}")


class QuixBugsRealFitness(FitnessFn):
    """Real multi-objective fitness function using config parameters."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.evaluation_config = self.config.get('evaluation', {})
        
        # Get fitness weights from config
        self.fitness_weights = self.evaluation_config.get('fitness_weights', {})
        if not self.fitness_weights:
            raise ValueError("FAIL-FAST: fitness_weights not specified in config")
        
        # Initialize simple emergent behavior tracker
        emergent_config = self.config.get('emergent_tracking', {})
        if emergent_config.get('enabled', False):
            try:
                # Get emergent behavior path from centralized config
                from coral.config.path_utils import get_emergent_behavior_path
                output_dir = Path(get_emergent_behavior_path(self.config))
                self.emergent_tracker = SimpleEmergentTracker(output_dir)
                print(f"🌟 Emergent behavior tracking ENABLED: {output_dir}")
                print(f"   • Alert threshold: {emergent_config.get('alert_threshold', 0.8)}")
                print(f"   • Save frequency: {emergent_config.get('save_frequency', 20)} evaluations")
                
            except Exception as tracker_error:
                print(f"❌ EMERGENT TRACKING FAILED TO INITIALIZE: {tracker_error}")
                print(f"   • Config section: {emergent_config}")
                print(f"   • FAIL-FAST: Fix emergent tracking configuration")
                self.emergent_tracker = None
                raise RuntimeError(f"FAIL-FAST: Emergent tracking initialization failed: {tracker_error}")
        else:
            self.emergent_tracker = None
            print(f"📊 Emergent behavior tracking disabled in config")
    
    def __call__(self, 
                 genome: Genome,
                 model: ModelRunner, 
                 problems: Iterable[Dict[str, Any]],
                 ca_features = None) -> float:  # 🔥 NEW: Accept pre-computed features
        """Single-objective evaluation for compatibility."""
        multi_scores = self.evaluate_multi_objective(genome, model, problems, ca_features)  # 🔥 FIX: Pass ca_features
        return multi_scores.overall_fitness(weights=self.fitness_weights)
    
    def evaluate_multi_objective(self, 
                                genome: Genome,
                                model: ModelRunner, 
                                problems: Iterable[Dict[str, Any]],
                                ca_features = None) -> MultiObjectiveScores:  # 🔥 NEW: Accept pre-computed features
        """Real multi-objective evaluation using config parameters."""
        
        print(f"\n🧬 MULTI-OBJECTIVE EVALUATION")
        print(f"{'='*50}")
        print(f"🔬 Genome ID: {genome.id if hasattr(genome, 'id') else 'unknown'}")
        
        total_start_time = time.time()
        
        # 🔥 FIX: Use pre-computed CA features from genome if available
        if ca_features is not None:
            print(f"✅ Using provided CA features (consistency ensured)")
            print(f"   🔬 CA FEATURES PROVIDED:")
            print(f"      • Complexity: {ca_features.complexity:.4f} (drives temperature)")
            print(f"      • Intensity: {ca_features.intensity:.4f} (drives top_p)")
            print(f"      • Periodicity: {ca_features.periodicity:.4f} (drives repetition penalty)")
            print(f"      • Convergence: {ca_features.convergence:.4f} (drives top_k)")
        elif hasattr(genome, 'ca_features') and genome.ca_features is not None:
            print(f"✅ Using genome's stored CA features (consistency ensured)")
            ca_features = genome.ca_features
            print(f"   🔬 CA FEATURES FROM GENOME:")
            print(f"      • Complexity: {ca_features.complexity:.4f} (drives temperature)")
            print(f"      • Intensity: {ca_features.intensity:.4f} (drives top_p)")
            print(f"      • Periodicity: {ca_features.periodicity:.4f} (drives repetition penalty)")
            print(f"      • Convergence: {ca_features.convergence:.4f} (drives top_k)")
        else:
            # FAIL-FAST: No fallback computation allowed - architectural integrity required
            raise RuntimeError(
                f"FAIL-FAST: No CA features available for evaluation consistency. "
                f"Genome {genome.id} missing stored CA features and none provided. "
                f"This breaks the two-loop architecture integrity (CA → LoRA vs CA → cheap knobs). "
                f"Fix genome creation to store CA features."
            )
        
        # TWO-LOOP ARCHITECTURE: Generate cheap knobs from CA features
        print(f"🎛️ Generating Cheap Knobs from CA Features...")
        knobs_start_time = time.time()
        
        from coral.domain.cheap_knobs import map_ca_features_to_cheap_knobs
        
        # FAIL-FAST: Cheap knobs config must be explicitly provided
        if 'cheap_knobs' not in self.config:
            raise ValueError("FAIL-FAST: 'cheap_knobs' section missing from config - two-loop architecture requires explicit parameter ranges")
        
        knobs_config = self.config['cheap_knobs']
        cheap_knobs = map_ca_features_to_cheap_knobs(ca_features, knobs_config)
        
        knobs_time = time.time() - knobs_start_time
        print(f"   ✅ Cheap knobs generated in {knobs_time:.4f}s:")
        print(f"      • Temperature: {cheap_knobs.temperature} (complexity-driven)")
        print(f"      • Top-p: {cheap_knobs.top_p} (intensity-driven)")
        print(f"      • Top-k: {cheap_knobs.top_k} (convergence-driven)")
        print(f"      • Repetition penalty: {cheap_knobs.repetition_penalty} (periodicity-driven)")
        print(f"      • Max tokens: {cheap_knobs.max_new_tokens} (combined features)")
        print(f"      • Sampling: {cheap_knobs.do_sample} (creativity-driven)")
        
        # Map features to LoRA config using config parameters
        print(f"🔗 Using genome's existing LoRA Configuration...")
        mapping_start_time = time.time()
        
        # 🔥 FIX: Use genome's existing LoRA config (preserves adapter_type and run_id)
        # instead of deriving fresh config that loses these values
        derived_lora = genome.lora_cfg
        
        # Debug: Show what we're actually using
        print(f"   🔍 Heavy genes (what model learns):")
        print(f"      • Rank: {derived_lora.r}")
        print(f"      • Alpha: {derived_lora.alpha}")
        print(f"      • Dropout: {derived_lora.dropout}")
        print(f"      • Adapter type: {getattr(derived_lora, 'adapter_type', 'lora')}")
        print(f"      • Target modules: {derived_lora.target_modules}")
        
        mapping_time = time.time() - mapping_start_time
        print(f"   ✅ Using existing config completed in {mapping_time:.4f}s")
        
        # Evaluate on QuixBugs problems
        print(f"\n🎯 EVALUATING ON QUIXBUGS PROBLEMS")
        print(f"{'─'*50}")
        
        bugfix_scores = []
        style_scores = []
        security_scores = []
        runtime_scores = []
        syntax_scores = []  # NEW: Track syntax scores
        
        problems_list = list(problems)
        
        # Get evaluation parameters from config
        adaptive_config = self.evaluation_config.get('adaptive_testing', {})
        if adaptive_config.get('enable', False):
            selected_problems = self._select_adaptive_test_cases(problems_list, genome)
        else:
            selected_problems = problems_list
        
        print(f"📁 Selected: {len(selected_problems)}/{len(problems_list)} QuixBugs problems")
        
        for i, problem in enumerate(selected_problems, 1):
            problem_name = problem.get('name', f'problem_{i}')
            print(f"\n🔸 Problem {i}/{len(selected_problems)}: {problem_name}")
            
            try:
                # Generate solution using config parameters
                print(f"   🤖 Generating solution...")
                generation_start = time.time()
                gen_config = self.config.get('generation', {})
                max_tokens = gen_config.get('max_tokens', 512)
                
                try:
                    # 🔍 DEBUGGING: Verify cheap knobs before passing to model
                    print(f"🔍 FITNESS FUNCTION DEBUG:")
                    print(f"   • About to call model.generate() with cheap_knobs: {cheap_knobs}")
                    print(f"   • cheap_knobs type: {type(cheap_knobs)}")
                    if cheap_knobs:
                        print(f"   • cheap_knobs.temperature: {cheap_knobs.temperature}")
                        print(f"   • cheap_knobs.top_p: {cheap_knobs.top_p}")
                        print(f"   • cheap_knobs.top_k: {cheap_knobs.top_k}")
                    else:
                        print(f"   ❌ CRITICAL: cheap_knobs is None in fitness function!")
                        raise RuntimeError("FAIL-FAST: cheap_knobs is None in fitness function - CA feature mapping failed")
                    
                    # TWO-LOOP ARCHITECTURE: Pass cheap knobs to control HOW the model generates
                    generated_code = model.generate(problem["prompt"], max_tokens, cheap_knobs=cheap_knobs)
                    generation_time = time.time() - generation_start
                    print(f"   ✅ Generated in {generation_time:.2f}s ({len(generated_code)} chars)")
                    print(f"   📋 Code preview: {generated_code[:100]}...")
                except Exception as gen_error:
                    print(f"   ❌ Generation failed: {gen_error}")
                    # Log generation failure details
                    bugfix_scores.append(0.0)
                    style_scores.append(0.0) 
                    security_scores.append(0.0)
                    runtime_scores.append(0.0)
                    syntax_scores.append(0.0)  # NEW: Add zero syntax score
                    continue
                
                # Load test cases
                print(f"   📚 Loading test cases...")
                try:
                    test_cases = self._load_test_cases_for_problem(problem_name)
                    if test_cases:
                        print(f"   ✅ Test cases loaded ({len(test_cases)} chars)")
                    else:
                        print(f"   ⚠️  No test cases found for {problem_name} - will use evolutionary pressure")
                except Exception as test_load_error:
                    print(f"   ⚠️  Test case loading issue: {test_load_error} - using evolutionary pressure")
                    test_cases = None
                
                # Evaluate using domain logic
                print(f"   📊 Evaluating solution...")
                eval_start = time.time()
                try:
                    # Handle missing test cases gracefully
                    if test_cases is None:
                        print(f"   ⚠️  No test cases available - using syntax/style-only evaluation")
                        
                    evaluation_result = evaluate_quixbugs_code(generated_code, problem, test_cases)
                    eval_time = time.time() - eval_start
                    
                    # ADD SIMPLE EMERGENT BEHAVIOR TRACKING
                    if self.emergent_tracker:
                        try:
                            # Get current generation from genome ID (more reliable than config)
                            current_generation = self._extract_generation_from_genome_id(genome.id if hasattr(genome, 'id') else 'gen0_unknown')
                            
                            # Track this evaluation for emergent behaviors
                            self.emergent_tracker.track_evaluation(
                                problem_name=problem_name,
                                genome_id=genome.id if hasattr(genome, 'id') else f"genome_{hash(str(genome)) % 10000}",
                                generation=current_generation,
                                ca_features=ca_features.__dict__,  # Convert to dict
                                lora_config=derived_lora.__dict__,  # Convert to dict
                                evaluation_result={
                                    'bugfix': evaluation_result.bugfix,
                                    'style': evaluation_result.style,
                                    'security': evaluation_result.security,
                                    'runtime': evaluation_result.runtime,
                                    'test_cases_passed': evaluation_result.test_cases_passed,
                                    'test_cases_run': evaluation_result.test_cases_run,
                                    'test_execution_time': evaluation_result.test_execution_time
                                },
                                generated_code=generated_code
                            )
                        except Exception as tracking_error:
                            print(f"   ⚠️  Emergent tracking failed: {tracking_error}")
                    
                    print(f"   ✅ Evaluation completed in {eval_time:.2f}s")
                    
                    # ENHANCED LOGGING: Show detailed test execution results
                    print(f"   🔍 Detailed Results:")
                    print(f"      • Syntax valid: {evaluation_result.syntax_valid}")
                    print(f"      • Function defined: {evaluation_result.function_defined}")
                    print(f"      • Tests executed: {evaluation_result.test_cases_run}")
                    print(f"      • Tests passed: {evaluation_result.test_cases_passed}")
                    
                    if evaluation_result.test_cases_run > 0:
                        pass_rate = evaluation_result.test_cases_passed / evaluation_result.test_cases_run
                        print(f"      • Pass rate: {pass_rate:.1%} ({evaluation_result.test_cases_passed}/{evaluation_result.test_cases_run})")
                        
                        if evaluation_result.test_cases_passed == evaluation_result.test_cases_run:
                            print(f"      ✅ ALL TESTS PASSED!")
                        elif evaluation_result.test_cases_passed > 0:
                            print(f"      ⚠️  PARTIAL SUCCESS: {evaluation_result.test_cases_passed} passed, {evaluation_result.test_cases_run - evaluation_result.test_cases_passed} failed")
                        else:
                            print(f"      ❌ ALL TESTS FAILED")
                    else:
                        print(f"      ⚠️  NO TESTS EXECUTED")
                    
                    print(f"      • Test execution time: {evaluation_result.test_execution_time:.3f}s") 
                    print(f"      • Style violations: {evaluation_result.style_violations}")
                    
                    if evaluation_result.security_issues:
                        print(f"      • Security issues: {len(evaluation_result.security_issues)}")
                        for issue in evaluation_result.security_issues[:3]:  # Show first 3
                            print(f"        - {issue}")
                    
                    # NEW: Evaluate syntax as 5th objective
                    from coral.domain.quixbugs_evaluation import evaluate_syntax_multi_objective
                    syntax_score = evaluate_syntax_multi_objective(generated_code, problem_name)
                    print(f"      • Syntax score: {syntax_score:.3f}")
                    
                    print(f"   📈 Scores: B:{evaluation_result.bugfix:.3f} S:{evaluation_result.style:.3f} Sec:{evaluation_result.security:.3f} R:{evaluation_result.runtime:.3f} Syn:{syntax_score:.3f}")
                    
                    bugfix_scores.append(evaluation_result.bugfix)
                    style_scores.append(evaluation_result.style)
                    security_scores.append(evaluation_result.security)
                    runtime_scores.append(evaluation_result.runtime)
                    syntax_scores.append(syntax_score)  # NEW: Add syntax score
                    
                except Exception as eval_error:
                    print(f"   ❌ Evaluation failed: {eval_error}")
                    print(f"   🔍 Generated code that failed evaluation:")
                    print(f"   {generated_code[:200]}...")
                    
                    # Add zero scores for failed evaluation
                    bugfix_scores.append(0.0)
                    style_scores.append(0.0)
                    security_scores.append(0.0) 
                    runtime_scores.append(0.0)
                    syntax_scores.append(0.0)  # NEW: Add zero syntax score
                
            except Exception as e:
                print(f"   ❌ Evaluation failed: FAIL-FAST: {str(e)}")
                
                # Add zero scores for completely failed problems
                bugfix_scores.append(0.0)
                style_scores.append(0.0)
                security_scores.append(0.0)
                runtime_scores.append(0.0)
                syntax_scores.append(0.0)  # NEW: Add zero syntax score
        
        # Calculate averages
        avg_bugfix = sum(bugfix_scores) / max(len(bugfix_scores), 1)
        avg_style = sum(style_scores) / max(len(style_scores), 1)
        avg_security = sum(security_scores) / max(len(security_scores), 1)
        avg_runtime = sum(runtime_scores) / max(len(runtime_scores), 1)
        avg_syntax = sum(syntax_scores) / max(len(syntax_scores), 1)  # NEW: Calculate syntax average
        
        print(f"\n📊 FINAL MULTI-OBJECTIVE SCORES")
        print(f"{'─'*50}")
        print(f"🎯 Average Scores Across {len(bugfix_scores)} Problems:")
        print(f"   • Bugfix:   {avg_bugfix:.3f}")
        print(f"   • Style:    {avg_style:.3f}")
        print(f"   • Security: {avg_security:.3f}")
        print(f"   • Runtime:  {avg_runtime:.3f}")
        print(f"   • Syntax:   {avg_syntax:.3f}")  # NEW: Display syntax score
        
        total_time = time.time() - total_start_time
        print(f"⏱️  Total Evaluation Time: {total_time:.2f}s")
        
        # Check thresholds from config
        threshold_config = self.config.get('threshold', {})
        base_thresholds = threshold_config.get('base_thresholds', {})
        
        if base_thresholds:
            print(f"\n📈 PROGRESS TOWARDS CONFIG TARGETS:")
            for metric, score in [('bugfix', avg_bugfix), ('style', avg_style), ('security', avg_security), ('runtime', avg_runtime), ('syntax', avg_syntax)]:
                target = base_thresholds.get(metric, 1.0)
                progress = (score / target * 100) if target > 0 else 100
                print(f"   • {metric.capitalize()}: {score:.3f} / {target:.2f} target ({progress:.1f}%)")
        
        # Print emergent behavior summary if tracking enabled
        if self.emergent_tracker:
            self.emergent_tracker.print_progress_summary()
        
        return MultiObjectiveScores(
            bugfix=avg_bugfix,
            style=avg_style,
            security=avg_security,
            runtime=avg_runtime,
            syntax=avg_syntax  # NEW: Include syntax score
        )
    
    def _run_ca_from_genome(self, genome: Genome) -> list:
        """Run CA evolution from genome using config parameters."""
        try:
            # Get CA configuration parameters
            ca_config = self.config.get('evo', {}).get('ca', {})
            steps = ca_config.get('steps_range', [5, 20])[1]  # Use max steps
            
            # Debug: Check genome structure
            print(f"   🔍 Genome seed type: {type(genome.seed)}")
            print(f"   🔍 Genome seed grid type: {type(genome.seed.grid)}")
            print(f"   🔍 Genome seed grid shape: {getattr(genome.seed.grid, 'shape', 'no shape')}")
            print(f"   🔍 Genome seed rule: {genome.seed.rule}")
            print(f"   🔍 CA steps: {steps}")
            
            # Convert grid to proper format
            if hasattr(genome.seed.grid, 'tolist'):
                grid = genome.seed.grid  # Already numpy array
            elif isinstance(genome.seed.grid, list):
                grid = np.array(genome.seed.grid)
            else:
                # Try to convert whatever it is to numpy array
                grid = np.array(genome.seed.grid)
            
            print(f"   🔍 Converted grid shape: {grid.shape}")
            print(f"   🔍 Grid sample: {grid.flatten()[:10]}...")
            
            # Create CA seed dictionary
            seed_dict = {
                'grid': grid,
                'rule': genome.seed.rule,
                'steps': steps
            }
            
            return self._run_ca_local(seed_dict)
            
        except Exception as e:
            print(f"   ❌ CA evolution failed: {e}")
            print(f"   🔍 Genome object: {genome}")
            print(f"   🔍 Genome seed: {genome.seed if hasattr(genome, 'seed') else 'no seed'}")
            
            # FAIL-FAST: No empty history fallbacks allowed
            raise RuntimeError(
                f"FAIL-FAST: CA evolution failed for genome {genome.id if hasattr(genome, 'id') else 'unknown'}: {e}. "
                f"Cannot proceed with empty CA history - fix the CA evolution pipeline."
            )
    
    def _run_ca_local(self, seed_dict: Dict[str, Any]) -> list:
        """Local CA evolution implementation."""
        try:
            grid = seed_dict['grid']
            steps = seed_dict.get('steps', 15)
            rule = seed_dict.get('rule', 30)
            
            # Ensure grid is numpy array
            if not isinstance(grid, np.ndarray):
                grid = np.array(grid)
            
            print(f"   🔍 CA Local: grid shape {grid.shape}, steps {steps}, rule {rule}")
            
            history = []
            current_state = grid.copy()
            
            # Run CA evolution for specified steps
            for step in range(steps):
                history.append(current_state.copy())
                
                # Apply CA rule
                new_state = np.zeros_like(current_state)
                height, width = current_state.shape
                
                for i in range(height):
                    for j in range(width):
                        # Count neighbors (Moore neighborhood)
                        neighbor_count = 0
                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                if di == 0 and dj == 0:
                                    continue
                                ni, nj = (i + di) % height, (j + dj) % width
                                neighbor_count += current_state[ni, nj]
                        
                        # Apply Game of Life rules
                        if current_state[i, j] == 1:
                            # Live cell
                            new_state[i, j] = 1 if neighbor_count in [2, 3] else 0
                        else:
                            # Dead cell
                            new_state[i, j] = 1 if neighbor_count == 3 else 0
                
                current_state = new_state
            
            print(f"   ✅ CA Local: generated {len(history)} states")
            return history
            
        except Exception as e:
            print(f"   ❌ CA Local evolution failed: {e}")
            print(f"   🔍 Seed dict: {seed_dict}")
            
            # FAIL-FAST: No minimal history fallbacks allowed
            raise RuntimeError(
                f"FAIL-FAST: Local CA evolution failed: {e}. "
                f"Cannot proceed with minimal/dummy CA history - fix the CA implementation."
            )
    
    def _select_adaptive_test_cases(self, problems_list: List[Dict[str, Any]], genome: Genome) -> List[Dict[str, Any]]:
        """Select test cases based on difficulty classification and genome capability."""
        adaptive_config = self.evaluation_config.get('adaptive_testing', {})
        
        if not adaptive_config.get('enable', False):
            return problems_list
        
        # Get limits and thresholds from config
        max_easy = adaptive_config.get('max_easy_problems', 10)
        max_medium = adaptive_config.get('max_medium_problems', 12)
        max_hard = adaptive_config.get('max_hard_problems', 8)
        
        capability_thresholds = adaptive_config.get('capability_thresholds', {})
        easy_cutoff = capability_thresholds.get('easy_cutoff', 0.3)
        medium_cutoff = capability_thresholds.get('medium_cutoff', 0.6)
        
        # Classify problems by difficulty based on known characteristics
        easy_problems = []
        medium_problems = []
        hard_problems = []
        
        for problem in problems_list:
            problem_name = problem.get('name', '')
            difficulty = self._classify_problem_difficulty(problem_name)
            
            if difficulty == 'easy':
                easy_problems.append(problem)
            elif difficulty == 'medium':
                medium_problems.append(problem)
            else:
                hard_problems.append(problem)
        
        # Estimate genome capability from its characteristics
        genome_capability = self._estimate_genome_capability(genome)
        
        print(f"   🎯 Adaptive Selection:")
        print(f"      • Genome capability: {genome_capability:.3f}")
        print(f"      • Available: {len(easy_problems)} easy, {len(medium_problems)} medium, {len(hard_problems)} hard")
        
        # Select problems based on capability
        selected_problems = []
        
        # Always include some easy problems for baseline
        selected_problems.extend(easy_problems[:max_easy])
        
        # Add medium problems if genome shows reasonable capability
        if genome_capability >= easy_cutoff:
            selected_problems.extend(medium_problems[:max_medium])
            print(f"      • Including medium problems (capability >= {easy_cutoff})")
        
        # Add hard problems only for high-capability genomes
        if genome_capability >= medium_cutoff:
            selected_problems.extend(hard_problems[:max_hard])
            print(f"      • Including hard problems (capability >= {medium_cutoff})")
        
        print(f"      • Selected: {len(selected_problems)} total problems")
        
        return selected_problems

    def _classify_problem_difficulty(self, problem_name: str) -> str:
        """Classify QuixBugs problems by difficulty using centralized constants."""
        # Import centralized problem classifications
        from coral.domain.dataset_constants import (
            EASY_PROBLEMS, MEDIUM_PROBLEMS, HARD_PROBLEMS
        )
        
        if problem_name in EASY_PROBLEMS:
            return 'easy'
        elif problem_name in HARD_PROBLEMS:
            return 'hard'
        elif problem_name in MEDIUM_PROBLEMS:
            return 'medium'
        else:
            # For any new problems not in our classification
            return 'medium'  # Default to medium

    def _estimate_genome_capability(self, genome: Genome) -> float:
        """Estimate genome capability from its characteristics."""
        try:
            # Base capability from LoRA configuration
            capability = 0.0
            
            # Higher rank suggests more adaptation capability
            if hasattr(genome, 'lora_cfg') and genome.lora_cfg:
                rank = genome.lora_cfg.r
                # Normalize rank to 0-1 scale (typical range 4-64)
                capability += min(rank / 64.0, 1.0) * 0.4
            
            # CA seed characteristics
            if hasattr(genome, 'seed') and genome.seed:
                # More complex CA rules suggest higher capability
                if hasattr(genome.seed, 'rule'):
                    rule = genome.seed.rule
                    # Rules closer to middle range (30-220) often more complex
                    rule_complexity = 1.0 - abs(rule - 127) / 127.0
                    capability += rule_complexity * 0.3
                
                # Initial density affects CA dynamics
                if hasattr(genome.seed, 'grid'):
                    try:
                        grid = genome.seed.grid
                        if hasattr(grid, 'sum'):
                            density = grid.sum() / grid.size
                        else:
                            raise ValueError(f"FAIL-FAST: Genome grid has no sum() method - invalid grid type: {type(grid)}")
                        # Moderate density (0.2-0.5) often more interesting
                        density_score = 1.0 - abs(density - 0.35) / 0.35
                        capability += density_score * 0.2
                    except:
                        capability += 0.1  # Small bonus for having grid
            
            # Add some randomness based on genome ID for diversity
            if hasattr(genome, 'id'):
                genome_hash = hash(genome.id) % 1000
                capability += (genome_hash / 1000.0) * 0.1
            
            # Clamp to reasonable range
            return max(0.0, min(capability, 1.0))
            
        except Exception as e:
            print(f"   ❌ Capability estimation failed: {e}")
            raise RuntimeError(
                f"FAIL-FAST: Cannot estimate genome capability: {e}. "
                f"No default capability values allowed - fix the capability estimation."
            )
    
    def _extract_generation_from_genome_id(self, genome_id: str) -> int:
        """Extract generation number from genome ID like 'gen4_cross_9416x9416_5650'."""
        try:
            if genome_id.startswith('gen'):
                # Extract number after 'gen' prefix
                gen_part = genome_id.split('_')[0]  # 'gen4'
                return int(gen_part[3:])  # Extract '4' from 'gen4'
            return 0  # Default for unknown format
        except (ValueError, IndexError):
            return 0  # Fallback for parsing errors
    
    def _load_test_cases_for_problem(self, problem_name: str) -> Optional[str]:
        """Load test cases for a specific problem using config-driven dataset path."""
        
        print(f"🔍 Loading test cases for '{problem_name}'...")
        
        # FIX: Get dataset path using proper PathConfig creation
        from coral.config.path_utils import create_path_config_from_dict
        
        # Determine executor type from config
        executor_type = self.config.get('infra', {}).get('executor', 'local')
        
        try:
            path_config = create_path_config_from_dict(self.config, executor_type)
            dataset_path = path_config.dataset
        except Exception as path_error:
            # Fallback: try to get dataset path directly from experiment config
            print(f"⚠️  PathConfig creation failed: {path_error}")
            experiment_config = self.config.get('experiment', {})
            dataset_config = experiment_config.get('dataset', {})
            dataset_path = dataset_config.get('path', '/cache/quixbugs_dataset')
            print(f"   Using fallback dataset path: {dataset_path}")
        
        if not Path(dataset_path).exists():
            # FAIL-FAST: No fallback paths allowed  
            raise ValueError(
                f"FAIL-FAST: Dataset not found at configured path: {dataset_path}. "
                f"No fallback paths allowed - ensure dataset is properly configured and available."
            )
        else:
            print(f"   ✅ Found dataset at: {dataset_path}")
        
        # Look for test cases with the working naming convention
        test_locations = [
            Path(dataset_path) / "python_testcases" / f"test_{problem_name}.py",
            Path(dataset_path) / "python_programs" / f"{problem_name}_test.py",
        ]
        
        for test_path in test_locations:
            if test_path.exists():
                content = test_path.read_text()
                print(f"   ✅ Test cases loaded: {len(content)} chars")
                return content
        
        # Enhanced error with directory listing
        dataset_root = Path(dataset_path)
        print(f"🔍 Available test files in {dataset_path}/python_testcases:")
        try:
            testcases_dir = dataset_root / "python_testcases"
            if testcases_dir.exists():
                test_files = list(testcases_dir.glob("test_*.py"))[:10]
                for test_file in test_files:
                    print(f"   📄 {test_file.name}")
                if len(list(testcases_dir.glob("test_*.py"))) > 10:
                    print(f"   ... and {len(list(testcases_dir.glob('test_*.py'))) - 10} more")
        except Exception as e:
            print(f"   ❌ Could not list directory: {e}")
        
        # EVOLUTIONARY PRESSURE: Return None instead of failing
        print(f"⚠️  Test cases not found for '{problem_name}' in {dataset_path}")
        print(f"   • This will be handled as evolutionary pressure via low scores")
        return None  # Return None instead of failing hard


class QuixBugsCodeLlamaRealPlugin:
    """Main plugin class - config-driven."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Validate required config sections
        if 'experiment' not in config:
            raise ValueError(f"FAIL-FAST: Required config section 'experiment' not found")
        
        experiment_config = config['experiment']
        required_exp_sections = ['dataset', 'model']
        for section in required_exp_sections:
            if section not in experiment_config:
                raise ValueError(f"FAIL-FAST: Required experiment section '{section}' not found")
        
        print(f"🔌 QuixBugs + CodeLlama plugin initialized")
        print(f"   📁 Dataset: {experiment_config['dataset'].get('path', 'not specified')}")
        print(f"   🤖 Model: {experiment_config['model'].get('name', 'not specified')}")
    
    def get_modal_config(self, coral_config) -> Dict[str, Any]:
        """Get Modal-compatible configuration with all necessary sections."""
        return {
            'evo': self.config['evo'],  # Raw evo config with all fields including target_modules
            'execution': coral_config.execution,
            'experiment': coral_config.experiment,
            'infra': coral_config.infra,
            'cache': coral_config.cache,
            'threshold': {
                'base_thresholds': coral_config.threshold.base_thresholds.__dict__,
                'max_thresholds': coral_config.threshold.max_thresholds.__dict__,
                'schedule': coral_config.threshold.schedule
            },
            'evaluation': coral_config.evaluation,
            'seed': coral_config.seed,
            'adapter_type': getattr(coral_config, 'adapter_type', 'lora'),
            'paths': self.config.get('paths', {}),  # Include paths for dataset access
            'cheap_knobs': self.config.get('cheap_knobs', {}),  # Include cheap knobs config
            'training': self.config.get('training', {})  # Include training config
        }
    
    def dataset(self) -> DatasetProvider:
        """Create dataset provider from config."""
        return QuixBugsRealDataset(self.config)
    
    def model_factory(self) -> Callable[[LoRAConfig], ModelRunner]:
        """Create model factory from config."""
        def create_model(lora_cfg: LoRAConfig, genome: Genome = None) -> ModelRunner:
            return CodeLlamaRealRunner(lora_cfg, self.config, genome=genome)
        return create_model
    
    def fitness_fn(self) -> FitnessFn:
        """Create fitness function from config."""
        return QuixBugsRealFitness(self.config) 