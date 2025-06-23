"""
Simple Emergent Behavior Detection for CORAL-X - Focus on Current Logs

This module provides simple functions for detecting basic emergent behaviors
that can be spotted from current CORAL-X training logs and evaluation results.
"""

import re
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass(frozen=True)
class SimplePattern:
    """Simple code pattern that can be easily detected."""
    name: str
    pattern_type: str  # 'code_structure', 'performance', 'evolution'
    value: float
    description: str


@dataclass(frozen=True)
class SimpleBehavior:
    """Simple emergent behavior detection."""
    behavior_id: str
    behavior_type: str
    generation: int
    genome_id: str
    problem_name: str
    confidence: float
    description: str
    evidence: Dict[str, Any]


def detect_simple_code_patterns(code: str, problem_name: str) -> List[SimplePattern]:
    """
    Detect simple code patterns that indicate interesting behavior.
    
    Pure function: str × str → List[SimplePattern]
    """
    patterns = []
    
    if not code or not code.strip():
        return patterns
    
    # Pattern 1: Recursion usage
    if re.search(r'def\s+(\w+).*:\s*.*\1\s*\(', code, re.DOTALL):
        patterns.append(SimplePattern(
            name="recursion",
            pattern_type="code_structure", 
            value=1.0,
            description="Uses recursion"
        ))
    
    # Pattern 2: Loop complexity (nested loops)
    loop_count = len(re.findall(r'\bfor\b|\bwhile\b', code))
    if loop_count > 1:
        patterns.append(SimplePattern(
            name="complex_loops",
            pattern_type="code_structure",
            value=min(1.0, loop_count / 3.0),
            description=f"Has {loop_count} loops"
        ))
    
    # Pattern 3: Built-in function usage (pythonic solutions)
    builtin_funcs = ['map', 'filter', 'reduce', 'zip', 'enumerate', 'any', 'all']
    builtin_count = sum(1 for func in builtin_funcs if func in code)
    if builtin_count > 0:
        patterns.append(SimplePattern(
            name="pythonic_builtins",
            pattern_type="code_structure",
            value=min(1.0, builtin_count / 3.0),
            description=f"Uses {builtin_count} built-in functions"
        ))
    
    # Pattern 4: Code length (conciseness vs verbosity)
    lines = len([l for l in code.split('\n') if l.strip()])
    if lines <= 5:
        patterns.append(SimplePattern(
            name="concise_solution",
            pattern_type="code_structure",
            value=1.0,
            description=f"Very concise ({lines} lines)"
        ))
    elif lines > 20:
        patterns.append(SimplePattern(
            name="verbose_solution", 
            pattern_type="code_structure",
            value=0.7,
            description=f"Verbose solution ({lines} lines)"
        ))
    
    return patterns


def detect_performance_patterns(evaluation_result: Dict[str, Any]) -> List[SimplePattern]:
    """
    Detect performance-based patterns from evaluation results.
    
    Pure function: Dict → List[SimplePattern]
    """
    patterns = []
    
    bugfix_score = evaluation_result.get('bugfix', 0.0)
    style_score = evaluation_result.get('style', 0.0)
    runtime_score = evaluation_result.get('runtime', 0.0)
    tests_passed = evaluation_result.get('test_cases_passed', 0)
    tests_total = evaluation_result.get('test_cases_run', 1)
    
    # Pattern 1: Perfect bug fix
    if bugfix_score >= 0.95 and tests_passed == tests_total and tests_total > 0:
        patterns.append(SimplePattern(
            name="perfect_fix",
            pattern_type="performance",
            value=1.0,
            description=f"Perfect solution: {tests_passed}/{tests_total} tests passed"
        ))
    
    # Pattern 2: High performance across all metrics
    avg_score = (bugfix_score + style_score + runtime_score) / 3.0
    if avg_score > 0.8:
        patterns.append(SimplePattern(
            name="high_performance",
            pattern_type="performance", 
            value=avg_score,
            description=f"High overall performance: {avg_score:.3f}"
        ))
    
    # Pattern 3: Style-focused solution (high style, moderate bugfix)
    if style_score > 0.9 and 0.4 <= bugfix_score <= 0.7:
        patterns.append(SimplePattern(
            name="style_focused",
            pattern_type="performance",
            value=style_score,
            description="Prioritizes code style over pure functionality"
        ))
    
    # Pattern 4: Partial but consistent success
    if 0.3 <= bugfix_score <= 0.7 and tests_passed > 0:
        pass_rate = tests_passed / tests_total if tests_total > 0 else 0
        if pass_rate >= 0.3:  # At least 30% tests passing
            patterns.append(SimplePattern(
                name="partial_success",
                pattern_type="performance",
                value=pass_rate,
                description=f"Consistent partial success: {pass_rate:.1%} pass rate"
            ))
    
    return patterns


def detect_evolution_patterns(
    current_generation: int,
    ca_features: Dict[str, Any],
    lora_config: Dict[str, Any],
    evaluation_result: Dict[str, Any]
) -> List[SimplePattern]:
    """
    Detect evolution-specific patterns from CA features and LoRA config.
    
    Pure function: int × Dict × Dict × Dict → List[SimplePattern]
    """
    patterns = []
    
    # Pattern 1: Low-rank high-performance (potential synergy)
    lora_rank = lora_config.get('r', 8)  # 🔥 FIXED: Use 'r' not 'rank'
    bugfix_score = evaluation_result.get('bugfix', 0.0)
    
    if lora_rank <= 4 and bugfix_score > 0.7:
        patterns.append(SimplePattern(
            name="low_rank_success",
            pattern_type="evolution",
            value=bugfix_score,
            description=f"High performance with low LoRA rank ({lora_rank})"
        ))
    
    # Pattern 2: CA complexity correlation
    ca_complexity = ca_features.get('pattern_complexity', 0.0)
    if ca_complexity > 0.7 and bugfix_score > 0.6:
        patterns.append(SimplePattern(
            name="complex_ca_success", 
            pattern_type="evolution",
            value=ca_complexity,
            description=f"Complex CA features correlate with success"
        ))
    
    # Pattern 3: Late generation breakthrough
    if current_generation > 20 and bugfix_score > 0.8:
        patterns.append(SimplePattern(
            name="late_breakthrough",
            pattern_type="evolution", 
            value=bugfix_score,
            description=f"Breakthrough in late generation {current_generation}"
        ))
    
    # Pattern 4: Evolution progress indicator
    generation_progress = min(1.0, current_generation / 40.0)  # Assuming 40 generations
    if bugfix_score > generation_progress + 0.2:  # Performing better than expected
        patterns.append(SimplePattern(
            name="ahead_of_schedule",
            pattern_type="evolution",
            value=bugfix_score - generation_progress,
            description=f"Performing ahead of expected progress curve"
        ))
    
    return patterns


def detect_simple_emergent_behaviors(
    problem_name: str,
    genome_id: str,
    generation: int,
    ca_features: Dict[str, Any],
    lora_config: Dict[str, Any],
    evaluation_result: Dict[str, Any],
    generated_code: str
) -> List[SimpleBehavior]:
    """
    Main function to detect simple emergent behaviors.
    
    This is the simplified entry point that replaces the complex detection pipeline.
    """
    behaviors = []
    
    # Get all patterns
    code_patterns = detect_simple_code_patterns(generated_code, problem_name)
    perf_patterns = detect_performance_patterns(evaluation_result)
    evo_patterns = detect_evolution_patterns(generation, ca_features, lora_config, evaluation_result)
    
    all_patterns = code_patterns + perf_patterns + evo_patterns
    
    # Look for interesting combinations that suggest emergent behavior
    
    # Behavior 1: Perfect solution with simple code
    perfect_patterns = [p for p in perf_patterns if p.name == "perfect_fix"]
    concise_patterns = [p for p in code_patterns if p.name == "concise_solution"]
    
    if perfect_patterns and concise_patterns:
        behaviors.append(SimpleBehavior(
            behavior_id=f"elegant_solution_{genome_id}_{generation}",
            behavior_type="elegant_solution",
            generation=generation,
            genome_id=genome_id,
            problem_name=problem_name,
            confidence=0.9,
            description="Perfect solution with concise, elegant code",
            evidence={
                "perfect_fix": perfect_patterns[0].description,
                "concise_code": concise_patterns[0].description,
                "bugfix_score": evaluation_result.get('bugfix', 0.0)
            }
        ))
    
    # Behavior 2: Low-rank LoRA achieving high performance
    low_rank_patterns = [p for p in evo_patterns if p.name == "low_rank_success"]
    if low_rank_patterns:
        behaviors.append(SimpleBehavior(
            behavior_id=f"efficient_adaptation_{genome_id}_{generation}",
            behavior_type="efficient_adaptation",
            generation=generation,
            genome_id=genome_id,
            problem_name=problem_name,
            confidence=0.8,
            description="High performance with efficient (low-rank) adaptation",
            evidence={
                "lora_rank": lora_config.get('r', 8),  # 🔥 FIXED: Use 'r' not 'rank'
                "performance": evaluation_result.get('bugfix', 0.0),
                "efficiency_ratio": evaluation_result.get('bugfix', 0.0) / max(1, lora_config.get('r', 8))  # 🔥 FIXED: Use 'r' not 'rank'
            }
        ))
    
    # Behavior 3: Pythonic evolution (using built-ins effectively)
    pythonic_patterns = [p for p in code_patterns if p.name == "pythonic_builtins"]
    high_perf_patterns = [p for p in perf_patterns if p.name == "high_performance"]
    
    if pythonic_patterns and high_perf_patterns:
        behaviors.append(SimpleBehavior(
            behavior_id=f"pythonic_evolution_{genome_id}_{generation}",
            behavior_type="pythonic_evolution",
            generation=generation,
            genome_id=genome_id,
            problem_name=problem_name,
            confidence=0.7,
            description="Evolution towards pythonic, idiomatic solutions",
            evidence={
                "builtin_usage": pythonic_patterns[0].description,
                "performance": high_perf_patterns[0].description,
                "style_score": evaluation_result.get('style', 0.0)
            }
        ))
    
    # Behavior 4: Late-stage breakthrough
    late_patterns = [p for p in evo_patterns if p.name == "late_breakthrough"]
    if late_patterns:
        behaviors.append(SimpleBehavior(
            behavior_id=f"late_breakthrough_{genome_id}_{generation}",
            behavior_type="late_breakthrough", 
            generation=generation,
            genome_id=genome_id,
            problem_name=problem_name,
            confidence=0.8,
            description="Unexpected breakthrough in late evolution stage",
            evidence={
                "generation": generation,
                "performance": evaluation_result.get('bugfix', 0.0),
                "breakthrough_indicator": late_patterns[0].description
            }
        ))
    
    return behaviors


def should_alert_simple_behavior(behavior: SimpleBehavior) -> bool:
    """
    Simple threshold for alerting on interesting behaviors.
    
    Pure function: SimpleBehavior → bool
    """
    # Alert on high-confidence behaviors
    if behavior.confidence >= 0.8:
        return True
    
    # Always alert on perfect solutions
    if behavior.behavior_type == "elegant_solution":
        return True
    
    # Alert on late breakthroughs
    if behavior.behavior_type == "late_breakthrough" and behavior.generation > 25:
        return True
    
    return False


def format_simple_behavior_alert(behavior: SimpleBehavior) -> str:
    """
    Format behavior for logging/alerting.
    
    Pure function: SimpleBehavior → str
    """
    return f"""
🌟 EMERGENT BEHAVIOR: {behavior.behavior_type}
   • Generation: {behavior.generation}
   • Problem: {behavior.problem_name}
   • Genome: {behavior.genome_id}
   • Confidence: {behavior.confidence:.2f}
   • Description: {behavior.description}
   • Evidence: {behavior.evidence}
""" 