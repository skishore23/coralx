"""
Simple Integration for Emergent Behavior Detection in CORAL-X

This module provides a simple way to track emergent behaviors during
LoRA testing without complex infrastructure. Focus on spotting patterns
from current training logs.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

from .emergent_behavior import (
    SimpleBehavior, SimplePattern,
    detect_simple_emergent_behaviors,
    should_alert_simple_behavior, 
    format_simple_behavior_alert
)


class SimpleEmergentTracker:
    """
    Simple tracker for emergent behaviors - minimal overhead, maximum insight.
    
    Tracks patterns across generations without complex state management.
    """
    
    def __init__(self, output_dir: Path = Path("results/emergent_simple")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Simple in-memory tracking
        self.detected_behaviors: List[SimpleBehavior] = []
        self.generation_stats: Dict[int, Dict[str, int]] = {}
    
    def track_evaluation(
        self,
        problem_name: str,
        genome_id: str,
        generation: int,
        ca_features: Dict[str, Any],
        lora_config: Dict[str, Any],
        evaluation_result: Dict[str, Any],
        generated_code: str
    ) -> List[SimpleBehavior]:
        """
        Track a single evaluation for emergent behaviors.
        
        This is the main integration point - call this after each evaluation.
        """
        # Detect behaviors for this evaluation
        behaviors = detect_simple_emergent_behaviors(
            problem_name=problem_name,
            genome_id=genome_id,
            generation=generation,
            ca_features=ca_features,
            lora_config=lora_config,
            evaluation_result=evaluation_result,
            generated_code=generated_code
        )
        
        # Store detected behaviors
        self.detected_behaviors.extend(behaviors)
        
        # Update generation stats
        if generation not in self.generation_stats:
            self.generation_stats[generation] = {
                "total_evaluations": 0,
                "behaviors_detected": 0,
                "elegant_solutions": 0,
                "efficient_adaptations": 0,
                "late_breakthroughs": 0,
                "pythonic_evolutions": 0
            }
        
        self.generation_stats[generation]["total_evaluations"] += 1
        self.generation_stats[generation]["behaviors_detected"] += len(behaviors)
        
        for behavior in behaviors:
            self.generation_stats[generation][f"{behavior.behavior_type}s"] += 1
        
        # Alert on interesting behaviors
        for behavior in behaviors:
            if should_alert_simple_behavior(behavior):
                print(format_simple_behavior_alert(behavior))
        
        # Log progress every 20 evaluations
        total_evals = sum(stats["total_evaluations"] for stats in self.generation_stats.values())
        if total_evals % 20 == 0:
            self._save_progress_log()
        
        return behaviors
    
    def get_generation_summary(self, generation: int) -> Dict[str, Any]:
        """Get summary statistics for a generation."""
        if generation not in self.generation_stats:
            return {"error": f"No data for generation {generation}"}
        
        stats = self.generation_stats[generation]
        behavior_rate = stats["behaviors_detected"] / max(1, stats["total_evaluations"])
        
        return {
            "generation": generation,
            "total_evaluations": stats["total_evaluations"],
            "behaviors_detected": stats["behaviors_detected"],
            "behavior_rate": behavior_rate,
            "breakdown": {
                "elegant_solutions": stats.get("elegant_solutions", 0),
                "efficient_adaptations": stats.get("efficient_adaptations", 0), 
                "late_breakthroughs": stats.get("late_breakthroughs", 0),
                "pythonic_evolutions": stats.get("pythonic_evolutions", 0)
            }
        }
    
    def print_progress_summary(self) -> None:
        """Print a quick progress summary for current training."""
        if not self.generation_stats:
            print("📊 No emergent behavior data yet")
            return
        
        latest_gen = max(self.generation_stats.keys())
        total_behaviors = len(self.detected_behaviors)
        total_evals = sum(stats["total_evaluations"] for stats in self.generation_stats.values())
        
        print(f"\n📊 EMERGENT BEHAVIOR PROGRESS SUMMARY")
        print(f"{'─'*50}")
        print(f"   • Latest generation: {latest_gen}")
        print(f"   • Total evaluations: {total_evals}")
        print(f"   • Total behaviors detected: {total_behaviors}")
        print(f"   • Detection rate: {total_behaviors/max(1,total_evals):.1%}")
        
        # Show recent generation stats
        recent_gens = sorted(self.generation_stats.keys())[-3:]
        for gen in recent_gens:
            summary = self.get_generation_summary(gen)
            print(f"   • Gen {gen}: {summary['behaviors_detected']}/{summary['total_evaluations']} " +
                  f"({summary['behavior_rate']:.1%} rate)")
    
    def save_simple_report(self) -> Path:
        """Save a simple JSON report of all detected behaviors."""
        report_data = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "summary": {
                "total_behaviors": len(self.detected_behaviors),
                "total_evaluations": sum(stats["total_evaluations"] for stats in self.generation_stats.values()),
                "generations_tracked": len(self.generation_stats)
            },
            "generation_stats": self.generation_stats,
            "recent_behaviors": [
                {
                    "behavior_type": b.behavior_type,
                    "generation": b.generation,
                    "problem": b.problem_name,
                    "confidence": b.confidence,
                    "description": b.description,
                    "evidence": b.evidence
                }
                for b in self.detected_behaviors[-10:]  # Last 10 behaviors
            ]
        }
        
        report_file = self.output_dir / f"simple_emergent_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"📄 Simple emergent behavior report saved: {report_file}")
        return report_file
    
    def _save_progress_log(self) -> None:
        """Save periodic progress log."""
        log_file = self.output_dir / "progress_log.json"
        log_data = {
            "last_updated": time.strftime('%Y-%m-%d %H:%M:%S'),
            "generation_stats": self.generation_stats,
            "total_behaviors": len(self.detected_behaviors),
            "latest_generation": max(self.generation_stats.keys()) if self.generation_stats else 0
        }
        
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)


# =============================================================================
# SIMPLE INTEGRATION FUNCTIONS
# =============================================================================

def add_simple_emergent_tracking_to_evaluation(
    evaluation_result: Dict[str, Any],
    problem_name: str,
    genome_id: str,
    generation: int,
    ca_features: Dict[str, Any],
    lora_config: Dict[str, Any],
    generated_code: str,
    tracker: SimpleEmergentTracker
) -> Dict[str, Any]:
    """
    Add simple emergent behavior tracking to existing evaluation results.
    
    This is the minimal integration point - just add this call after evaluation.
    """
    # Track this evaluation
    behaviors = tracker.track_evaluation(
        problem_name=problem_name,
        genome_id=genome_id,
        generation=generation,
        ca_features=ca_features,
        lora_config=lora_config,
        evaluation_result=evaluation_result,
        generated_code=generated_code
    )
    
    # Enhance evaluation result with simple behavior info
    enhanced_result = evaluation_result.copy()
    enhanced_result["emergent_behaviors"] = {
        "count": len(behaviors),
        "types": [b.behavior_type for b in behaviors],
        "max_confidence": max([0.0] + [b.confidence for b in behaviors])
    }
    
    return enhanced_result


def quick_behavior_check(
    evaluation_result: Dict[str, Any],
    generated_code: str,
    generation: int = 0
) -> str:
    """
    Quick check for interesting patterns without full tracking.
    
    Use this for one-off analysis or debugging.
    """
    # Minimal CA features and LoRA config for testing
    dummy_ca_features = {"pattern_complexity": 0.5}
    dummy_lora_config = {"r": 8, "alpha": 16.0}  # 🔥 FIXED: Use 'r' not 'rank'
    
    behaviors = detect_simple_emergent_behaviors(
        problem_name="test",
        genome_id="debug",
        generation=generation,
        ca_features=dummy_ca_features,
        lora_config=dummy_lora_config,
        evaluation_result=evaluation_result,
        generated_code=generated_code
    )
    
    if behaviors:
        return f"🌟 Found {len(behaviors)} interesting patterns: " + \
               ", ".join(b.behavior_type for b in behaviors)
    else:
        return "📊 No particularly interesting patterns detected"


def create_simple_config() -> Dict[str, Any]:
    """Create minimal config for emergent behavior tracking."""
    return {
        "emergent_tracking": {
            "enabled": True,
            "output_dir": "results/emergent_simple",
            "alert_threshold": 0.8,
            "save_frequency": 20  # Save progress every N evaluations
        }
    } 