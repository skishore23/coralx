# Simple Emergent Behavior Tracking for CORAL-X

## ✅ Ready to Use with Your Current Runs!

The simplified emergent behavior tracking system is now **integrated directly into your existing CORAL-X evaluation pipeline**. No separate examples needed - just enable it in your config!

## 🔧 How to Enable (2 Steps)

### Step 1: Add to Your Config

Add this section to any of your existing `coral_x_*.yaml` config files:

```yaml
# ADD THIS to your existing config
emergent_tracking:
  enabled: true                          # Enable tracking
  output_dir: "results/emergent_simple"  # Where to save data
  alert_threshold: 0.8                   # Confidence for alerts
  save_frequency: 20                     # Save every N evaluations
```

### Step 2: Run Your Normal Evolution

```bash
# Your normal CORAL-X command - no changes needed!
python run_coral_x_evolution.py --config coral_x_your_config.yaml
```

That's it! The tracking will automatically run alongside your regular evolution.

## 📊 What You'll See

### Real-time Alerts During Evolution

```
🌟 EMERGENT BEHAVIOR: elegant_solution
   • Generation: 15
   • Problem: bitcount
   • Genome: genome_a1b2c3d4
   • Confidence: 0.90
   • Description: Perfect solution with concise, elegant code
   • Evidence: {'perfect_fix': 'Perfect solution: 7/7 tests passed', ...}
```

### Progress Summaries

```
📊 EMERGENT BEHAVIOR PROGRESS SUMMARY
──────────────────────────────────────────────────
   • Latest generation: 15
   • Total evaluations: 320
   • Total behaviors detected: 47
   • Detection rate: 14.7%
   • Gen 13: 3/20 (15.0% rate)
   • Gen 14: 5/20 (25.0% rate) 
   • Gen 15: 2/20 (10.0% rate)
```

### Saved Reports

JSON files saved in `results/emergent_simple/`:
- `progress_log.json` - Updated every 20 evaluations
- `simple_emergent_report_*.json` - Full reports with all detected behaviors

## 🎯 What It Detects

### 4 Types of Simple Emergent Behaviors:

1. **🎨 Elegant Solutions** - Perfect fixes with concise code
   - **Triggers**: 100% test pass rate + ≤5 lines of code
   - **Example**: One-liner bit manipulation solutions

2. **⚡ Efficient Adaptation** - High performance with low-rank LoRA
   - **Triggers**: High performance (>0.7) + low LoRA rank (≤4)
   - **Example**: Rank-4 LoRA achieving 90% bugfix rate

3. **🐍 Pythonic Evolution** - Using built-in functions effectively
   - **Triggers**: Uses built-ins (map, filter, etc.) + high performance
   - **Example**: Evolution discovers `bin(n).count('1')` for bit counting

4. **🚀 Late Breakthrough** - Unexpected success in late generations
   - **Triggers**: High performance (>0.8) after generation 20
   - **Example**: Algorithm breakthrough in generation 35

## 📈 Integration Points

The tracking hooks into your existing evaluation at these points:

```python
# In plugins/quixbugs_codellama/plugin.py - ALREADY INTEGRATED
evaluation_result = evaluate_quixbugs_code(generated_code, problem, test_cases)

# ADD SIMPLE EMERGENT BEHAVIOR TRACKING (ALREADY DONE!)
if self.emergent_tracker:
    self.emergent_tracker.track_evaluation(...)
```

## 🔍 Quick Debug Check

Test the system works before running full evolution:

```python
from coral.domain.emergent_behavior_integration import quick_behavior_check

evaluation_result = {'bugfix': 0.95, 'style': 0.85, 'test_cases_passed': 5, 'test_cases_run': 5}
test_code = "def bitcount(n):\n    return bin(n).count('1')"

result = quick_behavior_check(evaluation_result, test_code)
print(result)  # "🌟 Found 1 interesting patterns: elegant_solution"
```

## 📁 Files Modified

- ✅ `coral/domain/emergent_behavior.py` - Simple pattern detection
- ✅ `coral/domain/emergent_behavior_integration.py` - Simple tracker  
- ✅ `plugins/quixbugs_codellama/plugin.py` - Integration added
- ✅ `coral_x_emergent_config.yaml` - Example config

## 🎯 Production Benefits

### For Current Training Runs:
- **Zero overhead** - Runs alongside normal evaluation
- **Real-time insights** - See interesting patterns as they emerge
- **No workflow changes** - Just add config section and run normally

### For Research:
- **Evolution patterns** - Track how behaviors emerge over generations
- **Parameter insights** - See which CA/LoRA combinations work best
- **Publication data** - Quantitative evidence of emergent capabilities

## 🚀 Next Steps

1. **Add config section** to your favorite `coral_x_*.yaml` file
2. **Run normal evolution** - tracking happens automatically
3. **Check `results/emergent_simple/`** for saved reports
4. **Tune thresholds** if you want more/fewer alerts

The system follows CORAL-X principles:
- ✅ **Pure functions** - No side effects in detection logic
- ✅ **Fail-fast** - Clear errors, no silent fallbacks  
- ✅ **Immutable data** - All tracking data is frozen dataclasses
- ✅ **Small functions** - Simple, composable detection functions

**Ready to track emergent behaviors in your current CORAL-X evolution runs!** 🚀 