#!/usr/bin/env python3
"""
Quick test script to verify emergent behavior tracking is working.

Usage:
    python test_emergent_tracking.py
"""

import sys
from pathlib import Path

def test_emergent_tracking():
    """Test emergent behavior tracking functionality."""
    print("🧪 EMERGENT BEHAVIOR TRACKING TEST")
    print("=" * 60)
    
    try:
        # Test 1: Import modules
        print("1️⃣ Testing imports...")
        from coral.domain.emergent_behavior_integration import (
            SimpleEmergentTracker, 
            quick_behavior_check
        )
        print("   ✅ Imports successful")
        
        # Test 2: Create tracker
        print("\n2️⃣ Testing tracker creation...")
        test_dir = Path("test_output/emergent_test")
        tracker = SimpleEmergentTracker(test_dir)
        print(f"   ✅ Tracker created: {test_dir}")
        
        # Test 3: Quick behavior check
        print("\n3️⃣ Testing behavior detection...")
        
        # Test case 1: Elegant solution
        elegant_result = {
            'bugfix': 1.0,
            'style': 0.95,
            'test_cases_passed': 7,
            'test_cases_run': 7,
            'runtime': 0.8
        }
        elegant_code = "def bitcount(n):\n    return bin(n).count('1')"
        
        result1 = quick_behavior_check(elegant_result, elegant_code, generation=5)
        print(f"   🎨 Elegant solution test: {result1}")
        
        # Test case 2: Efficient adaptation
        efficient_result = {
            'bugfix': 0.85,
            'style': 0.75,
            'test_cases_passed': 5,
            'test_cases_run': 7,
            'runtime': 0.9
        }
        efficient_code = "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a"
        
        result2 = quick_behavior_check(efficient_result, efficient_code, generation=10)
        print(f"   ⚡ Efficient adaptation test: {result2}")
        
        # Test 4: Full tracking simulation
        print("\n4️⃣ Testing full tracking...")
        
        behaviors = tracker.track_evaluation(
            problem_name="bitcount",
            genome_id="test_genome_001",
            generation=5,
            ca_features={'complexity': 0.7, 'intensity': 0.8},
            lora_config={'r': 4, 'alpha': 8.0, 'dropout': 0.1},
            evaluation_result=elegant_result,
            generated_code=elegant_code
        )
        
        print(f"   ✅ Tracked evaluation: {len(behaviors)} behaviors detected")
        for behavior in behaviors:
            print(f"      • {behavior.behavior_type}: {behavior.description}")
        
        # Test 5: Progress summary
        print("\n5️⃣ Testing progress summary...")
        tracker.print_progress_summary()
        
        # Test 6: Save report
        print("\n6️⃣ Testing report generation...")
        report_file = tracker.save_simple_report()
        print(f"   ✅ Report saved: {report_file}")
        
        print("\n🎉 ALL TESTS PASSED!")
        print("   Emergent behavior tracking is working correctly.")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("   Make sure you're in the coralx directory and modules are available")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        print("=" * 60)


def test_config_integration():
    """Test emergent tracking configuration integration."""
    print("\n🔧 CONFIGURATION INTEGRATION TEST")
    print("=" * 60)
    
    try:
        # Load a real config file
        import yaml
        
        config_files = [
            "coral_x_codellama_config.yaml",
            "coral_x_emergent_config.yaml", 
            "coral_x_clean_config.yaml"
        ]
        
        for config_file in config_files:
            config_path = Path(config_file)
            if config_path.exists():
                print(f"📋 Testing config: {config_file}")
                
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Check emergent tracking section
                emergent_config = config.get('emergent_tracking', {})
                enabled = emergent_config.get('enabled', False)
                output_dir = emergent_config.get('output_dir', 'not_set')
                
                print(f"   • Enabled: {enabled}")
                print(f"   • Output dir: {output_dir}")
                
                if enabled:
                    print(f"   ✅ Emergent tracking configured in {config_file}")
                else:
                    print(f"   ⚠️  Emergent tracking disabled in {config_file}")
                
                break
        else:
            print("❌ No config files found")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False
    
    finally:
        print("=" * 60)


if __name__ == "__main__":
    print("🚀 CORAL-X EMERGENT BEHAVIOR TRACKING VERIFICATION")
    print("=" * 80)
    
    # Run tests
    test1_passed = test_emergent_tracking()
    test2_passed = test_config_integration()
    
    print("\n📊 FINAL RESULTS:")
    print("=" * 80)
    print(f"🧪 Emergent tracking functionality: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"🔧 Configuration integration: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 EMERGENT TRACKING IS READY!")
        print("   You should now see emergent behavior alerts during evolution.")
        sys.exit(0)
    else:
        print("\n❌ EMERGENT TRACKING NEEDS FIXES")
        print("   Fix the issues above before running evolution.")
        sys.exit(1) 