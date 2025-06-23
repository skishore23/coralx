#!/usr/bin/env python3
"""
CORAL-X Evolution Monitor
Monitor live evolution progress and inspect test execution details.
"""
import modal
import time
import json
from pathlib import Path

def monitor_coral_evolution():
    """Monitor the live CORAL-X evolution and show detailed test execution."""
    print("🔍 CORAL-X Evolution Monitor")
    print("=" * 60)
    
    try:
        # Connect to the running Modal app
        app_name = "coral-x-production"
        
        # Check if we can connect to the running experiment
        print(f"📡 Connecting to Modal app: {app_name}")
        
        # Get recent evolution reports
        results_dir = Path("results/evolution")
        if results_dir.exists():
            report_files = sorted(results_dir.glob("coral_evolution_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
            
            if report_files:
                latest_report = report_files[0]
                print(f"📋 Latest Report: {latest_report.name}")
                
                with open(latest_report) as f:
                    report_data = json.load(f)
                
                print(f"⏱️  Experiment Time: {report_data.get('experiment_time', 0):.2f}s")
                print(f"✅ Success: {report_data.get('success', False)}")
                
                if 'error' in report_data:
                    print(f"❌ Error: {report_data['error']}")
                
                print(f"\n📊 Parameters:")
                params = report_data.get('parameters', {})
                for key, value in params.items():
                    print(f"   • {key}: {value}")
            else:
                print("⚠️  No evolution reports found")
        
        # Try to inspect Modal logs
        print(f"\n🔍 Checking Modal app status...")
        import subprocess
        result = subprocess.run(['modal', 'app', 'list'], capture_output=True, text=True)
        
        if app_name in result.stdout:
            print(f"✅ Modal app '{app_name}' is running")
            
            # Show recent logs
            print(f"\n📜 Recent Modal logs:")
            log_result = subprocess.run(['modal', 'app', 'logs', app_name, '--lines', '50'], 
                                       capture_output=True, text=True)
            
            if log_result.stdout:
                # Filter for interesting log lines
                lines = log_result.stdout.split('\n')
                interesting_lines = []
                
                keywords = ['🧬', '🤖', '📊', '✅', '❌', 'EVALUATION', 'Generated', 'Test', 'Score', 'Bugfix', 'Style', 'Security', 'Runtime']
                
                for line in lines[-30:]:  # Last 30 lines
                    if any(keyword in line for keyword in keywords):
                        interesting_lines.append(line)
                
                if interesting_lines:
                    print("📋 Key events:")
                    for line in interesting_lines[-10:]:  # Show last 10 interesting lines
                        print(f"   {line}")
                else:
                    print("   (No detailed logs available yet)")
            else:
                print("   (No logs available)")
        else:
            print(f"⚠️  Modal app '{app_name}' not found in running apps")
        
        # Show system status
        print(f"\n🎯 Expected Evolution Progress:")
        print(f"   1. Initial Population: 32 genomes")
        print(f"   2. Cache Miss → LoRA Training (2-5 min per unique adapter)")
        print(f"   3. Code Generation per problem")
        print(f"   4. Multi-objective evaluation:")
        print(f"      • Bugfix: Test pass rate + compilation")
        print(f"      • Style: Code quality metrics")  
        print(f"      • Security: Safety checks")
        print(f"      • Runtime: Performance analysis")
        print(f"   5. Threshold gate filtering")
        print(f"   6. Next generation...")
        
    except Exception as e:
        print(f"❌ Monitor error: {e}")


if __name__ == "__main__":
    monitor_coral_evolution() 