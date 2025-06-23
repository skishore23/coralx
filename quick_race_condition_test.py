#!/usr/bin/env python3
"""
Quick reproduction test for Modal volume race condition.
Run this to trigger the issue faster than waiting for generation 1.
"""

def quick_race_test():
    """Test that can trigger the race condition within minutes."""
    import modal
    import time
    from pathlib import Path
    
    # Create minimal Modal app for testing
    app = modal.App("race-condition-test")
    volume = modal.Volume.from_name("coral-x-clean-cache")
    
    @app.function(volumes={"/cache": volume})
    def simulate_training():
        """Simulate training function saving an adapter."""
        import os
        import time
        
        # Create a fake adapter directory
        adapter_path = Path("/cache/adapters/adapter_test_race_condition")
        adapter_path.mkdir(parents=True, exist_ok=True)
        
        # Create fake adapter files
        (adapter_path / "adapter_config.json").write_text('{"test": "data"}')
        (adapter_path / "adapter_model.safetensors").write_bytes(b"fake_model_data")
        
        print(f"✅ Training: Created adapter at {adapter_path}")
        
        # Commit volume changes
        volume.commit()
        print(f"✅ Training: Committed volume changes")
        
        return str(adapter_path)
    
    @app.function(volumes={"/cache": volume})
    def simulate_generation():
        """Simulate generation function trying to access adapter."""
        import time
        
        adapter_path = Path("/cache/adapters/adapter_test_race_condition")
        
        print(f"🔍 Generation: Looking for adapter at {adapter_path}")
        
        if adapter_path.exists():
            print(f"✅ Generation: Found adapter immediately")
            return "success_immediate"
        
        print(f"❌ Generation: Adapter not found (race condition!)")
        print(f"🔄 Generation: Trying volume.reload()...")
        
        volume.reload()
        time.sleep(2)
        
        if adapter_path.exists():
            print(f"✅ Generation: Found adapter after reload")
            return "success_after_reload"
        else:
            print(f"❌ Generation: Still not found after reload")
            return "failed"
    
    @app.local_entrypoint()
    def main():
        # Run training and generation in quick succession
        print("🏋️ Starting training...")
        training_result = simulate_training.remote()
        
        # Start generation immediately (race condition)
        print("🎯 Starting generation immediately...")
        generation_result = simulate_generation.remote()
        
        print(f"Results:")
        print(f"  Training: {training_result}")
        print(f"  Generation: {generation_result}")
        
        if generation_result == "failed":
            print("❌ RACE CONDITION REPRODUCED!")
        else:
            print("✅ Race condition avoided (possibly due to timing)")

if __name__ == "__main__":
    quick_race_test()
