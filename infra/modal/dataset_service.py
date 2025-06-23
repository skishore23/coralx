"""
Modal service for dataset management.
Infrastructure layer - handles QuixBugs dataset caching and loading.
"""
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Any


def cache_quixbugs_dataset_modal():
    """Cache QuixBugs dataset in Modal volume (fail-fast approach)."""
    print("📦 Caching QuixBugs dataset in Modal...")
    
    # Use the correct Modal volume path (verified by modal volume ls)
    cache_dir = Path("/cache/quixbugs_dataset")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Clone QuixBugs repository if not already cached
    if not (cache_dir / ".git").exists():
        print("🔄 Cloning QuixBugs repository...")
        result = subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/jkoppel/QuixBugs.git",
            str(cache_dir)
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"FAIL-FAST: Failed to clone QuixBugs: {result.stderr}")
    
    # Dataset files are directly in the cache_dir
    python_dir = cache_dir / "python_programs"
    testcases_dir = cache_dir / "python_testcases"
    
    if not python_dir.exists():
        raise RuntimeError(f"FAIL-FAST: QuixBugs Python programs not found at {python_dir}")
    
    if not testcases_dir.exists():
        raise RuntimeError(f"FAIL-FAST: QuixBugs Python test cases not found at {testcases_dir}")
    
    # List available problems for debugging
    python_files = list(python_dir.glob("*.py"))
    test_files = list(testcases_dir.glob("test_*.py"))
    
    print(f"✅ QuixBugs dataset cached at {cache_dir}")
    print(f"   📁 Found {len(python_files)} Python programs")
    print(f"   🧪 Found {len(test_files)} test files")
    
    # Show sample problems for verification
    if python_files:
        sample_problems = [f.stem for f in python_files[:5]]
        print(f"   📝 Sample problems: {', '.join(sample_problems)}")
    
    return cache_dir  # Return the cache directory


def load_quixbugs_problems_modal() -> List[Dict[str, Any]]:
    """Load real QuixBugs problems (fail-fast approach)."""
    dataset_path = cache_quixbugs_dataset_modal()
    
    python_dir = dataset_path / "python_programs"
    problems = []
    
    for py_file in python_dir.glob("*.py"):
        if py_file.name.startswith("test_"):
            continue
            
        problem_name = py_file.stem
        
        # Read the buggy code
        buggy_code = py_file.read_text()
        
        # Create problem prompt
        prompt = f"""# Fix the bug in this {problem_name} function:
{buggy_code}

# Implement the corrected version:"""
        
        problems.append({
            "name": problem_name,
            "prompt": prompt,
            "buggy_code": buggy_code
        })
    
    print(f"📁 Loaded {len(problems)} real QuixBugs problems")
    return problems 