"""
Modal service for dataset management.
Infrastructure layer - handles QuixBugs dataset caching and loading.
"""
from pathlib import Path
from typing import List, Dict, Any


def setup_quixbugs_dataset_modal(config: dict = None) -> dict:
    """Setup QuixBugs dataset - returns existing dataset info since it's already cached."""
    print("📦 Setting up QuixBugs dataset...")

    # Get dataset path from config, not hardcoded
    if config:
        from core.config.path_utils import get_dataset_path
        dataset_path = Path(get_dataset_path(config))
    else:
        # Use default Modal path if no config provided
        dataset_path = Path("/cache/quixbugs_dataset")

    if not dataset_path.exists():
        return {
            'status': 'not_found',
            'error': 'QuixBugs dataset not found in Modal volume'
        }

    # Check for required directories
    python_dir = dataset_path / "python_programs"
    testcases_dir = dataset_path / "python_testcases"

    if not python_dir.exists():
        return {
            'status': 'incomplete',
            'error': f'Python programs directory not found at {python_dir}'
        }

    if not testcases_dir.exists():
        return {
            'status': 'incomplete',
            'error': f'Test cases directory not found at {testcases_dir}'
        }

    # Count available files
    python_files = list(python_dir.glob("*.py"))
    test_files = list(testcases_dir.glob("test_*.py"))

    print(f"✅ QuixBugs dataset ready at {dataset_path}")
    print(f"   📁 Found {len(python_files)} Python programs")
    print(f"   🧪 Found {len(test_files)} test files")

    return {
        'status': 'ready',
        'dataset_path': str(dataset_path),
        'python_programs': len(python_files),
        'test_files': len(test_files)
    }


def cache_quixbugs_dataset_modal(config: dict = None):
    """Use the existing QuixBugs dataset in Modal volume - simple approach."""
    print("📦 Using QuixBugs dataset from Modal volume...")

    # Get dataset path from config, not hardcoded
    if config:
        from core.config.path_utils import get_dataset_path
        dataset_path = Path(get_dataset_path(config))
    else:
        # Use default Modal path if no config provided
        dataset_path = Path("/cache/quixbugs_dataset")

    # Check if it exists
    if not dataset_path.exists():
        raise RuntimeError(
            f"  QuixBugs dataset not found at {dataset_path}. "
            f"Dataset should be pre-cached in Modal volume at this location."
        )

    # Verify required directories exist
    python_dir = dataset_path / "python_programs"
    testcases_dir = dataset_path / "python_testcases"

    if not python_dir.exists():
        raise RuntimeError(f"  Python programs not found at {python_dir}")

    if not testcases_dir.exists():
        raise RuntimeError(f"  Python test cases not found at {testcases_dir}")

    # List available problems for verification
    python_files = list(python_dir.glob("*.py"))
    test_files = list(testcases_dir.glob("test_*.py"))

    print(f"✅ QuixBugs dataset ready at {dataset_path}")
    print(f"   📁 Found {len(python_files)} Python programs")
    print(f"   🧪 Found {len(test_files)} test files")

    return dataset_path


def load_quixbugs_problems_modal(config: dict = None) -> List[Dict[str, Any]]:
    """Load real QuixBugs problems."""
    dataset_path = cache_quixbugs_dataset_modal(config=config)

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
