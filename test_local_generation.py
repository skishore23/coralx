#!/usr/bin/env python3
"""
Local test for debugging the buggy_code variable issue.
Tests the generation pipeline locally before Modal deployment.
"""

def test_generation_pipeline():
    """Test the generation pipeline with detect_cycle problem."""
    
    # Sample problem data from the user's example
    problem = {
        'name': 'detect_cycle',
        'buggy_code': '''def detect_cycle(node):
    hare = tortoise = node

    while True:
        if hare.successor is None:
            return False

        tortoise = tortoise.successor
        hare = hare.successor.successor

        if hare is tortoise:
            return True''',
        'test_cases': '''def test1():
    """Case 1: Empty list
    Expected Output: False
    """
    assert not detect_cycle(None)

def test2():
    """Case 2: One node pointing to itself
    Expected Output: True  
    """
    node = Node(0)
    node.successor = node
    assert detect_cycle(node)'''
    }
    
    # Sample config for generation
    config = {
        'generation': {
            'max_tokens': 512,
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 50
        },
        'experiment': {
            'model': {
                'name': 'codellama/CodeLlama-7b-Python-hf'
            }
        }
    }
    
    print("🧪 Testing Generation Pipeline Locally")
    print("=" * 50)
    
    # Test 1: Basic problem structure validation
    print("📋 Test 1: Problem Structure")
    problem_name = problem.get('name')
    buggy_code = problem.get('buggy_code', '')
    
    print(f"   • Problem name: {repr(problem_name)}")
    print(f"   • Buggy code type: {type(buggy_code)}")
    print(f"   • Buggy code length: {len(buggy_code)} chars")
    print(f"   • Has buggy_code: {'buggy_code' in problem}")
    
    if not problem_name or not buggy_code:
        print("❌ Problem structure invalid")
        return False
    
    print("✅ Problem structure valid")
    
    # Test 2: Generation request creation
    print("\n📋 Test 2: GenerationRequest Creation")
    try:
        from coral.domain.codellama_generation import GenerationRequest
        
        # Extract generation config
        gen_config = config.get('generation', {})
        base_model_name = config.get('experiment', {}).get('model', {}).get('name', 'codellama/CodeLlama-7b-Python-hf')
        adapter_path = '/cache/adapters/adapter_test123'  # Dummy path
        
        # Create the request exactly as done in the Modal function
        request = GenerationRequest(
            problem_name=problem_name,
            buggy_code=buggy_code,  # This should work
            model_name=base_model_name,
            max_tokens=gen_config.get('max_tokens', 512),
            temperature=gen_config.get('temperature', 0.7),
            adapter_path=adapter_path,
            top_p=gen_config.get('top_p', 0.9),
            top_k=gen_config.get('top_k', 50)
        )
        
        print(f"✅ GenerationRequest created successfully")
        print(f"   • Request: {request}")
        
    except Exception as e:
        print(f"❌ GenerationRequest creation failed: {e}")
        return False
    
    # Test 3: Prompt creation
    print("\n📋 Test 3: Prompt Creation")
    try:
        from coral.domain.codellama_generation import create_codellama_prompt
        
        prompt = create_codellama_prompt(problem)
        print(f"✅ Prompt created successfully")
        print(f"   • Prompt length: {len(prompt)} chars")
        print(f"   • Prompt preview: {prompt[:200]}...")
        
    except Exception as e:
        print(f"❌ Prompt creation failed: {e}")
        return False
    
    print("\n🎉 All local tests passed!")
    return True


def simulate_modal_evaluation():
    """Simulate the Modal evaluation function logic locally."""
    
    print("\n🧪 Simulating Modal Evaluation Logic")
    print("=" * 50)
    
    # Sample genome data
    genome_data = {
        'id': 'test_genome_001',
        'lora_config': {
            'r': 16,
            'alpha': 16.0,
            'dropout': 0.05,
            'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        }
    }
    
    # Sample config
    config = {
        'generation': {
            'max_tokens': 512,
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 50
        },
        'experiment': {
            'model': {
                'name': 'codellama/CodeLlama-7b-Python-hf'
            },
            'dataset': {
                'path': '/cache/quixbugs_dataset'
            }
        }
    }
    
    # Sample problem (like what would be loaded from dataset)
    problem = {
        'name': 'detect_cycle',
        'buggy_code': '''def detect_cycle(node):
    hare = tortoise = node

    while True:
        if hare.successor is None:
            return False

        tortoise = tortoise.successor
        hare = hare.successor.successor

        if hare is tortoise:
            return True'''
    }
    
    print(f"📊 Genome: {genome_data.get('id')}")
    print(f"🎯 Problem: {problem.get('name')}")
    
    # Simulate the exact logic from the Modal function
    try:
        # Extract heavy genes (like in Modal function)
        from infra.adapter_cache import HeavyGenes
        
        lora_cfg = genome_data.get('lora_config', {})
        heavy_genes = HeavyGenes(
            rank=lora_cfg.get('r', 8),
            alpha=lora_cfg.get('alpha', 16.0),
            dropout=lora_cfg.get('dropout', 0.1),
            target_modules=tuple(lora_cfg.get('target_modules', ['q_proj', 'v_proj'])),
            adapter_type=lora_cfg.get('adapter_type', 'lora'),
            run_id=lora_cfg.get('run_id', 'test_run')
        )
        
        adapter_hash = heavy_genes.to_hash()
        adapter_path = f"/cache/adapters/adapter_{adapter_hash}"
        
        print(f"🔍 Heavy genes: {heavy_genes}")
        print(f"📁 Adapter path: {adapter_path}")
        
        # Simulate generation request creation (the failing part)
        from coral.domain.codellama_generation import GenerationRequest
        
        # Get config values
        gen_config = config.get('generation', {})
        base_model_name = config.get('experiment', {}).get('model', {}).get('name', 'codellama/CodeLlama-7b-Python-hf')
        
        # Extract problem data
        problem_name = problem.get('name')
        buggy_code = problem.get('buggy_code', '')
        
        print(f"🔧 Generation setup:")
        print(f"   • problem_name: {repr(problem_name)}")
        print(f"   • buggy_code available: {bool(buggy_code)}")
        print(f"   • model: {base_model_name}")
        
        # This is the exact code from the Modal function
        if not buggy_code:
            raise RuntimeError(f"FAIL-FAST: No buggy_code found in problem '{problem_name}'")
        
        request = GenerationRequest(
            problem_name=problem_name,
            buggy_code=buggy_code,  # This line should work
            model_name=base_model_name,
            max_tokens=gen_config.get('max_tokens', 512),
            temperature=gen_config.get('temperature', 0.7),
            adapter_path=adapter_path,
            top_p=gen_config.get('top_p', 0.9),
            top_k=gen_config.get('top_k', 50)
        )
        
        print(f"✅ GenerationRequest created in simulation!")
        print(f"   • Request: {request}")
        
        return True
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        print(f"   • Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 CORAL-X Local Generation Tests")
    print("=" * 60)
    
    # Run basic tests
    success1 = test_generation_pipeline()
    
    # Run Modal simulation
    success2 = simulate_modal_evaluation()
    
    if success1 and success2:
        print("\n🎉 All tests passed! The issue might be elsewhere.")
    else:
        print("\n❌ Tests failed - found the issue!") 