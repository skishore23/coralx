"""
Pure CodeLlama generation domain logic.
Contains pure functions for prompt creation and code extraction.
"""
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationRequest:
    """Immutable request for code generation with cheap knobs support."""
    problem_name: str
    buggy_code: str
    model_name: str
    max_tokens: int = 512
    temperature: float = 0.7
    adapter_path: Optional[str] = None  # LoRA adapter path
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0  # Cheap knob for repetition control
    do_sample: bool = True          # Cheap knob for sampling vs greedy


@dataclass(frozen=True)
class GenerationResult:
    """Immutable result of code generation."""
    generated_code: str
    function_name: str
    generation_time: float


def create_codellama_prompt(problem: Dict[str, Any]) -> str:
    """Pure function to create CodeLlama prompt with proper [INST] format and markdown request."""
    # Ensure types are correct
    problem_name_raw = problem.get('name', 'unknown')
    buggy_code_raw = problem.get('buggy_code', '')
    
    # Convert to strings to prevent errors
    problem_name = str(problem_name_raw) if problem_name_raw is not None else 'unknown'
    buggy_code = str(buggy_code_raw) if buggy_code_raw is not None else ''
    
    # Validate inputs
    if not problem_name or problem_name == 'None':
        raise ValueError(f"Invalid problem name: {problem_name_raw} (type: {type(problem_name_raw)})")
    
    if not buggy_code or buggy_code == 'None':
        raise ValueError(f"Invalid buggy code: {buggy_code_raw} (type: {type(buggy_code_raw)})")
    
    # Decode literal escape sequences to actual characters
    # Handle cases where buggy_code contains literal \n instead of actual newlines
    if '\\n' in buggy_code and buggy_code.count('\\n') > buggy_code.count('\n'):
        print(f"Unescaping literal newlines in buggy_code")
        # Only decode common escape sequences to avoid breaking actual code
        buggy_code = buggy_code.replace('\\n', '\n')
        buggy_code = buggy_code.replace('\\t', '\t')
        buggy_code = buggy_code.replace('\\r', '\r')
        print(f"Unescaped buggy_code, new length: {len(buggy_code)} chars")
    
    # FIXED: Complete prompt format - request full function, not completion
    return f"""[INST] Fix the bug in this Python function and return ONLY the corrected function:

```python
{buggy_code}
```

Return the fixed function as a complete Python function definition starting with 'def {problem_name}'. [/INST]"""


def extract_function_from_generation(generated_text: str, function_name: str) -> str:
    """Pure function to extract function from CodeLlama response - prioritizes explicit format."""
    import re
    
    print(f"Extracting function '{function_name}' from CodeLlama response")
    print(f"Generated text length: {len(generated_text)} chars")
    print(f"First 200 chars: {generated_text[:200]}...")
    
    # Look for [/INST] marker first
    if '[/INST]' in generated_text:
        response_part = generated_text.split('[/INST]', 1)[1].strip()
        print(f"Found [/INST] marker, extracting response part ({len(response_part)} chars)")
    else:
        response_part = generated_text
        print(f"No [/INST] marker found, using full text")
    
    # Show the actual response for debugging function extraction failures
    print(f"FULL RESPONSE DEBUG (first 1000 chars):")
    print("─" * 50)
    debug_text = response_part[:1000] + "..." if len(response_part) > 1000 else response_part
    for i, line in enumerate(debug_text.split('\n')[:30]):
        print(f"{i:3d}: {line}")
    print("─" * 50)
    
    # Essential extraction strategies only
    extraction_strategies = [
        ("Code blocks", _extract_from_code_blocks),
        ("Function search", _extract_function_flexible)
    ]
    
    for strategy_name, extraction_func in extraction_strategies:
        try:
            print(f"Trying strategy: {strategy_name}")
            extracted_code = extraction_func(response_part, function_name)
            
            if extracted_code and f'def {function_name}' in extracted_code:
                print(f"Success with {strategy_name}: {len(extracted_code)} chars")
                print(f"   First 100 chars: {extracted_code[:100]}...")
                return extracted_code.strip()
            else:
                print(f"   {strategy_name} didn't find the function")
                
        except Exception as e:
            print(f"   {strategy_name} failed: {e}")
            continue
    
    # Show debug info and fail fast (since prompt is very explicit)
    print(f"EXTRACTION FAILURE with explicit prompt")
    print(f"Full generated text for debugging:")
    print("─" * 50)
    for i, line in enumerate(response_part.split('\n')[:20]):
        print(f"{i:3d}: {line}")
    print("─" * 50)
    
    # Look for ANY function definitions to help debug
    all_functions = re.findall(r'def\s+(\w+)\s*\([^)]*\):', response_part)
    if all_functions:
        print(f"Found these function names: {all_functions}")
        print(f"Expected: '{function_name}' - CodeLlama used wrong name despite explicit prompt")
    else:
        print(f"No function definitions found - CodeLlama didn't follow format")
        # Try to see if there's ANY code-like content
        code_patterns = [
            r'```[\s\S]*?```',  # Any code blocks
            r'def\s+\w+',       # Any function starts
            r'class\s+\w+',     # Any class definitions
            r'import\s+\w+',    # Any imports
            r'return\s+\w+',    # Any returns
        ]
        found_patterns = []
        for pattern in code_patterns:
            matches = re.findall(pattern, response_part)
            if matches:
                found_patterns.append(f"{pattern}: {len(matches)} matches")
        
        if found_patterns:
            print(f"Found code patterns: {found_patterns}")
        else:
            print(f"No code patterns found - CodeLlama may have generated plain text")
    
    raise RuntimeError(
        f"Function '{function_name}' not found despite explicit prompt. "
        f"Found functions: {all_functions if all_functions else 'None'}. "
        f"CodeLlama did not follow the explicit format instructions. "
        f"Generated text preview: {response_part[:300]}..."
    )


def _extract_from_code_blocks(response_part: str, function_name: str) -> str:
    """Extract from markdown code blocks - now handles direct completion format."""
    import re
    
    # Handle direct function completion format
    # Prompt ends with: "def function_name("
    # CodeLlama completes with: "params):\n    body..."
    # We need to reconstruct: "def function_name(params):\n    body..."
    
    # More robust detection of completion format
    # Look for pattern: "...params):\n" at the start of response
    if '):\n' in response_part[:100]:  # Check first 100 chars for completion pattern
        # This looks like a completion starting with parameters
        print(f"Detected completion format - reconstructing full function")
        
        # Find where the function body starts (after the first "):") 
        colon_pos = response_part.find('):')
        if colon_pos != -1:
            # Split at the first "):" to separate params from body
            params_part = response_part[:colon_pos]      # "length_by_edge, startnode, goalnode"
            body_part = response_part[colon_pos+1:]      # ":\n    body..."
            
            # Reconstruct the complete function
            complete_function = f"def {function_name}({params_part}){body_part}"
            print(f"Reconstructed function: {len(complete_function)} chars")
            print(f"   First 100 chars: {complete_function[:100]}...")
            return complete_function.strip()
    
    # Look for complete function in response
    completion_pattern = rf'def\s+{re.escape(function_name)}\s*\([^)]*\):.*?(?=\n#|\n\n|\Z)'
    completion_match = re.search(completion_pattern, response_part, re.DOTALL)
    if completion_match:
        return completion_match.group(0).strip()
    
    # Check code block patterns
    code_block_patterns = [
        r'```python\s*\n(.*?)\n```',           # Standard markdown: ```python\ncode\n```
        r'```\s*\n(.*?)\n```',                 # Generic markdown: ```\ncode\n```
        r'```python(.*?)```',                  # Inline python: ```pythoncode```
        r'```(.*?)```',                        # Any code block: ```code```
    ]
    
    for i, pattern in enumerate(code_block_patterns):
        matches = re.findall(pattern, response_part, re.DOTALL)
        if matches:
            for match in matches:
                code_candidate = match.strip()
                if f'def {function_name}' in code_candidate:
                    return code_candidate
    
    return ""


def _extract_function_flexible(response_part: str, function_name: str) -> str:
    """Flexible function extraction that handles various formats."""
    import re
    
    # Look for any line that starts with def function_name
    lines = response_part.split('\n')
    func_start = -1
    
    for i, line in enumerate(lines):
        # More flexible matching - handle whitespace and variations
        if re.search(rf'^\s*def\s+{re.escape(function_name)}\s*\(', line):
            func_start = i
            break
    
    if func_start == -1:
        return ""
    
    # Extract the function - find where it ends
    func_lines = [lines[func_start]]
    base_indent = len(lines[func_start]) - len(lines[func_start].lstrip())
    
    for i in range(func_start + 1, len(lines)):
        line = lines[i]
        
        # Empty line or comment - include it
        if not line.strip() or line.strip().startswith('#'):
            func_lines.append(line)
            continue
        
        # Calculate indentation
        current_indent = len(line) - len(line.lstrip())
        
        # If same or less indentation than function def, we're done
        if current_indent <= base_indent and line.strip():
            break
        
        func_lines.append(line)
    
    return '\n'.join(func_lines)




def create_generation_request(problem: Dict[str, Any], genome_data: Dict[str, Any], config: Dict[str, Any], adapter_path_override: Optional[str] = None) -> GenerationRequest:
    """Pure function to create generation request from inputs."""
    # Use explicit adapter path if provided
    adapter_path = adapter_path_override
    
    return GenerationRequest(
        problem_name=problem.get('name', 'unknown'),
        buggy_code=problem.get('buggy_code', ''),
        model_name=config.get('model', {}).get('name', 'codellama/CodeLlama-7b-Python-hf'),
        max_tokens=config.get('generation', {}).get('max_tokens', 512),
        temperature=config.get('generation', {}).get('temperature', 0.7),
        adapter_path=adapter_path,
        top_p=config.get('generation', {}).get('top_p', 0.9),
        top_k=config.get('generation', {}).get('top_k', 50)
    ) 