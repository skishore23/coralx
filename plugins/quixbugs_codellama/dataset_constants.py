"""
Dataset Constants - Centralized Train/Test Split
===============================================

This file defines the authoritative train/test split for QuixBugs dataset
to prevent data leakage between LoRA/DoRA training and evaluation.

ALL components (training, evaluation, plugins) MUST use these constants.
"""

# Authoritative training problems list
# These problems are used for LoRA/DoRA adapter training and must be excluded from evaluation
QUIXBUGS_TRAINING_PROBLEMS = frozenset({
    'gcd', 'get_factors', 'is_valid_parenthesization', 'levenshtein',
    'longest_common_subsequence', 'max_sublist_sum', 'pascal', 'reverse_linked_list',
    'hanoi', 'mergesort', 'bitcount', 'bucketsort', 'find_first_in_sorted', 
    'find_in_sorted', 'flatten', 'knapsack', 'kth', 'lis', 'powerset',
    'quicksort', 'rpn_eval', 'shunting_yard', 'sqrt', 'subsequences'
})

# Clean evaluation problems
# These problems have JSON test data but are not used in training
QUIXBUGS_CLEAN_TEST_PROBLEMS = frozenset({
    'kheapsort', 'lcs_length', 'next_palindrome', 'next_permutation',
    'possible_change', 'sieve', 'to_base', 'wrap'
})

# Problem difficulty classifications
# Easy problems: Simple algorithms, basic data structures
EASY_PROBLEMS = frozenset({
    'gcd', 'bitcount', 'sqrt', 'is_valid_parenthesization',
    'reverse_linked_list', 'flatten', 'to_base', 'get_factors'
})

# Hard problems: Complex algorithms, advanced data structures
HARD_PROBLEMS = frozenset({
    'hanoi', 'mergesort', 'quicksort', 'kheapsort',
    'levenshtein', 'longest_common_subsequence', 'knapsack',
    'next_permutation', 'powerset', 'subsequences'
})

# Medium problems: Everything else that has JSON test data
MEDIUM_PROBLEMS = frozenset({
    'kth', 'lis', 'lcs_length', 'max_sublist_sum', 'pascal',
    'bucketsort', 'find_in_sorted', 'find_first_in_sorted',
    'next_palindrome', 'possible_change', 'rpn_eval', 'shunting_yard',
    'sieve', 'wrap'
})

# Dataset statistics
TOTAL_TRAINING_PROBLEMS = len(QUIXBUGS_TRAINING_PROBLEMS)  # 24
TOTAL_CLEAN_TEST_PROBLEMS = len(QUIXBUGS_CLEAN_TEST_PROBLEMS)  # 8
TOTAL_PROBLEMS_WITH_JSON = TOTAL_TRAINING_PROBLEMS + TOTAL_CLEAN_TEST_PROBLEMS  # 32

# Validation functions
def validate_no_overlap():
    """Ensure training and test sets don't overlap."""
    overlap = QUIXBUGS_TRAINING_PROBLEMS & QUIXBUGS_CLEAN_TEST_PROBLEMS
    if overlap:
        raise ValueError(f"Training/test overlap detected: {overlap}")
    return True

def is_training_problem(problem_name: str) -> bool:
    """Check if a problem is used for training (and should be excluded from evaluation)."""
    return problem_name in QUIXBUGS_TRAINING_PROBLEMS

def is_clean_test_problem(problem_name: str) -> bool:
    """Check if a problem is clean for testing (never seen during training)."""
    return problem_name in QUIXBUGS_CLEAN_TEST_PROBLEMS

# Run validation on import
validate_no_overlap()

# Export for type hints
__all__ = [
    'QUIXBUGS_TRAINING_PROBLEMS',
    'QUIXBUGS_CLEAN_TEST_PROBLEMS', 
    'EASY_PROBLEMS',
    'MEDIUM_PROBLEMS', 
    'HARD_PROBLEMS',
    'is_training_problem',
    'is_clean_test_problem',
    'validate_no_overlap'
] 