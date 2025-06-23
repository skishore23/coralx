"""
Dataset Constants - Centralized Train/Test Split
===============================================

This file defines the authoritative train/test split for QuixBugs dataset
to prevent data leakage between LoRA/DoRA training and evaluation.

ALL components (training, evaluation, plugins) MUST use these constants.
"""

# 🚨 AUTHORITATIVE TRAINING PROBLEMS LIST
# These problems are used for LoRA/DoRA adapter training and MUST BE EXCLUDED from evaluation
QUIXBUGS_TRAINING_PROBLEMS = frozenset({
    'gcd', 'get_factors', 'is_valid_parenthesization', 'levenshtein',
    'longest_common_subsequence', 'max_sublist_sum', 'pascal', 'reverse_linked_list',
    'hanoi', 'mergesort', 'bitcount', 'bucketsort', 'find_first_in_sorted', 
    'find_in_sorted', 'flatten', 'knapsack', 'kth', 'lis', 'powerset',
    'quicksort', 'rpn_eval', 'shunting_yard', 'sqrt', 'subsequences'
})

# ✅ CLEAN EVALUATION PROBLEMS 
# These problems have JSON test data but are NOT used in training
QUIXBUGS_CLEAN_TEST_PROBLEMS = frozenset({
    'kheapsort', 'lcs_length', 'next_palindrome', 'next_permutation',
    'possible_change', 'sieve', 'to_base', 'wrap'
})

# 📊 DATASET STATISTICS
TOTAL_TRAINING_PROBLEMS = len(QUIXBUGS_TRAINING_PROBLEMS)  # 24
TOTAL_CLEAN_TEST_PROBLEMS = len(QUIXBUGS_CLEAN_TEST_PROBLEMS)  # 8
TOTAL_PROBLEMS_WITH_JSON = TOTAL_TRAINING_PROBLEMS + TOTAL_CLEAN_TEST_PROBLEMS  # 32

# 🔍 VALIDATION FUNCTIONS
def validate_no_overlap():
    """Ensure training and test sets don't overlap."""
    overlap = QUIXBUGS_TRAINING_PROBLEMS & QUIXBUGS_CLEAN_TEST_PROBLEMS
    if overlap:
        raise ValueError(f"CRITICAL: Training/test overlap detected: {overlap}")
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
    'is_training_problem',
    'is_clean_test_problem',
    'validate_no_overlap'
] 