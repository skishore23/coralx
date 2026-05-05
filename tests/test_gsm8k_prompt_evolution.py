"""Tests for GSM8K prompt-evolution mechanics."""

from pathlib import Path

import pytest

from core.domain.prompt_genome import PromptGenome
from plugins.gsm8k_lora.plugin import _extract_final_answer
from plugins.gsm8k_prompt_evolution.plugin import (
    PROMPT_EVAL_VERSION,
    GSM8KPromptRunner,
    PromptEvaluationMetrics,
    PromptProofLogger,
    _extract_generated_answer,
    _settings,
    append_jsonl,
    assert_no_test_leakage,
    create_initial_prompt_population,
    extract_prompt_final_answer,
    load_completed_candidate_records,
    split_fingerprint,
)


def _config(tmp_path: Path) -> dict:
    return {
        "seed": 7,
        "execution": {"output_dir": str(tmp_path)},
        "experiment": {
            "model": {
                "name": "Qwen/Qwen2.5-Math-1.5B-Instruct",
                "max_seq_length": 512,
            },
            "evaluation": {
                "reflection_samples": 4,
                "dev_samples": 4,
                "test_samples": 4,
                "max_new_tokens": 64,
                "self_consistency_values": [1],
                "few_shot_count": 2,
                "resume": True,
            },
        },
        "cheap_knobs": {
            "temperature_range": [0.0, 0.2],
            "top_p_range": [0.9, 1.0],
            "max_tokens_range": [32, 96],
        },
    }


def _rows(prefix: str, count: int) -> list[dict]:
    return [
        {
            "id": f"{prefix}_{index}",
            "question": f"{prefix} question {index}?",
            "answer": f"work #### {index}",
            "final_answer": str(index),
        }
        for index in range(count)
    ]


def test_prompt_genome_creation_is_deterministic(tmp_path):
    settings = _settings(_config(tmp_path))
    reflection_rows = _rows("reflection", 4)

    first = create_initial_prompt_population(
        population_size=4,
        seed=123,
        reflection_rows=reflection_rows,
        settings=settings,
    )
    second = create_initial_prompt_population(
        population_size=4,
        seed=123,
        reflection_rows=reflection_rows,
        settings=settings,
    )

    assert [genome.prompt_payload() for genome in first] == [
        genome.prompt_payload() for genome in second
    ]
    assert [genome.id for genome in first] == [genome.id for genome in second]


def test_prompt_cache_key_is_stable_and_split_sensitive():
    genome = PromptGenome(
        id="candidate",
        system_prompt="system",
        reasoning_instruction="reason",
        answer_format_instruction="answer",
        verification_instruction="verify",
        few_shot_example_ids=("reflection_1",),
        few_shot_order=(0,),
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=64,
        self_consistency_n=1,
    )

    first = genome.cache_key(model="model", split_fingerprint="dev-a", seed=1)
    second = genome.cache_key(model="model", split_fingerprint="dev-a", seed=1)
    changed = genome.cache_key(model="model", split_fingerprint="dev-b", seed=1)

    assert first == second
    assert first != changed


def test_prompt_runner_cache_key_includes_render_settings(tmp_path):
    genome = PromptGenome(
        id="candidate",
        system_prompt="system",
        reasoning_instruction="reason",
        answer_format_instruction="answer",
        verification_instruction="verify",
        few_shot_example_ids=("reflection_1",),
        few_shot_order=(0,),
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=64,
        self_consistency_n=1,
    )
    rows = _rows("dev", 2)
    reflection_rows = _rows("reflection", 2)
    base_config = _config(tmp_path)
    changed_config = _config(tmp_path)
    changed_config["experiment"]["evaluation"]["use_chat_template"] = False

    base_key = GSM8KPromptRunner(base_config).cache_key(
        genome, rows, reflection_rows, "hybrid:dev"
    )
    changed_key = GSM8KPromptRunner(changed_config).cache_key(
        genome, rows, reflection_rows, "hybrid:dev"
    )

    assert base_key != changed_key


def test_prompt_cache_key_changes_with_huggingface_revisions(tmp_path):
    """Prompt-eval cache keys should isolate model and dataset revisions."""
    genome = PromptGenome(
        id="candidate",
        system_prompt="system",
        reasoning_instruction="reason",
        answer_format_instruction="answer",
        verification_instruction="verify",
        few_shot_example_ids=("reflection_1",),
        few_shot_order=(0,),
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=64,
        self_consistency_n=1,
    )
    rows = _rows("dev", 2)
    reflection_rows = _rows("reflection", 2)
    base_config = _config(tmp_path)
    changed_model = _config(tmp_path)
    changed_model["experiment"]["model"]["revision"] = "model-rev-a"
    changed_dataset = _config(tmp_path)
    changed_dataset["experiment"]["dataset"] = {"revision": "dataset-rev-a"}

    base_key = GSM8KPromptRunner(base_config).cache_key(
        genome, rows, reflection_rows, "hybrid:dev"
    )

    assert base_key != GSM8KPromptRunner(changed_model).cache_key(
        genome, rows, reflection_rows, "hybrid:dev"
    )
    assert base_key != GSM8KPromptRunner(changed_dataset).cache_key(
        genome, rows, reflection_rows, "hybrid:dev"
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("We compute 2 + 2. \\boxed{4}", "4"),
        ("Reasoning here #### 1,234", "1234"),
        ("Final: 7.5", "7.5"),
    ],
)
def test_exact_answer_parsing(text, expected):
    assert _extract_final_answer(text) == expected


def test_prompt_answer_parser_requires_final_marker():
    assert extract_prompt_final_answer("Equation list: 1. x + y = 2") is None
    assert extract_prompt_final_answer("Final: \\boxed{56}") == "56"
    assert extract_prompt_final_answer("Answer: 1,234") == "1234"


def test_prefilled_boxed_answer_parser():
    assert _extract_generated_answer("Question\nAnswer: \\boxed{", "56}") == "56"


def test_no_test_set_leakage_guard_raises():
    reflection = _rows("reflection", 1)
    dev = _rows("dev", 1)
    test = _rows("test", 1)
    test[0]["question"] = dev[0]["question"]

    with pytest.raises(ValueError, match="Test-set leakage"):
        assert_no_test_leakage(reflection, dev, test)


def test_split_fingerprint_is_stable():
    rows = _rows("dev", 3)
    assert split_fingerprint(rows) == split_fingerprint(list(rows))


def test_resume_loads_partial_candidate_jsonl(tmp_path):
    path = tmp_path / "candidate_evaluations.jsonl"
    append_jsonl(
        path,
        {
            "suite": "hybrid",
            "split": "dev",
            "candidate_id": "candidate_a",
            "evaluation_version": "gsm8k_prompt_v5",
            "genome": {"decode": {"max_new_tokens": 64}},
            "metrics": {"exact_accuracy": 0.5},
        },
    )

    completed = load_completed_candidate_records(path)
    key = ("hybrid", "dev", "candidate_a")
    matching_keys = [item for item in completed if item[:3] == key]

    assert len(matching_keys) == 1
    assert completed[matching_keys[0]]["metrics"]["exact_accuracy"] == 0.5


def test_resume_records_are_cache_key_sensitive(tmp_path):
    path = tmp_path / "candidate_evaluations.jsonl"
    genome = PromptGenome(
        id="candidate_a",
        system_prompt="system",
        reasoning_instruction="reason",
        answer_format_instruction="answer",
        verification_instruction="verify",
        few_shot_example_ids=(),
        few_shot_order=(),
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=64,
        self_consistency_n=1,
    )
    base_record = {
        "suite": "hybrid",
        "split": "dev",
        "candidate_id": "candidate_a",
        "evaluation_version": PROMPT_EVAL_VERSION,
        "genome": genome.prompt_payload(),
        "metrics": {"cache_key": "cache-a", "exact_accuracy": 0.5},
    }
    append_jsonl(path, base_record)
    append_jsonl(
        path,
        {
            **base_record,
            "metrics": {"cache_key": "cache-b", "exact_accuracy": 0.6},
        },
    )

    completed = load_completed_candidate_records(path)

    assert len(completed) == 2
    assert {key[-1] for key in completed} == {"cache-a", "cache-b"}


def test_proof_logger_does_not_resume_stale_cache_key(tmp_path):
    genome = PromptGenome(
        id="candidate_a",
        system_prompt="system",
        reasoning_instruction="reason",
        answer_format_instruction="answer",
        verification_instruction="verify",
        few_shot_example_ids=(),
        few_shot_order=(),
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=64,
        self_consistency_n=1,
    )
    rows = _rows("dev", 1)
    reflection_rows = _rows("reflection", 1)
    append_jsonl(
        tmp_path / "candidate_evaluations.jsonl",
        {
            "suite": "hybrid",
            "split": "dev",
            "candidate_id": "candidate_a",
            "evaluation_version": PROMPT_EVAL_VERSION,
            "genome": genome.prompt_payload(),
            "metrics": {"cache_key": "cache-a", "exact_accuracy": 0.5},
        },
    )

    class FakeRunner:
        def __init__(self, cache_key: str):
            self.cache_key_value = cache_key
            self.evaluate_called = False

        def cache_key(self, *args, **kwargs):
            return self.cache_key_value

        def evaluate(self, *args, **kwargs):
            self.evaluate_called = True
            return PromptEvaluationMetrics(
                exact_accuracy=0.25,
                token_efficiency=1.0,
                latency_score=1.0,
                formatting_success_rate=1.0,
                novelty=0.0,
                avg_generated_tokens=1.0,
                latency_seconds=0.01,
                predictions=(),
                cache_key=self.cache_key_value,
            )

    logger = PromptProofLogger(tmp_path, resume=True)
    matching_runner = FakeRunner("cache-a")
    resumed = logger.evaluate_and_log(
        runner=matching_runner,
        genome=genome,
        rows=rows,
        reflection_rows=reflection_rows,
        suite="hybrid",
        split="dev",
        novelty=0.0,
    )
    stale_runner = FakeRunner("cache-b")
    recomputed = logger.evaluate_and_log(
        runner=stale_runner,
        genome=genome,
        rows=rows,
        reflection_rows=reflection_rows,
        suite="hybrid",
        split="dev",
        novelty=0.0,
    )

    assert resumed["metrics"]["cache_key"] == "cache-a"
    assert matching_runner.evaluate_called is False
    assert recomputed["metrics"]["cache_key"] == "cache-b"
    assert stale_runner.evaluate_called is True
