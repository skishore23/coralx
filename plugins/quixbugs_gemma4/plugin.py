"""Experimental QuixBugs plugin backed by local Gemma 4 E2B inference."""

from __future__ import annotations

import ast
import copy
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, ClassVar

from core.domain.cheap_knobs import map_ca_features_to_cheap_knobs
from core.domain.genome import Genome, MultiObjectiveScores
from core.domain.mapping import LoRAConfig
from core.domain.stable_hash import stable_int
from core.ports.interfaces import DatasetProvider, FitnessFn, ModelRunner
from plugins.quixbugs_mini.plugin import QUIXBUGS_MINI_PROBLEMS


class QuixBugsGemma4Dataset(DatasetProvider):
    """Small QuixBugs subset for real local model experiments."""

    def __init__(self, config: dict[str, Any]):
        dataset_cfg = config.get("experiment", {}).get("dataset", {})
        max_samples = int(dataset_cfg.get("max_samples") or len(QUIXBUGS_MINI_PROBLEMS))
        self._problems = QUIXBUGS_MINI_PROBLEMS[:max_samples]
        print(f"QuixBugs Gemma 4 dataset loaded: {len(self._problems)} problems")

    def problems(self) -> Iterable[dict[str, Any]]:
        yield from self._problems


class QuixBugsGemma4Runner(ModelRunner):
    """Gemma 4 E2B runner with model cache shared across genomes."""

    _processor: ClassVar[Any | None] = None
    _model: ClassVar[Any | None] = None
    _model_id: ClassVar[str | None] = None
    _model_revision: ClassVar[str | None] = None

    def __init__(
        self, lora_cfg: LoRAConfig, config: dict[str, Any], genome: Genome | None = None
    ):
        self.lora_cfg = lora_cfg
        self.config = config
        self.genome = genome
        self.model_id = (
            config.get("experiment", {})
            .get("model", {})
            .get("name", "google/gemma-4-E2B-it")
        )
        self.model_revision = str(
            config.get("experiment", {}).get("model", {}).get("revision") or "main"
        )
        print(
            "QuixBugs Gemma 4 runner initialized "
            f"for genome {genome.id if genome else 'unknown'}"
        )

    def generate(self, prompt: str, max_tokens: int = 256, cheap_knobs=None) -> str:
        """Generate a candidate Python fix with Gemma 4."""
        processor, model, torch = self._load_model()
        generation_kwargs = self._generation_kwargs(max_tokens)
        seed = stable_int(
            {
                "genome": self.genome.id if self.genome else "unknown",
                "prompt": prompt,
                "kwargs": generation_kwargs,
            },
            modulo=2**31 - 1,
        )
        torch.manual_seed(seed)

        instruction = (
            "Fix the buggy Python function below. Return only one complete Python "
            "function. Do not include Markdown fences or explanation.\n\n"
            f"Buggy code:\n{prompt}"
        )
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": instruction}],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        input_len = inputs["input_ids"].shape[-1]

        start = time.time()
        generation_config = copy.deepcopy(model.generation_config)
        generation_config.temperature = None
        generation_config.top_p = None
        generation_config.top_k = None
        for key, value in generation_kwargs.items():
            setattr(generation_config, key, value)
        with torch.inference_mode():
            output = model.generate(**inputs, generation_config=generation_config)
        elapsed = time.time() - start
        text = processor.decode(
            output[0][input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        print(
            "   Gemma 4 generation complete: "
            f"{len(text)} chars in {elapsed:.2f}s, kwargs={generation_kwargs}"
        )
        return text

    def _generation_kwargs(self, requested_max_tokens: int) -> dict[str, Any]:
        processor, _model, _torch = self._load_model()
        cheap_cfg = self.config.get("cheap_knobs")
        if cheap_knobs := getattr(self.genome, "ca_features", None):
            if cheap_cfg:
                knobs = map_ca_features_to_cheap_knobs(cheap_knobs, cheap_cfg)
                max_new_tokens = min(knobs.max_new_tokens, requested_max_tokens)
                kwargs: dict[str, Any] = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": False,
                    "repetition_penalty": knobs.repetition_penalty,
                }
            else:
                kwargs = {"max_new_tokens": min(96, requested_max_tokens)}
        else:
            kwargs = {"max_new_tokens": min(96, requested_max_tokens)}

        tokenizer = getattr(processor, "tokenizer", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            kwargs["pad_token_id"] = eos_token_id
        return kwargs

    def _load_model(self):
        if (
            self.__class__._processor is not None
            and self.__class__._model is not None
            and self.__class__._model_id == self.model_id
            and self.__class__._model_revision == self.model_revision
        ):
            return self.__class__._processor, self.__class__._model, self._torch()

        try:
            torch = self._torch()
            from transformers import AutoModelForCausalLM, AutoProcessor
            from transformers.utils import logging as transformers_logging
        except Exception as exc:  # pragma: no cover - depends on optional deps
            raise RuntimeError(
                "Gemma 4 experiments require optional ML dependencies. "
                "Use a native arm64 Python on Apple Silicon and install the ml extra, "
                "or use the prepared .venv for local runs."
            ) from exc

        transformers_logging.set_verbosity_error()

        if (
            not getattr(torch.backends, "mps", None)
            or not torch.backends.mps.is_available()
        ):
            print("   MPS is not available; Gemma 4 will run on the available device")

        print(f"   Loading Gemma 4 model: {self.model_id}")
        start = time.time()
        processor = AutoProcessor.from_pretrained(
            self.model_id,
            revision=self.model_revision,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            revision=self.model_revision,
            dtype="auto",
            device_map="auto",
            attn_implementation="sdpa",
        )
        print(
            f"   Gemma 4 model loaded in {time.time() - start:.2f}s on {model.device}"
        )

        self.__class__._processor = processor
        self.__class__._model = model
        self.__class__._model_id = self.model_id
        self.__class__._model_revision = self.model_revision
        return processor, model, torch

    @staticmethod
    def _torch():
        import torch

        major, minor, *_ = torch.__version__.split(".")
        if (int(major), int(minor)) < (2, 4):
            raise RuntimeError(
                f"Gemma 4 requires torch>=2.4 with current Transformers; got {torch.__version__}"
            )
        return torch


class QuixBugsGemma4Fitness(FitnessFn):
    """Execution-based fitness for Gemma-generated QuixBugs fixes."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        print("QuixBugs Gemma 4 fitness initialized")

    def __call__(
        self,
        genome: Genome,
        model: ModelRunner,
        problems: Iterable[dict[str, Any]],
        ca_features=None,
    ) -> float:
        return self.evaluate_multi_objective(
            genome, model, problems, ca_features
        ).overall_fitness()

    def evaluate_multi_objective(
        self,
        genome: Genome,
        model: ModelRunner,
        problems: Iterable[dict[str, Any]],
        ca_features=None,
    ) -> MultiObjectiveScores:
        print("\nQUIXBUGS GEMMA 4 EVALUATION")
        print("=" * 40)
        print(f"Genome ID: {genome.id}")

        bugfix_scores: list[float] = []
        style_scores: list[float] = []
        security_scores: list[float] = []
        runtime_scores: list[float] = []
        syntax_scores: list[float] = []

        for problem in problems:
            name = problem["name"]
            print(f"\nProblem: {name}")
            raw = model.generate(problem["prompt"], max_tokens=256)
            code = _extract_python_function(raw)
            syntax_ok = _syntax_ok(code)
            security_ok = _security_ok(code) if syntax_ok else False
            test_result = _run_problem_tests(name, code) if security_ok else None

            bugfix = (
                test_result["passed"] / test_result["total"] if test_result else 0.0
            )
            runtime = _runtime_score(test_result["elapsed"]) if test_result else 0.0
            style = _style_score(code)
            syntax = 1.0 if syntax_ok else 0.0
            security = 1.0 if security_ok else 0.0

            bugfix_scores.append(bugfix)
            style_scores.append(style)
            security_scores.append(security)
            runtime_scores.append(runtime)
            syntax_scores.append(syntax)

            print("   Extracted code:")
            for line in code.splitlines()[:8]:
                print(f"      {line}")
            print(
                "   Scores: "
                f"bugfix={bugfix:.3f} style={style:.3f} "
                f"security={security:.3f} runtime={runtime:.3f} syntax={syntax:.3f}"
            )

        return MultiObjectiveScores(
            bugfix=_avg(bugfix_scores),
            style=_avg(style_scores),
            security=_avg(security_scores),
            runtime=_avg(runtime_scores),
            syntax=_avg(syntax_scores),
        )


class QuixBugsGemma4Plugin:
    """Experimental plugin for tiny real-model Gemma 4 runs."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        model_name = (
            config.get("experiment", {})
            .get("model", {})
            .get("name", "google/gemma-4-E2B-it")
        )
        print("QuixBugs Gemma 4 plugin initialized")
        print(f"   Model: {model_name}")

    def dataset(self) -> DatasetProvider:
        return QuixBugsGemma4Dataset(self.config)

    def model_factory(self) -> Callable[[LoRAConfig, Genome | None], ModelRunner]:
        def create_model(
            lora_cfg: LoRAConfig, genome: Genome | None = None
        ) -> ModelRunner:
            return QuixBugsGemma4Runner(lora_cfg, self.config, genome=genome)

        return create_model

    def fitness_fn(self) -> FitnessFn:
        return QuixBugsGemma4Fitness(self.config)


def _extract_python_function(text: str) -> str:
    """Extract the first Python function from model text."""
    stripped = text.strip()
    if "```" in stripped:
        parts = stripped.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.startswith("python"):
                candidate = candidate.removeprefix("python").strip()
            if "def " in candidate:
                stripped = candidate
                break

    def_index = stripped.find("def ")
    if def_index >= 0:
        stripped = stripped[def_index:]

    lines = stripped.splitlines()
    function_lines: list[str] = []
    in_function = False
    for line in lines:
        if not in_function and line.lstrip().startswith("def "):
            in_function = True
            function_lines.append(line)
            continue
        if in_function:
            if line.strip() and not line.startswith((" ", "\t")):
                break
            function_lines.append(line)
    return "\n".join(function_lines).strip()


def _syntax_ok(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _security_ok(code: str) -> bool:
    tree = ast.parse(code)
    banned_calls = {"eval", "exec", "open", "__import__", "input", "compile"}
    banned_attrs = {"system", "popen", "remove", "unlink", "rmdir", "rmtree"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom | ast.Global | ast.Nonlocal):
            return False
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in banned_calls:
                return False
            if isinstance(func, ast.Attribute) and func.attr in banned_attrs:
                return False
    return True


def _run_problem_tests(name: str, code: str) -> dict[str, float] | None:
    tests = _problem_tests(name)
    if not tests:
        return None
    script = code + "\n\n" + "\n".join(tests)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
        handle.write(script)
        path = Path(handle.name)
    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        elapsed = time.time() - start
        passed = sum(
            1 for line in result.stdout.splitlines() if line.startswith("PASS:")
        )
        return {"passed": float(passed), "total": float(len(tests)), "elapsed": elapsed}
    except subprocess.TimeoutExpired:
        return {"passed": 0.0, "total": float(len(tests)), "elapsed": 5.0}
    finally:
        path.unlink(missing_ok=True)


def _problem_tests(name: str) -> list[str]:
    if name == "gcd":
        return [
            "_check = gcd(48, 18) == 6; print('PASS:gcd1' if _check else 'FAIL:gcd1')",
            "_check = gcd(17, 13) == 1; print('PASS:gcd2' if _check else 'FAIL:gcd2')",
            "_check = gcd(5, 0) == 5; print('PASS:gcd3' if _check else 'FAIL:gcd3')",
        ]
    if name == "is_valid_parenthesization":
        return [
            "_check = is_valid_parenthesization('(()())') is True; print('PASS:p1' if _check else 'FAIL:p1')",
            "_check = is_valid_parenthesization('(()') is False; print('PASS:p2' if _check else 'FAIL:p2')",
            "_check = is_valid_parenthesization(')(') is False; print('PASS:p3' if _check else 'FAIL:p3')",
        ]
    if name == "sqrt":
        return [
            "_check = abs(sqrt(4) - 2) < 1e-6; print('PASS:s1' if _check else 'FAIL:s1')",
            "_check = abs(sqrt(2) - 1.41421356237) < 1e-3; print('PASS:s2' if _check else 'FAIL:s2')",
            "_check = sqrt(-1) == -1; print('PASS:s3' if _check else 'FAIL:s3')",
        ]
    return []


def _runtime_score(elapsed: float) -> float:
    if elapsed <= 0.1:
        return 1.0
    if elapsed >= 2.0:
        return 0.2
    return max(0.2, 1.0 - ((elapsed - 0.1) / 1.9) * 0.8)


def _style_score(code: str) -> float:
    if not code:
        return 0.0
    score = 1.0
    if "```" in code:
        score -= 0.4
    if not code.lstrip().startswith("def "):
        score -= 0.3
    if any(len(line) > 100 for line in code.splitlines()):
        score -= 0.2
    return max(0.0, score)


def _avg(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)
