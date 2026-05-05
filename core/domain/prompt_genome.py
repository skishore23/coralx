"""Prompt genome structures for behavior-surface evolution."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from core.domain.stable_hash import stable_digest


@dataclass(frozen=True)
class PromptGenome:
    """Prompt, few-shot, and decoding policy candidate."""

    id: str
    system_prompt: str
    reasoning_instruction: str
    answer_format_instruction: str
    verification_instruction: str
    few_shot_example_ids: tuple[str, ...]
    few_shot_order: tuple[int, ...]
    temperature: float
    top_p: float
    max_new_tokens: int
    self_consistency_n: int
    origin: str = "initial"
    generation: int = 0
    seed: int = 0
    fitness: float | None = None
    metrics: dict[str, Any] | None = None

    def decode_params(self) -> dict[str, Any]:
        """Return generation settings as a stable dictionary."""
        return {
            "temperature": round(float(self.temperature), 6),
            "top_p": round(float(self.top_p), 6),
            "max_new_tokens": int(self.max_new_tokens),
            "self_consistency_n": int(self.self_consistency_n),
        }

    def prompt_payload(self) -> dict[str, Any]:
        """Return the behavior-surface payload that identifies this genome."""
        return {
            "system_prompt": self.system_prompt,
            "reasoning_instruction": self.reasoning_instruction,
            "answer_format_instruction": self.answer_format_instruction,
            "verification_instruction": self.verification_instruction,
            "few_shot_example_ids": list(self.few_shot_example_ids),
            "few_shot_order": list(self.few_shot_order),
            "decode": self.decode_params(),
        }

    def cache_key(
        self,
        *,
        model: str,
        split_fingerprint: str,
        seed: int,
        version: str = "gsm8k_prompt_v5",
    ) -> str:
        """Return a stable candidate-evaluation cache key."""
        return stable_digest(
            {
                "version": version,
                "model": model,
                "prompt": self.prompt_payload(),
                "split_fingerprint": split_fingerprint,
                "seed": seed,
            },
            length=20,
        )

    def structural_key(self) -> str:
        """Return a stable key for deduplicating equivalent prompt genomes."""
        return stable_digest(self.prompt_payload(), length=20)

    def with_generation(
        self, generation: int, origin: str, genome_id: str
    ) -> PromptGenome:
        """Return a copy with lineage fields updated."""
        return replace(self, generation=generation, origin=origin, id=genome_id)

    def with_result(self, fitness: float, metrics: dict[str, Any]) -> PromptGenome:
        """Return a copy with evaluation result attached."""
        return replace(self, fitness=fitness, metrics=metrics)


def prompt_genome_from_dict(data: dict[str, Any]) -> PromptGenome:
    """Build a prompt genome from a JSON-compatible dictionary."""
    return PromptGenome(
        id=str(data["id"]),
        system_prompt=str(data["system_prompt"]),
        reasoning_instruction=str(data["reasoning_instruction"]),
        answer_format_instruction=str(data["answer_format_instruction"]),
        verification_instruction=str(data["verification_instruction"]),
        few_shot_example_ids=tuple(str(item) for item in data["few_shot_example_ids"]),
        few_shot_order=tuple(int(item) for item in data["few_shot_order"]),
        temperature=float(data["temperature"]),
        top_p=float(data["top_p"]),
        max_new_tokens=int(data["max_new_tokens"]),
        self_consistency_n=int(data["self_consistency_n"]),
        origin=str(data.get("origin", "initial")),
        generation=int(data.get("generation", 0)),
        seed=int(data.get("seed", 0)),
        fitness=data.get("fitness"),
        metrics=data.get("metrics"),
    )
