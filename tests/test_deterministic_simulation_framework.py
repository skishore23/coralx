"""Deterministic simulation tests for CA, NEAT-style ops, and Pareto selection."""

from __future__ import annotations

from random import Random

import numpy as np
import pytest

from core.common.config import SelectionMode
from core.domain.ca import CASeed, evolve, next_step
from core.domain.feature_extraction import extract_features
from core.domain.genome import Genome, MultiObjectiveScores
from core.domain.mapping import LoRAConfig, map_features_to_lora_config
from core.domain.neat import Population, crossover, mutate
from core.domain.stable_hash import stable_int
from core.services.pareto.selection import (
    dominates,
    fast_non_dominated_sort,
    nsga2_select,
)
from tests.support.deterministic_simulation import (
    assert_core_simulation_invariants,
    build_simulation_config,
    genome_trace,
    run_deterministic_simulation,
)


def test_ca_rule_index_excludes_center_cell_and_keeps_input_pure():
    """CA rule semantics should be deterministic and side-effect free."""

    grid = np.zeros((5, 5), dtype=int)
    grid[2, 2] = 1
    original = grid.copy()

    live_cell_with_no_live_neighbors_survives = 1 << 8
    next_grid = next_step(grid, live_cell_with_no_live_neighbors_survives)

    expected = np.zeros((5, 5), dtype=int)
    expected[2, 2] = 1
    assert np.array_equal(next_grid, expected)
    assert np.array_equal(grid, original)

    dead_cell_with_one_wrapped_neighbor_activates = 1 << 1
    wrap_grid = np.zeros((5, 5), dtype=int)
    wrap_grid[0, 0] = 1
    wrapped = next_step(wrap_grid, dead_cell_with_one_wrapped_neighbor_activates)
    assert wrapped[4, 4] == 1
    assert wrapped[0, 0] == 0

    seed = CASeed(grid=grid, rule=live_cell_with_no_live_neighbors_survives, steps=3)
    history_a = evolve(seed)
    history_b = evolve(seed)

    assert len(history_a.history) == 4
    assert [state.tolist() for state in history_a.history] == [
        state.tolist() for state in history_b.history
    ]


@pytest.mark.parametrize(
    "selection_mode", [SelectionMode.TOURNAMENT, SelectionMode.PARETO]
)
def test_core_simulation_replays_for_same_seed_without_monotone_assumption(
    tmp_path, selection_mode
):
    """A no-model simulation should replay exactly for CA, scoring, selection, and NEAT."""

    config_a = build_simulation_config(
        tmp_path / "run-a", seed=123, selection_mode=selection_mode
    )
    config_b = build_simulation_config(
        tmp_path / "run-b", seed=123, selection_mode=selection_mode
    )

    trace_a = run_deterministic_simulation(config_a)
    trace_b = run_deterministic_simulation(config_b)

    assert_core_simulation_invariants(trace_a, config_a)
    assert trace_a == trace_b


def test_core_simulation_changes_when_seed_changes(tmp_path):
    """The harness should catch seed drift instead of snapshotting one static run."""

    trace_a = run_deterministic_simulation(
        build_simulation_config(tmp_path / "seed-a", seed=123)
    )
    trace_b = run_deterministic_simulation(
        build_simulation_config(tmp_path / "seed-b", seed=124)
    )

    assert trace_a.generations[0].evaluated_population != (
        trace_b.generations[0].evaluated_population
    )


def test_neat_mutation_and_crossover_are_replayable_and_remap_ca_features(tmp_path):
    """NEAT-style operations should be deterministic and preserve genome contracts."""

    config = build_simulation_config(tmp_path, seed=77)
    raw_config = config.model_dump()
    parent = _parent_genome("parent", active_at=(2, 2))

    ca_mutant_a = mutate(
        parent,
        config.evo,
        Random(1),
        generation=0,
        config_dict=raw_config,
        run_id=config.cache.run_id,
    )
    ca_mutant_b = mutate(
        parent,
        config.evo,
        Random(1),
        generation=0,
        config_dict=raw_config,
        run_id=config.cache.run_id,
    )

    assert genome_trace(ca_mutant_a) == genome_trace(ca_mutant_b)
    assert ca_mutant_a.run_id == "deterministic-sim"
    assert ca_mutant_a.fitness is None
    assert not np.array_equal(ca_mutant_a.seed.grid, parent.seed.grid)

    expected_features = extract_features(evolve(ca_mutant_a.seed))
    expected_lora = map_features_to_lora_config(
        expected_features,
        raw_config,
        diversity_strength=1.0,
        genome_index=stable_int(parent.id, 1000),
    )
    assert ca_mutant_a.ca_features == expected_features
    assert ca_mutant_a.lora_cfg == expected_lora

    lora_mutant = mutate(
        parent,
        config.evo,
        _ForcedLoRAMutationRng(),
        generation=1,
        config_dict=raw_config,
        run_id=config.cache.run_id,
    )
    assert np.array_equal(lora_mutant.seed.grid, parent.seed.grid)
    assert lora_mutant.ca_features == parent.ca_features
    assert lora_mutant.lora_cfg == LoRAConfig(
        r=16,
        alpha=32,
        dropout=0.2,
        target_modules=("q_proj", "v_proj"),
    )

    right_parent = _parent_genome("right", active_at=(1, 3), rule=102, steps=4)
    child_a = crossover(
        parent,
        right_parent,
        config.evo,
        Random(5),
        generation=1,
        config_dict=raw_config,
        run_id=config.cache.run_id,
    )
    child_b = crossover(
        parent,
        right_parent,
        config.evo,
        Random(5),
        generation=1,
        config_dict=raw_config,
        run_id=config.cache.run_id,
    )

    assert genome_trace(child_a) == genome_trace(child_b)
    assert child_a.run_id == "deterministic-sim"
    assert child_a.seed.rule in {parent.seed.rule, right_parent.seed.rule}
    assert child_a.seed.steps in {parent.seed.steps, right_parent.seed.steps}
    assert any(
        np.array_equal(child_a.seed.grid, candidate.seed.grid)
        for candidate in (parent, right_parent)
    )


def test_pareto_selection_prefers_nondominated_front_over_scalar_fitness():
    """NSGA-II selection should use objective dominance, not scalar shortcuts."""

    balanced = _scored_genome("balanced", (0.9, 0.9, 0.9, 0.9, 0.9), fitness=0.1)
    task_extreme = _scored_genome("task_extreme", (1.0, 0.1, 0.1, 0.1, 0.1))
    quality_extreme = _scored_genome("quality_extreme", (0.1, 1.0, 0.1, 0.1, 0.1))
    dominated_mid = _scored_genome("dominated_mid", (0.8, 0.8, 0.8, 0.8, 0.8))
    dominated_low = _scored_genome("dominated_low", (0.5, 0.5, 0.5, 0.5, 0.5))

    assert dominates(balanced.multi_scores, dominated_mid.multi_scores)
    assert not dominates(balanced.multi_scores, task_extreme.multi_scores)

    population = Population(
        (
            dominated_low,
            dominated_mid,
            task_extreme,
            balanced,
            quality_extreme,
        )
    )
    fronts = fast_non_dominated_sort(list(population.genomes))

    assert {genome.id for genome in fronts[0]} == {
        "task_extreme",
        "balanced",
        "quality_extreme",
    }
    assert {genome.id for genome in fronts[1]} == {"dominated_mid"}
    assert {genome.id for genome in fronts[2]} == {"dominated_low"}

    selected = nsga2_select(population, 3)
    assert {genome.id for genome in selected.genomes} == {
        "task_extreme",
        "balanced",
        "quality_extreme",
    }


def _parent_genome(
    genome_id: str, *, active_at: tuple[int, int], rule: int = 30, steps: int = 3
) -> Genome:
    grid = np.zeros((5, 5), dtype=int)
    grid[active_at] = 1
    seed = CASeed(grid=grid, rule=rule, steps=steps)
    return Genome(
        seed=seed,
        lora_cfg=LoRAConfig(
            r=4,
            alpha=8,
            dropout=0.05,
            target_modules=("q_proj", "v_proj"),
        ),
        id=genome_id,
        ca_features=extract_features(evolve(seed)),
    )


def _scored_genome(
    genome_id: str, values: tuple[float, float, float, float, float], fitness=1.0
) -> Genome:
    return Genome(
        seed=CASeed(grid=np.zeros((1, 1), dtype=int), rule=0, steps=1),
        lora_cfg=LoRAConfig(r=4, alpha=8, dropout=0.05, target_modules=("q_proj",)),
        id=genome_id,
        fitness=fitness,
        multi_scores=MultiObjectiveScores(
            task_score=values[0],
            quality_score=values[1],
            risk_score=values[2],
            efficiency_score=values[3],
            validity_score=values[4],
        ),
    )


class _ForcedLoRAMutationRng:
    """Minimal deterministic RNG that forces direct LoRA mutation."""

    def __init__(self):
        self._draws = iter((0.9, 0.0, 0.0, 0.0))

    def randint(self, _start, _end):
        return 4321

    def random(self):
        return next(self._draws)

    def choice(self, values):
        return values[-1]
