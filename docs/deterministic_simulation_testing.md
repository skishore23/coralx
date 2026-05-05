# Deterministic Simulation Testing

CORAL-X now has a reusable pytest harness in
`tests/support/deterministic_simulation.py` for checking the local evolution
core without model noise.

The harness runs the controlled `ca_onemax` target in memory and records a
path-independent trace of each generation:

- CA grid digest, active-cell count, rule, and step count
- mapped adapter parameters
- multi-objective scores and scalar fitness
- survivor ids and next-population ids

These tests deliberately do not require best fitness to improve in every tiny
run. Improvement is a benchmark/runtime property. The deterministic framework
checks the stronger core contracts that should always hold:

- same seed and config replay exactly, independent of output path
- different seeds produce different initial traces
- CA evolution is pure and uses documented neighborhood semantics
- NEAT-style mutation/crossover are replayable and remap CA features correctly
- Pareto selection uses non-dominated fronts rather than scalar fitness

Run the focused gate with:

```bash
python -m pytest tests/test_deterministic_simulation_framework.py -q
```

Run the broader local gate with:

```bash
python -m pytest -q
```
