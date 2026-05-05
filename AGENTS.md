# CORAL-X Agent Instructions

These instructions apply to the whole repository. Future agents should treat them
as the default operating contract for any code, config, documentation, or
integration work in CORAL-X.

## Core Rule

All durable experiment work must use the CORAL-X framework:

- Configuration enters through `CoralConfig` and files under `config/examples/`.
- Targets are selected with `experiment.target` and resolved by `plugins/registry.py`.
- Target-specific behavior belongs behind the plugin boundary: `DatasetProvider`,
  `ModelRunner`, and `FitnessFn`.
- Evolution runs through `core.application.evolution_orchestrator` and the CLI in
  `core.cli.main`.
- Results, cache paths, proof reports, and benchmark artifacts must be reachable
  from normal framework runs, not only from standalone scripts.

Do not ship a new benchmark, model path, image workflow, proof path, or dataset
integration as a script-only feature. Scripts may be used for setup, migration,
asset generation, local diagnostics, or one-off developer utilities, but the
production path must be a first-class CORAL-X target.

## Required Shape For New Targets

When adding or changing an experiment target, provide all of these pieces:

- A plugin package or module under `plugins/` exposing dataset, model factory,
  and fitness function behavior.
- A registry entry so `experiment.target` resolves through `create_plugin()`.
- A runnable example config under `config/examples/`.
- A dry-run path that succeeds without optional external services when possible.
- Clear fail-fast validation for required files, models, datasets, API servers,
  workflow nodes, credentials, and hardware assumptions.
- Tests that exercise the plugin boundary without downloading large models or
  requiring live external services.
- Documentation that explains the framework command, not just helper scripts.

The minimum acceptance command for a new target should look like:

```bash
python -m core.cli.main run --config config/examples/<target>.yaml --dry-run
```

If the target needs an external runtime, also document the real run command and
the readiness check that proves the runtime is available.

## Integration Boundaries

Keep the architecture boundaries clean:

- `core/domain/`: pure data structures and deterministic transformations only.
- `core/application/`: orchestration and service wiring.
- `core/cli/`: user-facing command entry points and report presentation.
- `infra/`: executors, cache, and external execution mechanics.
- `plugins/`: experiment-specific datasets, model runners, and fitness logic.
- `scripts/`: setup and diagnostics only unless the same behavior is also wired
  through the framework.

If a script contains logic that determines benchmark behavior, candidate
evaluation, or scoring, move or wrap that logic behind a plugin runner before
calling the work complete.

## Framework-First Checklist

Before finishing work, verify these questions:

- Can a user discover the feature from a config file and `experiment.target`?
- Does the feature run through `python -m core.cli.main run` or
  `python -m core.cli.main prove`?
- Does the plugin expose real dataset, runner, and fitness responsibilities
  instead of calling an opaque script as the only integration point?
- Are failures explicit and early when external assets are missing?
- Are objective scores mapped into `MultiObjectiveScores` with meaningful
  dimensions instead of cloning one scalar across every field?
- Are artifact paths controlled by config and compatible with cache/output
  conventions?
- Do tests cover config validation, registry resolution, and the plugin boundary?
- Is documentation written around the framework command first, with scripts
  described only as setup or helper tools?

## Validation Gates

Run the focused gates appropriate to the change. For framework or plugin changes,
prefer the full gate:

```bash
python -m compileall -q core infra plugins tests
python -m pytest -q
python -m ruff check core infra plugins tests
python -m black --check .
python -m mypy core infra plugins
python -m bandit -r core infra plugins --severity-level medium
git diff --check
```

For documentation-only changes, `git diff --check` is sufficient unless the docs
include runnable examples that need validation.

## ComfyUI And Other External Runtimes

External runtimes are allowed, but they must be framework-owned:

- Keep local setup and health checks in scripts when useful.
- Put candidate rendering, waiting, scoring, and metric mapping behind the
  plugin runner.
- Validate runtime inputs before submitting work to the external service.
- Provide a smoke config that dry-runs through CORAL-X without requiring the
  external service.
- Document the real framework run once the external service is running.

For ComfyUI specifically, the durable path is `experiment.target:
sticker_lora_comfy`, not direct use of `scripts/sticker_lora_evolve.py` as the
primary entry point.

