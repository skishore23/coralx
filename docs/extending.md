# Extending CORAL-X

Add a plugin by implementing the plugin protocol, registering the target, and adding coverage for the expected run mode.

## Plugin Requirements

A plugin must implement the `Plugin` protocol:

```python
class Plugin(Protocol):
    def dataset(self) -> DatasetProvider: ...
    def model_factory(self) -> Callable[[LoRAConfig, Genome | None], ModelRunner]: ...
    def fitness_fn(self) -> FitnessFn: ...
```

Then register the target in `plugins/registry.py`.

## Acceptance Criteria

New targets need tests for:

- registry resolution
- no unintended plugin imports
- deterministic population creation
- local CLI dry-run validation
- full local run for targets intended to run without model downloads or external services

Targets that need downloads, private credentials, GPUs, or cloud services should document those requirements and keep import-time behavior lightweight. `quixbugs_gemma4` and `gsm8k_lora` are optional ML targets, so dry-run validation should not import model or dataset stacks.
