# Getting Started

Install local development dependencies:

```bash
pip install -e ".[dev]"
```

Validate the canonical config without running evolution:

```bash
python -m core.cli.main run --config config/examples/m1_tiny.yaml --dry-run
```

Run the local quickstart experiment:

```bash
python -m core.cli.main run --config config/examples/m1_tiny.yaml
```

Expected behavior:

- no model downloads
- no external services
- deterministic initial population for the configured seed
- local mock-mini evaluation

Run repository gates before check-in:

```bash
python -m compileall -q core infra plugins tests
python -m pytest -q
python -m ruff check core infra plugins tests
python -m mypy core infra plugins
```

Optional real-LoRA demo:

```bash
.venv/bin/python -m pip install -e ".[ml]"
.venv/bin/python -m core.cli.main run --config config/examples/gsm8k_lora_micro.yaml
```

This downloads a small Hugging Face model and GSM8K, so it is separate from the local quickstart run.

Optional proof report with controls:

```bash
.venv/bin/python -m core.cli.main prove --config config/examples/gsm8k_lora_micro.yaml --random-trials 4
```

This runs evolution, a fixed LoRA baseline, and same-budget random candidates, then writes a JSON report under `artifacts/gsm8k_lora_micro/`.

Stronger local math benchmark:

```bash
.venv/bin/python -m core.cli.main prove --config config/examples/gsm8k_math_benchmark.yaml --random-trials 12
```

This uses `Qwen/Qwen2.5-Math-1.5B-Instruct`, boxed-answer parsing, a larger GSM8K train/eval slice, and base/fixed/random controls. It is intended to run for hours on a Mac; use exact-answer accuracy in `proof_report.json` as the headline metric.

The April 27, 2026 local run was a negative proof result, not a marketing result: the base model scored higher exact accuracy than the evolved LoRA adapter, and the best random control matched evolution on exact accuracy.
