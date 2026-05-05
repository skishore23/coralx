# Operational Proof Report - 2026-04-28

This report records the first local operational proof run after enforcing
held-out evaluation and multi-seed execution in the proof harness.

## GSM8K Prompt Evolution

Command:

```bash
python -m core.cli.main prove --config config/examples/gsm8k_prompt_quick.yaml --random-trials 1
```

Artifacts:

- Aggregate report: `artifacts/gsm8k_prompt_quick/proof_report.json`
- Seed reports:
  - `artifacts/gsm8k_prompt_quick/seed_42/proof_report.json`
  - `artifacts/gsm8k_prompt_quick/seed_43/proof_report.json`
  - `artifacts/gsm8k_prompt_quick/seed_44/proof_report.json`

Proof budget:

- Seeds: `42`, `43`, `44`
- Population size: `2`
- Generations: `1`
- Random trials per seed: `1`
- Evolved suite: `hybrid`
- Held-out split: enabled

Held-out exact accuracy:

- Base prompt: `0.0`
- Fixed strong-CoT prompt: `0.0`
- Random baseline: `0.0`
- Hybrid evolved prompt: `0.0`

Proof verdict:

- Multi-seed support: `true`
- Evolved beats base: `false`
- Evolved beats fixed: `false`
- Evolved beats random: `false`
- Held-out beats fixed: `false`
- Held-out beats random: `false`
- Passes: `false`

Interpretation:

The run is operationally valid evidence generation: it executed three isolated
seeds, produced per-seed reports, evaluated held-out test records, and wrote an
aggregate proof report. It is not positive performance evidence yet because the
quick one-example held-out slices all scored zero exact accuracy.

## Sticker / Comfy Status

The sticker proof is not yet operationally complete on this machine.

Observed blocker:

- `http://127.0.0.1:8188/system_stats` refused connection.
- `/Users/kishore/qrates/ComfyUI` exists.
- `/Users/kishore/qrates/ComfyUI/models/checkpoints` has no `.safetensors` checkpoints.
- `/Users/kishore/qrates/ComfyUI/models/loras` has no `.safetensors` LoRAs.
- `artifacts/sticker_lora_comfy` has no generated API workflow JSON.

Required before a real sticker proof report:

1. Run `python scripts/sticker_lora_comfy.py setup --overwrite`.
2. Install or point ComfyUI at the required checkpoint.
3. Train or copy the `cxsticker` LoRA artifacts into ComfyUI.
4. Start ComfyUI on `127.0.0.1:8188`.
5. Run `scripts/sticker_lora_evolve.py` on fixed dev and held-out test splits.
