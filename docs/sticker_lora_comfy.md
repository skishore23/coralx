# Sticker LoRA Comfy Smoke Target

This is a deliberately small image target for CORAL-X: train a style LoRA that
pushes a base SD-style model toward centered sticker icons with thick outlines,
white borders, and clean vector-like composition.

The first useful run should prove plumbing, not quality. Use 24 synthetic
training images, rank 8 LoRA, 300 steps, batch size 1, and ComfyUI generation for
before/after samples.

## Local Setup

Create the dataset and Comfy workflow:

```bash
python scripts/sticker_lora_comfy.py setup --overwrite
```

Check what is still missing:

```bash
python scripts/sticker_lora_comfy.py check
```

Expected local paths:

- ComfyUI root: `/Users/kishore/qrates/ComfyUI`
- Checkpoints: `/Users/kishore/qrates/ComfyUI/models/checkpoints`
- LoRAs: `/Users/kishore/qrates/ComfyUI/models/loras`
- Dataset: `datasets/sticker_lora/10_cxsticker`
- Workflow: `artifacts/sticker_lora_comfy/workflow_sticker_lora_api.json`
- UI workflow: `artifacts/sticker_lora_comfy/workflow_sticker_lora_ui_v1.json`

## Current Local State

This machine is now set up for the sticker smoke target:

- SD1.5 fp16 checkpoint:
  `/Users/kishore/qrates/ComfyUI/models/checkpoints/v1-5-pruned-emaonly-fp16.safetensors`
- Trainer:
  `/Users/kishore/sd-scripts/train_network.py`
- Trained LoRA:
  `artifacts/sticker_lora_comfy/loras/cxsticker_v1.safetensors`
- Comfy-visible LoRA:
  `/Users/kishore/qrates/ComfyUI/models/loras/cxsticker_v1.safetensors`
- Generated comparison:
  `artifacts/sticker_lora_comfy/comparison_base_vs_lora.png`

The local `/Users/kishore/qrates/ComfyUI` checkout has source-level API
mismatches, so a clean worktree is used for generation:

```text
artifacts/sticker_lora_comfy/ComfyUI_clean
```

## Repeat Training

For the same 300-step smoke run:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=/Users/kishore/sd-scripts \
accelerate launch --num_cpu_threads_per_process 4 \
  /Users/kishore/sd-scripts/train_network.py \
  --pretrained_model_name_or_path /Users/kishore/qrates/ComfyUI/models/checkpoints/v1-5-pruned-emaonly-fp16.safetensors \
  --train_data_dir datasets/sticker_lora \
  --resolution 512,512 \
  --output_dir artifacts/sticker_lora_comfy/loras \
  --logging_dir artifacts/sticker_lora_comfy/logs \
  --output_name cxsticker_v1 \
  --network_module networks.lora \
  --network_dim 8 \
  --network_alpha 8 \
  --network_train_unet_only \
  --train_batch_size 1 \
  --max_train_steps 300 \
  --learning_rate 1e-4 \
  --unet_lr 1e-4 \
  --caption_extension .txt \
  --cache_latents \
  --mixed_precision no \
  --save_model_as safetensors \
  --optimizer_type AdamW \
  --lr_scheduler constant \
  --max_data_loader_n_workers 0 \
  --seed 7 \
  --gradient_checkpointing \
  --save_every_n_steps 300
```

The helper can also print the command:

```bash
python scripts/sticker_lora_comfy.py train-command \
  --checkpoint-path /Users/kishore/qrates/ComfyUI/models/checkpoints/v1-5-pruned-emaonly-fp16.safetensors
```

After training, copy the output LoRA into:

```text
/Users/kishore/qrates/ComfyUI/models/loras/cxsticker_v1.safetensors
```

Then start the clean ComfyUI worktree and load/post:

```bash
cd artifacts/sticker_lora_comfy/ComfyUI_clean
PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py --listen 127.0.0.1 --port 8188
```

```text
artifacts/sticker_lora_comfy/workflow_sticker_lora_api.json
```

The API workflow is for `/prompt` execution. To load the graph visually in the
Comfy browser canvas, use the UI workflow instead:

```text
artifacts/sticker_lora_comfy/workflow_sticker_lora_ui_v1.json
```

For the in-app browser, this helper page writes the UI workflow into Comfy's
local workflow state and redirects back to the canvas:

```text
http://127.0.0.1:8188/coralx_load_sticker_workflow.html
```

## Why Stickers

Sticker LoRA is a useful first image target because quality is easy to inspect
and score:

- centered subject
- thick dark outline
- white sticker border
- simple background
- style consistency across different objects

Once this smoke path works, CORAL-X can evolve LoRA rank, alpha, learning rate,
caption wording, repeats, training steps, Comfy prompt, LoRA strength, sampler,
CFG, and seed policy.

## Sticker Proxy Score

The current evaluator is intentionally stricter than the first smoke-run proxy.
It rewards:

- a clean white/plain image border
- one dominant connected foreground object
- centered object mass
- reasonable object area
- bold outlines
- local white rim/halo around the foreground object

It penalizes:

- textured or photo-like borders
- non-white or low-plainness backgrounds
- repeated foreground components
- excessive edge clutter

This score is still a proxy, not a semantic judge. It does not prove the object
is the requested class, and it should eventually be paired with CLIP, an object
detector, a VLM checklist, or human review.

## COCO Object Benchmark

The first reusable benchmark uses COCO class names as held-out prompt subjects.
This does not download COCO images. It uses the 80 object categories as a fixed
prompt list so evolution has to generalize across known object names.

Splits:

```text
coco/train: first 40 COCO object names
coco/dev:   next 20 COCO object names
coco/test:  final 20 COCO object names
```

Quick COCO smoke run:

```bash
python scripts/sticker_lora_evolve.py \
  --benchmark coco \
  --split dev \
  --subjects 2 \
  --population 2 \
  --generations 1 \
  --seed 505
```

## Core Plugin Smoke Run

The first-class CORAL-X plugin target uses the same Comfy workflow and existing
LoRA files, but runs through the normal plugin registry and orchestrator. It
does not train a new LoRA per genome; the core genome is mapped into Comfy
inference settings such as LoRA choice, strengths, CFG, steps, prompt template,
negative prompt, scheduler, and seed offset.

Validate the plugin wiring without contacting Comfy:

```bash
python -m core.cli.main run \
  --config config/examples/sticker_lora_comfy_smoke.yaml \
  --dry-run
```

After ComfyUI is running on `127.0.0.1:8188`, run the core target:

```bash
python -m core.cli.main run \
  --config config/examples/sticker_lora_comfy_smoke.yaml
```

The plugin fails fast if the API workflow is missing, required workflow nodes
are absent, or the Comfy API is not reachable. Use
`experiment.evaluation.workflow_path`, `comfy_output_dir`, `api_url`, and
`prompt_timeout` to point it at a different local Comfy setup.

Longer dev optimization:

```bash
python scripts/sticker_lora_evolve.py \
  --benchmark coco \
  --split dev \
  --subjects 20 \
  --population 12 \
  --generations 8 \
  --seed 202
```

Resume a stopped or timed-out run with the same run parameters:

```bash
python scripts/sticker_lora_evolve.py \
  --benchmark coco \
  --split dev \
  --subjects 20 \
  --population 12 \
  --generations 8 \
  --seed 202 \
  --resume \
  --prompt-timeout 900
```

Resume skips completed full candidates already written to
`candidate_evaluations.jsonl`. If a timeout happened mid-candidate, that
candidate is rerun from the beginning.

Final held-out test should be run only after choosing a completed dev run:

```bash
python scripts/sticker_lora_evolve.py \
  --benchmark coco \
  --split test \
  --subjects 20 \
  --candidate-json artifacts/sticker_lora_comfy/evolution/run_coco_dev_seed_202_p12_g8_s20/best.json \
  --seed 202
```
