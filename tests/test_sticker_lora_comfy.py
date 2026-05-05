from __future__ import annotations

import importlib.util
from pathlib import Path


def load_sticker_module():
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "sticker_lora_comfy.py"
    )
    spec = importlib.util.spec_from_file_location("sticker_lora_comfy", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sticker_dataset_is_deterministic(tmp_path):
    module = load_sticker_module()
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"

    first = module.create_sticker_dataset(
        first_root, count=3, token="cxsticker", seed=11
    )
    second = module.create_sticker_dataset(
        second_root, count=3, token="cxsticker", seed=11
    )

    assert [record["sha256"] for record in first["records"]] == [
        record["sha256"] for record in second["records"]
    ]
    assert len(list((first_root / "10_cxsticker").glob("*.png"))) == 3
    assert len(list((first_root / "10_cxsticker").glob("*.txt"))) == 3


def test_sticker_captions_include_trigger_token(tmp_path):
    module = load_sticker_module()
    module.create_sticker_dataset(tmp_path, count=2, token="cxsticker", seed=5)

    captions = sorted((tmp_path / "10_cxsticker").glob("*.txt"))
    assert captions
    assert all(
        path.read_text(encoding="utf-8").startswith("cxsticker,") for path in captions
    )


def test_workflow_uses_lora_loader_when_lora_is_set():
    module = load_sticker_module()

    workflow = module.build_comfy_workflow(
        checkpoint_name="base.safetensors",
        prompt="cxsticker, sticker style icon of a rocket",
        negative_prompt="blurry",
        lora_name="cxsticker_v1.safetensors",
        seed=1,
        steps=20,
        cfg=7.0,
        width=512,
        height=512,
        filename_prefix="cxsticker",
    )

    assert workflow["8"]["class_type"] == "LoraLoader"
    assert workflow["2"]["inputs"]["clip"] == ["8", 1]
    assert workflow["5"]["inputs"]["model"] == ["8", 0]


def test_readiness_report_blocks_training_without_checkpoint(tmp_path):
    module = load_sticker_module()
    comfy_root = tmp_path / "ComfyUI"
    (comfy_root / "models" / "checkpoints").mkdir(parents=True)
    (comfy_root / "models" / "loras").mkdir(parents=True)
    (comfy_root / "main.py").write_text("# fake comfy\n", encoding="utf-8")

    report = module.readiness_report(
        comfy_root=comfy_root,
        api_url="http://127.0.0.1:1",
        dataset_root=tmp_path / "missing_dataset",
    )

    assert report["checkpoint_count"] == 0
    assert report["ready_for_training"] is False
    assert report["ready_for_comfy_generation"] is False
