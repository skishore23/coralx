from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw


def load_evolve_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "sticker_lora_evolve.py"
    spec = importlib.util.spec_from_file_location("sticker_lora_evolve", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_coco_benchmark_splits_do_not_overlap():
    module = load_evolve_module()

    train = set(module.BENCHMARK_SPLITS["coco"]["train"])
    dev = set(module.BENCHMARK_SPLITS["coco"]["dev"])
    test = set(module.BENCHMARK_SPLITS["coco"]["test"])

    assert len(train) == 40
    assert len(dev) == 20
    assert len(test) == 20
    assert not train & dev
    assert not train & test
    assert not dev & test


def test_resolve_subjects_returns_deterministic_prefix():
    module = load_evolve_module()

    assert module.resolve_subjects("coco", "dev", 3) == ("wine glass", "cup", "fork")
    assert module.resolve_subjects("toy", "train", 2) == ("robot head", "donut")


def test_resolve_subjects_rejects_test_leakage_shape_errors():
    module = load_evolve_module()

    with pytest.raises(ValueError, match="exceeds coco/dev size"):
        module.resolve_subjects("coco", "dev", 21)

    with pytest.raises(ValueError, match="at least 1"):
        module.resolve_subjects("coco", "dev", 0)


def test_candidate_from_report_json_supports_held_out_eval(tmp_path):
    module = load_evolve_module()
    report = {
        "best": {
            "candidate": {
                "candidate_id": "g1_c03",
                "lora_name": "cxsticker_v2.safetensors",
                "strength_model": 0.4,
                "strength_clip": 0.57,
                "cfg": 7.3,
                "steps": 32,
                "prompt_template_id": 0,
                "negative_prompt_id": 1,
                "sampler_name": "euler",
                "scheduler": "normal",
                "seed_offset": 434206,
            }
        }
    }
    path = tmp_path / "report.json"
    path.write_text(module.json.dumps(report), encoding="utf-8")

    candidate = module.candidate_from_json(path, candidate_id="fixed_c00")

    assert candidate.candidate_id == "fixed_c00"
    assert candidate.lora_name == "cxsticker_v2.safetensors"


def test_load_completed_results_for_resume(tmp_path):
    module = load_evolve_module()
    path = tmp_path / "candidate_evaluations.jsonl"
    rows = [
        {"candidate": {"candidate_id": "g0_c00"}, "score": 0.1},
        {"candidate": {"candidate_id": "g0_c01"}, "score": 0.2},
    ]
    path.write_text("\n".join(module.json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    completed = module.load_completed_results(path)

    assert set(completed) == {"g0_c00", "g0_c01"}
    assert completed["g0_c01"]["score"] == 0.2


def write_clean_sticker(path: Path) -> None:
    image = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((60, 60, 196, 196), fill="black")
    draw.ellipse((72, 72, 184, 184), fill=(220, 40, 70))
    image.save(path)


def write_textured_background(path: Path) -> None:
    image = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(image)
    for y_pos in range(0, 256, 8):
        shade = 110 + (y_pos % 48)
        draw.rectangle((0, y_pos, 255, y_pos + 4), fill=(shade, shade, shade + 20))
    draw.ellipse((60, 60, 196, 196), fill="black")
    draw.ellipse((72, 72, 184, 184), fill=(220, 40, 70))
    image.save(path)


def write_repeated_objects(path: Path) -> None:
    image = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(image)
    for box in ((28, 72, 106, 150), (150, 72, 228, 150)):
        draw.ellipse(box, fill="black")
        inset = tuple(value + offset for value, offset in zip(box, (8, 8, -8, -8), strict=False))
        draw.ellipse(inset, fill=(220, 40, 70))
    image.save(path)


def test_image_proxy_penalizes_textured_background_and_repeated_objects(tmp_path):
    module = load_evolve_module()
    clean_path = tmp_path / "clean.png"
    textured_path = tmp_path / "textured.png"
    repeated_path = tmp_path / "repeated.png"
    write_clean_sticker(clean_path)
    write_textured_background(textured_path)
    write_repeated_objects(repeated_path)

    clean = module.image_proxy_metrics(clean_path)
    textured = module.image_proxy_metrics(textured_path)
    repeated = module.image_proxy_metrics(repeated_path)

    assert clean["score"] > textured["score"]
    assert clean["score"] > repeated["score"]
    assert clean["background"] > textured["background"]
    assert textured["border_texture_penalty"] > clean["border_texture_penalty"]
    assert repeated["component_count"] > clean["component_count"]
    assert repeated["repeated_object_penalty"] > clean["repeated_object_penalty"]
