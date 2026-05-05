"""Prepare a small sticker-style LoRA experiment for ComfyUI.

This script does not train by itself. It creates a deterministic toy sticker
dataset, writes a ComfyUI API workflow, checks local readiness, and prints the
smallest practical sd-scripts command for a real SD1.5 LoRA smoke run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

DEFAULT_COMFY_ROOT = Path("/Users/kishore/qrates/ComfyUI")
DEFAULT_DATASET_ROOT = Path("datasets/sticker_lora")
DEFAULT_ARTIFACT_ROOT = Path("artifacts/sticker_lora_comfy")
DEFAULT_TOKEN = "cxsticker"
DEFAULT_API_URL = "http://127.0.0.1:8188"
DEFAULT_CHECKPOINT = "v1-5-pruned-emaonly-fp16.safetensors"

SUBJECTS = (
    "rocket",
    "coffee cup",
    "lightning bolt",
    "camera",
    "skateboard",
    "planet",
    "sneaker",
    "pizza slice",
    "game controller",
    "paint brush",
    "music note",
    "sun",
)

PALE_BACKGROUNDS = (
    (255, 250, 240),
    (245, 252, 255),
    (250, 248, 255),
    (247, 255, 246),
)

FILL_COLORS = (
    (239, 91, 91),
    (67, 170, 139),
    (87, 117, 144),
    (249, 199, 79),
    (249, 132, 74),
    (144, 97, 249),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rounded_polygon(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
    width: int,
) -> None:
    draw.polygon(points, fill=fill)
    draw.line(points + [points[0]], fill=outline, width=width, joint="curve")


def _star_points(cx: float, cy: float, outer: float, inner: float, n: int = 5):
    points = []
    for idx in range(n * 2):
        radius = outer if idx % 2 == 0 else inner
        angle = -math.pi / 2 + idx * math.pi / n
        points.append((cx + math.cos(angle) * radius, cy + math.sin(angle) * radius))
    return points


def _draw_sticker_border(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]]):
    draw.line(points + [points[0]], fill=(255, 255, 255), width=42, joint="curve")
    draw.line(points + [points[0]], fill=(20, 20, 20), width=16, joint="curve")


def _draw_rocket(draw: ImageDraw.ImageDraw, fill: tuple[int, int, int]) -> None:
    body = [(256, 86), (342, 242), (286, 405), (226, 405), (170, 242)]
    _draw_sticker_border(draw, body)
    _rounded_polygon(draw, body, fill, (18, 18, 18), 10)
    draw.ellipse((223, 176, 289, 242), fill=(180, 230, 255), outline=(18, 18, 18), width=9)
    draw.polygon([(170, 275), (103, 350), (192, 335)], fill=(249, 132, 74), outline=(18, 18, 18))
    draw.polygon([(342, 275), (409, 350), (320, 335)], fill=(249, 132, 74), outline=(18, 18, 18))
    draw.polygon([(229, 405), (256, 470), (283, 405)], fill=(255, 209, 102), outline=(18, 18, 18))


def _draw_coffee(draw: ImageDraw.ImageDraw, fill: tuple[int, int, int]) -> None:
    draw.rounded_rectangle((145, 145, 330, 405), radius=34, fill=(255, 255, 255), outline=(18, 18, 18), width=18)
    draw.rounded_rectangle((166, 184, 309, 380), radius=24, fill=fill, outline=(18, 18, 18), width=8)
    draw.arc((300, 220, 425, 340), start=-70, end=80, fill=(18, 18, 18), width=20)
    draw.arc((318, 238, 394, 322), start=-70, end=80, fill=(255, 255, 255), width=16)
    for x_pos in (205, 256, 303):
        draw.arc((x_pos - 16, 85, x_pos + 16, 158), start=205, end=330, fill=(18, 18, 18), width=7)


def _draw_lightning(draw: ImageDraw.ImageDraw, fill: tuple[int, int, int]) -> None:
    points = [(301, 70), (151, 277), (252, 277), (204, 442), (374, 221), (268, 221)]
    _draw_sticker_border(draw, points)
    _rounded_polygon(draw, points, fill, (18, 18, 18), 11)


def _draw_camera(draw: ImageDraw.ImageDraw, fill: tuple[int, int, int]) -> None:
    draw.rounded_rectangle((104, 164, 408, 386), radius=45, fill=(255, 255, 255), outline=(18, 18, 18), width=18)
    draw.rounded_rectangle((138, 195, 374, 360), radius=28, fill=fill, outline=(18, 18, 18), width=8)
    draw.ellipse((196, 195, 316, 315), fill=(245, 252, 255), outline=(18, 18, 18), width=12)
    draw.ellipse((230, 229, 282, 281), fill=(87, 117, 144), outline=(18, 18, 18), width=6)
    draw.rounded_rectangle((151, 130, 245, 190), radius=18, fill=(249, 199, 79), outline=(18, 18, 18), width=8)


def _draw_board(draw: ImageDraw.ImageDraw, fill: tuple[int, int, int]) -> None:
    draw.rounded_rectangle((106, 225, 406, 319), radius=47, fill=(255, 255, 255), outline=(18, 18, 18), width=20)
    draw.rounded_rectangle((139, 248, 373, 296), radius=24, fill=fill, outline=(18, 18, 18), width=7)
    for x_pos in (177, 335):
        draw.ellipse((x_pos - 32, 313, x_pos + 32, 377), fill=(87, 117, 144), outline=(18, 18, 18), width=8)


def _draw_planet(draw: ImageDraw.ImageDraw, fill: tuple[int, int, int]) -> None:
    draw.ellipse((142, 130, 370, 358), fill=(255, 255, 255), outline=(18, 18, 18), width=20)
    draw.ellipse((165, 153, 347, 335), fill=fill, outline=(18, 18, 18), width=7)
    draw.arc((70, 170, 442, 358), start=12, end=168, fill=(18, 18, 18), width=20)
    draw.arc((82, 184, 430, 344), start=12, end=168, fill=(255, 255, 255), width=9)
    draw.ellipse((204, 201, 237, 234), fill=(255, 255, 255), outline=(18, 18, 18), width=5)
    draw.ellipse((281, 262, 319, 300), fill=(255, 255, 255), outline=(18, 18, 18), width=5)


def _draw_sneaker(draw: ImageDraw.ImageDraw, fill: tuple[int, int, int]) -> None:
    points = [(104, 315), (185, 236), (297, 269), (391, 309), (421, 364), (102, 364)]
    _draw_sticker_border(draw, points)
    _rounded_polygon(draw, points, fill, (18, 18, 18), 10)
    draw.rounded_rectangle((123, 344, 408, 387), radius=20, fill=(255, 255, 255), outline=(18, 18, 18), width=8)
    for x_pos in (230, 264, 298):
        draw.line((x_pos, 281, x_pos + 32, 312), fill=(255, 255, 255), width=8)


def _draw_pizza(draw: ImageDraw.ImageDraw, fill: tuple[int, int, int]) -> None:
    points = [(156, 107), (382, 152), (245, 426)]
    _draw_sticker_border(draw, points)
    _rounded_polygon(draw, points, fill, (18, 18, 18), 10)
    draw.line((169, 137, 358, 174), fill=(249, 132, 74), width=28)
    for cx, cy in ((242, 204), (286, 274), (230, 331)):
        draw.ellipse((cx - 17, cy - 17, cx + 17, cy + 17), fill=(239, 91, 91), outline=(18, 18, 18), width=4)


def _draw_controller(draw: ImageDraw.ImageDraw, fill: tuple[int, int, int]) -> None:
    draw.rounded_rectangle((96, 203, 416, 352), radius=58, fill=(255, 255, 255), outline=(18, 18, 18), width=18)
    draw.rounded_rectangle((126, 226, 386, 330), radius=38, fill=fill, outline=(18, 18, 18), width=8)
    draw.line((168, 280, 224, 280), fill=(18, 18, 18), width=12)
    draw.line((196, 252, 196, 308), fill=(18, 18, 18), width=12)
    for cx, cy in ((309, 264), (347, 292)):
        draw.ellipse((cx - 15, cy - 15, cx + 15, cy + 15), fill=(255, 255, 255), outline=(18, 18, 18), width=5)


def _draw_brush(draw: ImageDraw.ImageDraw, fill: tuple[int, int, int]) -> None:
    draw.line((161, 361, 341, 181), fill=(255, 255, 255), width=52)
    draw.line((161, 361, 341, 181), fill=(18, 18, 18), width=36)
    draw.line((161, 361, 341, 181), fill=fill, width=20)
    bristles = [(331, 175), (394, 111), (412, 210), (353, 231)]
    _rounded_polygon(draw, bristles, (239, 91, 91), (18, 18, 18), 8)


def _draw_music(draw: ImageDraw.ImageDraw, fill: tuple[int, int, int]) -> None:
    draw.line((308, 109, 308, 338), fill=(255, 255, 255), width=42)
    draw.line((308, 109, 308, 338), fill=(18, 18, 18), width=25)
    draw.line((308, 109, 398, 145), fill=(18, 18, 18), width=25)
    draw.ellipse((165, 308, 281, 412), fill=(255, 255, 255), outline=(18, 18, 18), width=20)
    draw.ellipse((190, 327, 264, 394), fill=fill, outline=(18, 18, 18), width=7)


def _draw_sun(draw: ImageDraw.ImageDraw, fill: tuple[int, int, int]) -> None:
    points = _star_points(256, 256, 183, 130, n=12)
    _draw_sticker_border(draw, points)
    _rounded_polygon(draw, points, fill, (18, 18, 18), 9)
    draw.ellipse((164, 164, 348, 348), fill=(255, 247, 130), outline=(18, 18, 18), width=9)


DRAWERS: dict[str, Callable[[ImageDraw.ImageDraw, tuple[int, int, int]], None]] = {
    "rocket": _draw_rocket,
    "coffee cup": _draw_coffee,
    "lightning bolt": _draw_lightning,
    "camera": _draw_camera,
    "skateboard": _draw_board,
    "planet": _draw_planet,
    "sneaker": _draw_sneaker,
    "pizza slice": _draw_pizza,
    "game controller": _draw_controller,
    "paint brush": _draw_brush,
    "music note": _draw_music,
    "sun": _draw_sun,
}


def create_sticker_dataset(
    output_root: Path,
    count: int,
    token: str,
    seed: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create a deterministic kohya/sd-scripts compatible image-caption dataset."""
    rng = random.Random(seed)
    dataset_dir = output_root / f"10_{token}"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    existing = list(dataset_dir.glob("*.png")) + list(dataset_dir.glob("*.txt"))
    if existing and not overwrite:
        raise FileExistsError(
            f"{dataset_dir} already contains files. Use --overwrite to replace them."
        )

    if overwrite:
        for path in existing:
            path.unlink()

    records = []
    for idx in range(count):
        subject = SUBJECTS[idx % len(SUBJECTS)]
        fill = FILL_COLORS[(idx + rng.randrange(len(FILL_COLORS))) % len(FILL_COLORS)]
        bg = PALE_BACKGROUNDS[(idx + rng.randrange(len(PALE_BACKGROUNDS))) % len(PALE_BACKGROUNDS)]
        image = Image.new("RGB", (512, 512), bg)
        draw = ImageDraw.Draw(image)
        DRAWERS[subject](draw, fill)

        image_path = dataset_dir / f"{token}_{idx:03d}.png"
        caption_path = dataset_dir / f"{token}_{idx:03d}.txt"
        caption = (
            f"{token}, sticker style icon of a {subject}, thick black outline, "
            "white border, clean vector art, centered subject, simple background"
        )
        image.save(image_path, optimize=True)
        caption_path.write_text(caption + "\n", encoding="utf-8")
        records.append(
            {
                "image": str(image_path),
                "caption": caption,
                "sha256": _sha256_file(image_path),
            }
        )

    manifest = {
        "created_at": int(time.time()),
        "token": token,
        "count": count,
        "seed": seed,
        "dataset_dir": str(dataset_dir),
        "records": records,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build_comfy_workflow(
    checkpoint_name: str,
    prompt: str,
    negative_prompt: str,
    lora_name: str | None,
    seed: int,
    steps: int,
    cfg: float,
    width: int,
    height: int,
    filename_prefix: str,
) -> dict[str, Any]:
    """Build a minimal ComfyUI API workflow for SD-style image generation."""
    if lora_name:
        model_ref: list[Any] = ["8", 0]
        clip_ref: list[Any] = ["8", 1]
    else:
        model_ref = ["1", 0]
        clip_ref = ["1", 1]

    workflow: dict[str, Any] = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint_name},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": clip_ref},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": clip_ref},
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": model_ref,
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
            },
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
        },
        "7": {
            "class_type": "SaveImage",
            "inputs": {"images": ["6", 0], "filename_prefix": filename_prefix},
        },
    }

    if lora_name:
        workflow["8"] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["1", 0],
                "clip": ["1", 1],
                "lora_name": lora_name,
                "strength_model": 0.85,
                "strength_clip": 0.85,
            },
        }

    return workflow


def write_workflow(
    output_path: Path,
    checkpoint_name: str,
    token: str,
    lora_name: str | None,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workflow = build_comfy_workflow(
        checkpoint_name=checkpoint_name,
        prompt=(
            f"{token}, sticker style icon of a rocket, thick black outline, "
            "white border, clean vector art, centered subject, simple background"
        ),
        negative_prompt="photo, realistic, blurry, low contrast, cropped, text, watermark",
        lora_name=lora_name,
        seed=123456789,
        steps=24,
        cfg=7.0,
        width=512,
        height=512,
        filename_prefix=f"{token}_sample",
    )
    output_path.write_text(json.dumps(workflow, indent=2) + "\n", encoding="utf-8")
    return workflow


def _is_comfy_root(path: Path) -> bool:
    return (path / "main.py").exists() and (path / "models").is_dir()


def find_comfy_root(explicit: Path | None) -> Path | None:
    candidates = []
    if explicit:
        candidates.append(explicit)
    candidates.extend(
        [
            DEFAULT_COMFY_ROOT,
            Path.home() / "ComfyUI",
            Path.home() / "comfyui",
            Path.home() / "qrates" / "ComfyUI",
        ]
    )
    for candidate in candidates:
        if _is_comfy_root(candidate):
            return candidate
    return None


def list_model_files(path: Path) -> list[str]:
    if not path.exists():
        return []
    suffixes = {".safetensors", ".ckpt", ".pt"}
    return sorted(str(file.relative_to(path)) for file in path.rglob("*") if file.suffix in suffixes)


def find_trainer_script() -> str | None:
    env_path = os.environ.get("CORALX_SD_SCRIPTS")
    candidates = []
    if env_path:
        candidates.append(Path(env_path) / "train_network.py")
    candidates.extend(
        [
            Path.home() / "sd-scripts" / "train_network.py",
            Path.home() / "kohya_ss" / "sd-scripts" / "train_network.py",
            Path.home() / "kohya_ss" / "train_network.py",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def check_api(api_url: str, timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(f"{api_url.rstrip('/')}/system_stats", timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


def readiness_report(
    comfy_root: Path | None,
    api_url: str,
    dataset_root: Path,
) -> dict[str, Any]:
    checkpoints: list[str] = []
    loras: list[str] = []
    if comfy_root:
        checkpoints = list_model_files(comfy_root / "models" / "checkpoints")
        loras = list_model_files(comfy_root / "models" / "loras")

    images = sorted(dataset_root.rglob("*.png")) if dataset_root.exists() else []
    captions = sorted(dataset_root.rglob("*.txt")) if dataset_root.exists() else []
    return {
        "comfy_root": str(comfy_root) if comfy_root else None,
        "comfy_api_alive": check_api(api_url),
        "checkpoint_count": len(checkpoints),
        "checkpoints": checkpoints,
        "lora_count": len(loras),
        "loras": loras,
        "trainer_script": find_trainer_script(),
        "dataset_root": str(dataset_root),
        "dataset_image_count": len(images),
        "dataset_caption_count": len(captions),
        "ready_for_comfy_generation": bool(comfy_root and checkpoints),
        "ready_for_training": bool(checkpoints and find_trainer_script() and images and captions),
    }


def build_train_command(
    trainer_script: str,
    checkpoint_path: str,
    dataset_root: Path,
    artifact_root: Path,
    output_name: str,
) -> list[str]:
    output_dir = artifact_root / "loras"
    log_dir = artifact_root / "logs"
    return [
        "accelerate",
        "launch",
        trainer_script,
        "--pretrained_model_name_or_path",
        checkpoint_path,
        "--train_data_dir",
        str(dataset_root),
        "--resolution",
        "512,512",
        "--output_dir",
        str(output_dir),
        "--logging_dir",
        str(log_dir),
        "--output_name",
        output_name,
        "--network_module",
        "networks.lora",
        "--network_dim",
        "8",
        "--network_alpha",
        "8",
        "--train_batch_size",
        "1",
        "--max_train_steps",
        "300",
        "--learning_rate",
        "1e-4",
        "--unet_lr",
        "1e-4",
        "--text_encoder_lr",
        "5e-5",
        "--caption_extension",
        ".txt",
        "--cache_latents",
        "--mixed_precision",
        "no",
        "--save_model_as",
        "safetensors",
    ]


def print_report(report: dict[str, Any]) -> None:
    print(json.dumps(report, indent=2))
    missing = []
    if not report["comfy_root"]:
        missing.append("ComfyUI root")
    if report["checkpoint_count"] == 0:
        missing.append("SD checkpoint in ComfyUI/models/checkpoints")
    if not report["trainer_script"]:
        missing.append("sd-scripts train_network.py")
    if report["dataset_image_count"] == 0:
        missing.append("sticker dataset")
    if missing:
        print("\nMissing:", ", ".join(missing))


def cmd_check(args: argparse.Namespace) -> None:
    comfy_root = find_comfy_root(args.comfy_root)
    report = readiness_report(comfy_root, args.api_url, args.dataset_root)
    print_report(report)


def cmd_dataset(args: argparse.Namespace) -> None:
    manifest = create_sticker_dataset(
        output_root=args.output,
        count=args.count,
        token=args.token,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "dataset_dir": manifest["dataset_dir"],
                "count": manifest["count"],
                "manifest": str(args.output / "manifest.json"),
            },
            indent=2,
        )
    )


def cmd_workflow(args: argparse.Namespace) -> None:
    workflow = write_workflow(
        output_path=args.output,
        checkpoint_name=args.checkpoint,
        token=args.token,
        lora_name=args.lora,
    )
    print(json.dumps({"workflow": str(args.output), "nodes": sorted(workflow)}, indent=2))


def cmd_setup(args: argparse.Namespace) -> None:
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    manifest = create_sticker_dataset(
        output_root=args.dataset_root,
        count=args.count,
        token=args.token,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    workflow_path = args.artifact_root / "workflow_sticker_lora_api.json"
    write_workflow(
        output_path=workflow_path,
        checkpoint_name=args.checkpoint,
        token=args.token,
        lora_name=args.lora,
    )
    comfy_root = find_comfy_root(args.comfy_root)
    report = readiness_report(comfy_root, args.api_url, args.dataset_root)
    setup_report = {
        "dataset_dir": manifest["dataset_dir"],
        "workflow": str(workflow_path),
        "readiness": report,
    }
    (args.artifact_root / "setup_report.json").write_text(
        json.dumps(setup_report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(setup_report, indent=2))


def cmd_train_command(args: argparse.Namespace) -> None:
    trainer_script = args.trainer_script or find_trainer_script()
    if not trainer_script:
        print(
            "sd-scripts train_network.py was not found. Set CORALX_SD_SCRIPTS or pass --trainer-script.",
            file=sys.stderr,
        )
        sys.exit(2)
    command = build_train_command(
        trainer_script=trainer_script,
        checkpoint_path=args.checkpoint_path,
        dataset_root=args.dataset_root,
        artifact_root=args.artifact_root,
        output_name=args.output_name,
    )
    print(" ".join(command))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(func=None)
    subparsers = parser.add_subparsers(dest="command")

    check = subparsers.add_parser("check", help="Check local Comfy/trainer readiness")
    check.add_argument("--comfy-root", type=Path, default=None)
    check.add_argument("--api-url", default=DEFAULT_API_URL)
    check.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    check.set_defaults(func=cmd_check)

    dataset = subparsers.add_parser("dataset", help="Create the sticker training dataset")
    dataset.add_argument("--output", type=Path, default=DEFAULT_DATASET_ROOT)
    dataset.add_argument("--count", type=int, default=24)
    dataset.add_argument("--token", default=DEFAULT_TOKEN)
    dataset.add_argument("--seed", type=int, default=7)
    dataset.add_argument("--overwrite", action="store_true")
    dataset.set_defaults(func=cmd_dataset)

    workflow = subparsers.add_parser("workflow", help="Write a ComfyUI API workflow")
    workflow.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_ROOT / "workflow_sticker_lora_api.json")
    workflow.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    workflow.add_argument("--token", default=DEFAULT_TOKEN)
    workflow.add_argument("--lora", default=None)
    workflow.set_defaults(func=cmd_workflow)

    setup = subparsers.add_parser("setup", help="Create dataset, workflow, and readiness report")
    setup.add_argument("--comfy-root", type=Path, default=None)
    setup.add_argument("--api-url", default=DEFAULT_API_URL)
    setup.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    setup.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    setup.add_argument("--count", type=int, default=24)
    setup.add_argument("--token", default=DEFAULT_TOKEN)
    setup.add_argument("--seed", type=int, default=7)
    setup.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    setup.add_argument("--lora", default="cxsticker_v1.safetensors")
    setup.add_argument("--overwrite", action="store_true")
    setup.set_defaults(func=cmd_setup)

    train_command = subparsers.add_parser("train-command", help="Print a minimal sd-scripts training command")
    train_command.add_argument("--trainer-script", default=None)
    train_command.add_argument("--checkpoint-path", required=True)
    train_command.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    train_command.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    train_command.add_argument("--output-name", default="cxsticker_v1")
    train_command.set_defaults(func=cmd_train_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.func is None:
        parser.print_help()
        sys.exit(2)
    args.func(args)


if __name__ == "__main__":
    main()
