"""Run a small evolutionary benchmark for the sticker LoRA ComfyUI setup.

This evolves generation behavior around already-trained LoRAs. It does not
retrain LoRA weights. The goal is to create a repeatable image benchmark before
spending hours on training evolution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageStat

API_URL = "http://127.0.0.1:8188"
ARTIFACT_ROOT = Path("artifacts/sticker_lora_comfy")
COMFY_OUTPUT = ARTIFACT_ROOT / "ComfyUI_clean" / "output"
WORKFLOW_API = ARTIFACT_ROOT / "workflow_sticker_lora_api.json"

EVAL_SUBJECTS = (
    "robot head",
    "donut",
    "bicycle",
    "headphones",
    "umbrella",
    "trophy",
)

COCO_OBJECT_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

BENCHMARK_SUBJECTS = {
    "toy": EVAL_SUBJECTS,
    "coco": COCO_OBJECT_CLASSES,
}

BENCHMARK_SPLITS = {
    "toy": {
        "all": EVAL_SUBJECTS,
        "train": EVAL_SUBJECTS[:4],
        "dev": EVAL_SUBJECTS[4:],
        "test": EVAL_SUBJECTS,
    },
    "coco": {
        "all": COCO_OBJECT_CLASSES,
        "train": COCO_OBJECT_CLASSES[:40],
        "dev": COCO_OBJECT_CLASSES[40:60],
        "test": COCO_OBJECT_CLASSES[60:],
    },
}

PROMPT_TEMPLATES = (
    "cxsticker, one single {subject}, isolated die-cut sticker, centered full object, thick black outline, white sticker border, simple flat vector icon, clean plain white background",
    "cxsticker, cute {subject} sticker icon, bold black contour, white adhesive border, flat graphic style, single centered object, no background",
    "cxsticker, minimalist sticker of a {subject}, crisp black outline, white border, clean vector art, centered, plain background",
    "cxsticker, app icon style die-cut sticker showing one {subject}, thick outline, white rim, flat colors, centered composition",
)

NEGATIVE_PROMPTS = (
    "multiple objects, repeating pattern, collage, scene, background texture, text, watermark, signature, logo, cropped, realistic photo, messy details, extra objects",
    "text, watermark, signature, logo, photorealistic, busy background, pattern, many objects, cropped, low contrast, blur",
    "repeating pattern, typography, letters, words, realism, shadows, clutter, borderless, off center, extra objects",
)


@dataclass(frozen=True)
class StickerCandidate:
    """Generation genome for sticker inference."""

    candidate_id: str
    lora_name: str
    strength_model: float
    strength_clip: float
    cfg: float
    steps: int
    prompt_template_id: int
    negative_prompt_id: int
    sampler_name: str
    scheduler: str
    seed_offset: int

    def key(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def triangular_score(value: float, target: float, width: float) -> float:
    return clamp(1.0 - abs(value - target) / width)


def connected_components(
    mask: list[bool], width: int, height: int, min_size: int
) -> list[list[int]]:
    """Return foreground connected components from a flat boolean mask."""
    seen = [False] * len(mask)
    components = []
    for start, is_foreground in enumerate(mask):
        if not is_foreground or seen[start]:
            continue

        stack = [start]
        seen[start] = True
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            x_pos = current % width
            y_pos = current // width
            for nx_pos, ny_pos in (
                (x_pos - 1, y_pos),
                (x_pos + 1, y_pos),
                (x_pos, y_pos - 1),
                (x_pos, y_pos + 1),
            ):
                if nx_pos < 0 or ny_pos < 0 or nx_pos >= width or ny_pos >= height:
                    continue
                neighbor = ny_pos * width + nx_pos
                if mask[neighbor] and not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(neighbor)

        if len(component) >= min_size:
            components.append(component)
    return components


def image_proxy_metrics(path: Path) -> dict[str, float]:
    """Compute rough, local proxy metrics for sticker-likeness.

    This is not a semantic benchmark. It rewards one centered foreground object,
    a clean white/plain border, bold outlines, and low clutter. It penalizes
    textured/photo borders and repeated foreground components.
    """
    image = Image.open(path).convert("RGB").resize((256, 256))
    gray = image.convert("L")
    pixels = list(gray.getdata())
    rgb_pixels = list(image.getdata())
    width, height = image.size

    border = []
    border_indices = []
    border_width = 20
    for y_pos in range(height):
        for x_pos in range(width):
            if (
                x_pos < border_width
                or y_pos < border_width
                or x_pos >= width - border_width
                or y_pos >= height - border_width
            ):
                index = y_pos * width + x_pos
                border_indices.append(index)
                border.append(pixels[index])
    border_mean_raw = sum(border) / len(border)
    border_mean = border_mean_raw / 255.0
    border_std = (
        sum((v / 255.0 - border_mean) ** 2 for v in border) / len(border)
    ) ** 0.5

    edge = gray.filter(ImageFilter.FIND_EDGES)
    edge_pixels = list(edge.getdata())
    edge_mean = ImageStat.Stat(edge).mean[0] / 255.0
    border_edge_mean = (
        sum(edge_pixels[index] for index in border_indices)
        / len(border_indices)
        / 255.0
    )
    border_rgb_mean = tuple(
        sum(rgb_pixels[index][channel] for index in border_indices)
        / len(border_indices)
        / 255.0
        for channel in range(3)
    )
    border_color_std = (
        sum(
            sum(
                (rgb_pixels[index][channel] / 255.0 - border_rgb_mean[channel]) ** 2
                for channel in range(3)
            )
            / 3.0
            for index in border_indices
        )
        / len(border_indices)
    ) ** 0.5

    whiteness_score = clamp((border_mean - 0.78) / 0.18)
    smoothness_score = clamp(1.0 - border_std / 0.08)
    border_edge_score = clamp(1.0 - border_edge_mean / 0.08)
    color_plainness_score = clamp(1.0 - border_color_std / 0.08)
    background_score = (
        whiteness_score * smoothness_score * border_edge_score * color_plainness_score
    )
    border_texture_penalty = max(
        clamp((border_std - 0.08) / 0.12),
        clamp((border_edge_mean - 0.08) / 0.08),
        clamp((border_color_std - 0.08) / 0.12),
        clamp((0.78 - border_mean) / 0.30),
    )

    dark_fraction = sum(1 for value in pixels if value < 55) / len(pixels)
    outline_score = triangular_score(dark_fraction, target=0.18, width=0.20)

    edge_score = triangular_score(edge_mean, target=0.075, width=0.070)
    clutter_penalty = clamp((edge_mean - 0.16) / 0.14)

    foreground_mask = [
        abs(value - border_mean_raw) > 34 or value < 80 for value in pixels
    ]
    mask_image = Image.new("L", (width, height))
    mask_image.putdata([255 if value else 0 for value in foreground_mask])
    dilated_mask = mask_image.filter(ImageFilter.MaxFilter(15))
    component_mask = [value > 0 for value in dilated_mask.getdata()]
    components = connected_components(
        component_mask, width, height, min_size=int(width * height * 0.004)
    )

    if components:
        largest_component = max(components, key=len)
        component_pixels = set(largest_component)
        xs = [index % width for index in largest_component]
        ys = [index // width for index in largest_component]
        total_component_area = sum(len(component) for component in components)
        area_fraction = len(largest_component) / (width * height)
        component_dominance = (
            len(largest_component) / total_component_area
            if total_component_area
            else 0.0
        )
        component_count = len(components)
        center_x = (min(xs) + max(xs)) / 2 / width
        center_y = (min(ys) + max(ys)) / 2 / height
        center_score = triangular_score(center_x, 0.5, 0.25) * triangular_score(
            center_y, 0.5, 0.25
        )
        area_score = triangular_score(area_fraction, 0.36, 0.30)

        repeated_object_penalty = clamp((component_count - 1) / 4.0) * clamp(
            1.0 - component_dominance
        )
        single_object_score = clamp((component_dominance - 0.62) / 0.30) * clamp(
            1.0 - (component_count - 1) / 5.0
        )

        rim_source = Image.new("L", (width, height))
        rim_source.putdata(
            [255 if index in component_pixels else 0 for index in range(width * height)]
        )
        rim_outer = rim_source.filter(ImageFilter.MaxFilter(11))
        rim_inner = rim_source.filter(ImageFilter.MaxFilter(3))
        outer_pixels = list(rim_outer.getdata())
        inner_pixels = list(rim_inner.getdata())
        rim_indices = [
            index
            for index, value in enumerate(outer_pixels)
            if value > 0 and inner_pixels[index] == 0
        ]
        if rim_indices:
            white_rim_fraction = sum(
                1 for index in rim_indices if pixels[index] > 220
            ) / len(rim_indices)
            rim_score = clamp((white_rim_fraction - 0.55) / 0.35)
        else:
            white_rim_fraction = 0.0
            rim_score = 0.0
    else:
        area_fraction = 0.0
        component_count = 0
        component_dominance = 0.0
        center_score = 0.0
        area_score = 0.0
        repeated_object_penalty = 1.0
        single_object_score = 0.0
        white_rim_fraction = 0.0
        rim_score = 0.0

    score = (
        0.26 * background_score
        + 0.15 * outline_score
        + 0.12 * edge_score
        + 0.15 * center_score
        + 0.08 * area_score
        + 0.14 * single_object_score
        + 0.10 * rim_score
        - 0.22 * border_texture_penalty
        - 0.18 * repeated_object_penalty
        - 0.16 * clutter_penalty
    )
    return {
        "score": clamp(score),
        "background": background_score,
        "outline": outline_score,
        "edge": edge_score,
        "center": center_score,
        "area": area_score,
        "single_object": single_object_score,
        "rim": rim_score,
        "area_fraction": area_fraction,
        "component_count": float(component_count),
        "component_dominance": component_dominance,
        "dark_fraction": dark_fraction,
        "edge_mean": edge_mean,
        "border_edge_mean": border_edge_mean,
        "border_std": border_std,
        "border_color_std": border_color_std,
        "white_rim_fraction": white_rim_fraction,
        "border_texture_penalty": border_texture_penalty,
        "repeated_object_penalty": repeated_object_penalty,
        "clutter_penalty": clutter_penalty,
    }


def post_prompt(workflow: dict[str, Any], api_url: str) -> str:
    payload = json.dumps({"prompt": workflow}).encode("utf-8")
    request = urllib.request.Request(
        f"{api_url.rstrip('/')}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))["prompt_id"]


def wait_for_output(prompt_id: str, api_url: str, timeout: float) -> Path:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with urllib.request.urlopen(
            f"{api_url.rstrip('/')}/history/{prompt_id}", timeout=20
        ) as response:
            history = json.loads(response.read().decode("utf-8"))
        item = history.get(prompt_id)
        if item:
            status = item.get("status", {})
            if (
                not status.get("status_str") == "success"
                and status.get("completed") is False
            ):
                messages = status.get("messages") or []
                raise RuntimeError(f"Comfy prompt {prompt_id} failed: {messages}")
            if item.get("outputs") and "7" not in item["outputs"]:
                raise RuntimeError(
                    f"Comfy prompt {prompt_id} completed without SaveImage output"
                )
        if item and item.get("status", {}).get("completed"):
            image = item["outputs"]["7"]["images"][0]
            return COMFY_OUTPUT / image["filename"]
        time.sleep(1.5)
    raise TimeoutError(f"Timed out waiting for Comfy prompt {prompt_id}")


def render_workflow(
    base_workflow: dict[str, Any],
    candidate: StickerCandidate,
    subject: str,
    subject_index: int,
    prefix: str,
) -> dict[str, Any]:
    workflow = json.loads(json.dumps(base_workflow))
    workflow["8"]["inputs"]["lora_name"] = candidate.lora_name
    workflow["8"]["inputs"]["strength_model"] = candidate.strength_model
    workflow["8"]["inputs"]["strength_clip"] = candidate.strength_clip
    workflow["2"]["inputs"]["text"] = PROMPT_TEMPLATES[
        candidate.prompt_template_id
    ].format(subject=subject)
    workflow["3"]["inputs"]["text"] = NEGATIVE_PROMPTS[candidate.negative_prompt_id]
    workflow["5"]["inputs"]["seed"] = 880_000 + candidate.seed_offset + subject_index
    workflow["5"]["inputs"]["steps"] = candidate.steps
    workflow["5"]["inputs"]["cfg"] = candidate.cfg
    workflow["5"]["inputs"]["sampler_name"] = candidate.sampler_name
    workflow["5"]["inputs"]["scheduler"] = candidate.scheduler
    workflow["7"]["inputs"]["filename_prefix"] = prefix
    return workflow


def evaluate_candidate(
    candidate: StickerCandidate,
    subjects: tuple[str, ...],
    api_url: str,
    output_dir: Path,
    prompt_timeout: float,
) -> dict[str, Any]:
    base_workflow = json.loads(WORKFLOW_API.read_text(encoding="utf-8"))
    subject_scores = []
    image_paths = {}
    for index, subject in enumerate(subjects):
        prefix = f"evo_{candidate.candidate_id}_{subject.replace(' ', '_')}"
        workflow = render_workflow(base_workflow, candidate, subject, index, prefix)
        prompt_id = post_prompt(workflow, api_url)
        image_path = wait_for_output(prompt_id, api_url, prompt_timeout)
        metrics = image_proxy_metrics(image_path)
        subject_scores.append(
            {
                "subject": subject,
                "prompt_id": prompt_id,
                "image": str(image_path),
                **metrics,
            }
        )
        image_paths[subject] = image_path
        print(
            f"  {candidate.candidate_id} {subject}: score={metrics['score']:.3f} "
            f"bg={metrics['background']:.2f} edge={metrics['edge']:.2f} center={metrics['center']:.2f}"
        )

    avg_score = sum(item["score"] for item in subject_scores) / len(subject_scores)
    sheet = write_sheet(output_dir / f"{candidate.candidate_id}_sheet.png", image_paths)
    return {
        "candidate": asdict(candidate),
        "candidate_key": candidate.key(),
        "score": avg_score,
        "subjects": subject_scores,
        "sheet": str(sheet),
    }


def write_sheet(path: Path, image_paths: dict[str, Path]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    thumb_w = 220
    thumb_h = 250
    cols = 3
    rows = (len(image_paths) + cols - 1) // cols
    canvas = Image.new("RGB", (thumb_w * cols, thumb_h * rows), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (subject, image_path) in enumerate(image_paths.items()):
        image = Image.open(image_path).convert("RGB").resize((220, 220))
        x_pos = (idx % cols) * thumb_w
        y_pos = (idx // cols) * thumb_h
        draw.text((x_pos + 8, y_pos + 8), subject, fill=(0, 0, 0))
        canvas.paste(image, (x_pos, y_pos + 28))
    canvas.save(path)
    return path


def initial_population(
    rng: random.Random, population_size: int
) -> list[StickerCandidate]:
    seeds = [
        StickerCandidate(
            "g0_c00",
            "cxsticker_v1.safetensors",
            0.35,
            0.30,
            8.0,
            32,
            0,
            0,
            "euler",
            "normal",
            1000,
        ),
        StickerCandidate(
            "g0_c01",
            "cxsticker_v1.safetensors",
            0.45,
            0.35,
            7.5,
            28,
            1,
            0,
            "euler",
            "normal",
            2000,
        ),
        StickerCandidate(
            "g0_c02",
            "cxsticker_v2.safetensors",
            0.55,
            0.45,
            8.0,
            32,
            0,
            0,
            "euler",
            "normal",
            3000,
        ),
        StickerCandidate(
            "g0_c03",
            "cxsticker_v2.safetensors",
            0.45,
            0.35,
            7.0,
            28,
            2,
            1,
            "euler",
            "normal",
            4000,
        ),
    ]
    while len(seeds) < population_size:
        seeds.append(random_candidate(rng, f"g0_c{len(seeds):02d}"))
    return seeds[:population_size]


def random_candidate(rng: random.Random, candidate_id: str) -> StickerCandidate:
    return StickerCandidate(
        candidate_id=candidate_id,
        lora_name=rng.choice(["cxsticker_v1.safetensors", "cxsticker_v2.safetensors"]),
        strength_model=round(rng.uniform(0.25, 0.75), 2),
        strength_clip=round(rng.uniform(0.20, 0.60), 2),
        cfg=round(rng.uniform(6.0, 9.0), 1),
        steps=rng.choice([20, 24, 28, 32, 36]),
        prompt_template_id=rng.randrange(len(PROMPT_TEMPLATES)),
        negative_prompt_id=rng.randrange(len(NEGATIVE_PROMPTS)),
        sampler_name="euler",
        scheduler=rng.choice(["normal", "karras"]),
        seed_offset=rng.randrange(10_000, 999_999),
    )


def mutate_candidate(
    rng: random.Random, parent: StickerCandidate, candidate_id: str
) -> StickerCandidate:
    data = asdict(parent)
    data["candidate_id"] = candidate_id
    data["strength_model"] = round(
        clamp(data["strength_model"] + rng.uniform(-0.12, 0.12), 0.15, 0.85), 2
    )
    data["strength_clip"] = round(
        clamp(data["strength_clip"] + rng.uniform(-0.10, 0.10), 0.10, 0.70), 2
    )
    data["cfg"] = round(clamp(data["cfg"] + rng.uniform(-0.8, 0.8), 4.5, 10.5), 1)
    if rng.random() < 0.35:
        data["steps"] = rng.choice([20, 24, 28, 32, 36])
    if rng.random() < 0.30:
        data["prompt_template_id"] = rng.randrange(len(PROMPT_TEMPLATES))
    if rng.random() < 0.25:
        data["negative_prompt_id"] = rng.randrange(len(NEGATIVE_PROMPTS))
    if rng.random() < 0.20:
        data["scheduler"] = rng.choice(["normal", "karras"])
    if rng.random() < 0.15:
        data["lora_name"] = rng.choice(
            ["cxsticker_v1.safetensors", "cxsticker_v2.safetensors"]
        )
    data["seed_offset"] = rng.randrange(10_000, 999_999)
    return StickerCandidate(**data)


def load_completed_results(jsonl_path: Path) -> dict[str, dict[str, Any]]:
    """Load completed candidate evaluations keyed by candidate id."""
    completed = {}
    if not jsonl_path.exists():
        return completed

    with jsonl_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                result = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL row {line_number} in {jsonl_path}"
                ) from exc
            candidate_id = result.get("candidate", {}).get("candidate_id")
            if not candidate_id:
                raise ValueError(
                    f"Missing candidate_id in JSONL row {line_number} of {jsonl_path}"
                )
            completed[candidate_id] = result
    return completed


def resolve_subjects(benchmark: str, split: str, count: int) -> tuple[str, ...]:
    """Return a fixed benchmark subject slice.

    COCO is used here as object-name supervision only. The runner does not
    download COCO images; it uses the class names as held-out generation prompts.
    """
    if benchmark not in BENCHMARK_SPLITS:
        raise ValueError(
            f"Unknown benchmark {benchmark!r}. Choose from {sorted(BENCHMARK_SUBJECTS)}"
        )
    if split not in BENCHMARK_SPLITS[benchmark]:
        raise ValueError(f"Unknown split {split!r} for benchmark {benchmark!r}")
    subjects = BENCHMARK_SPLITS[benchmark][split]
    if count < 1:
        raise ValueError("--subjects must be at least 1")
    if count > len(subjects):
        raise ValueError(
            f"--subjects={count} exceeds {benchmark}/{split} size {len(subjects)}. "
            f"Use --subjects {len(subjects)} or a smaller value."
        )
    return subjects[:count]


def run_evolution(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    subjects = resolve_subjects(args.benchmark, args.split, args.subjects)
    run_dir = (
        args.output_dir
        / f"run_{args.benchmark}_{args.split}_seed_{args.seed}_p{args.population}_g{args.generations}_s{len(subjects)}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = run_dir / "candidate_evaluations.jsonl"

    population = initial_population(rng, args.population)
    completed_results = load_completed_results(jsonl_path) if args.resume else {}
    if completed_results:
        print(f"resuming from {len(completed_results)} completed candidate evaluations")
    all_results = list(completed_results.values())
    best: dict[str, Any] | None = max(
        completed_results.values(), key=lambda item: item["score"], default=None
    )
    for generation in range(args.generations):
        print(f"generation {generation}")
        results = []
        for idx, candidate in enumerate(population):
            candidate = StickerCandidate(
                **{**asdict(candidate), "candidate_id": f"g{generation}_c{idx:02d}"}
            )
            if candidate.candidate_id in completed_results:
                result = completed_results[candidate.candidate_id]
                results.append(result)
                print(
                    f"  {candidate.candidate_id}: resumed score={result['score']:.3f}"
                )
                continue

            result = evaluate_candidate(
                candidate, subjects, args.api_url, run_dir, args.prompt_timeout
            )
            results.append(result)
            all_results.append(result)
            completed_results[candidate.candidate_id] = result
            with jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result) + "\n")
            if best is None or result["score"] > best["score"]:
                best = result
                (run_dir / "best.json").write_text(
                    json.dumps(best, indent=2) + "\n", encoding="utf-8"
                )
                print(f"  new best: {best['score']:.3f} {candidate.candidate_id}")

        ranked = sorted(results, key=lambda item: item["score"], reverse=True)
        elites = [
            StickerCandidate(**item["candidate"])
            for item in ranked[: max(2, args.population // 3)]
        ]
        next_population = elites[:]
        while len(next_population) < args.population:
            if rng.random() < args.random_immigrant_rate:
                next_population.append(
                    random_candidate(
                        rng, f"g{generation + 1}_c{len(next_population):02d}"
                    )
                )
            else:
                parent = rng.choice(elites)
                next_population.append(
                    mutate_candidate(
                        rng, parent, f"g{generation + 1}_c{len(next_population):02d}"
                    )
                )
        population = next_population[: args.population]

    assert best is not None
    report = {
        "seed": args.seed,
        "benchmark": args.benchmark,
        "split": args.split,
        "population": args.population,
        "generations": args.generations,
        "subjects": subjects,
        "best": best,
        "run_dir": str(run_dir),
        "evaluations": len(all_results),
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def candidate_from_json(
    path: Path, candidate_id: str = "fixed_c00"
) -> StickerCandidate:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "best" in data:
        candidate_data = data["best"]["candidate"]
    elif "candidate" in data:
        candidate_data = data["candidate"]
    else:
        candidate_data = data
    candidate_data = {**candidate_data, "candidate_id": candidate_id}
    return StickerCandidate(**candidate_data)


def run_fixed_candidate(args: argparse.Namespace) -> dict[str, Any]:
    assert args.candidate_json is not None
    subjects = resolve_subjects(args.benchmark, args.split, args.subjects)
    run_dir = (
        args.output_dir
        / f"eval_{args.benchmark}_{args.split}_seed_{args.seed}_s{len(subjects)}_{args.candidate_json.stem}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    candidate = candidate_from_json(args.candidate_json)
    result = evaluate_candidate(
        candidate, subjects, args.api_url, run_dir, args.prompt_timeout
    )
    report = {
        "seed": args.seed,
        "benchmark": args.benchmark,
        "split": args.split,
        "subjects": subjects,
        "source_candidate_json": str(args.candidate_json),
        "result": result,
        "run_dir": str(run_dir),
        "evaluations": 1,
    }
    (run_dir / "candidate_evaluation.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default=API_URL)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACT_ROOT / "evolution")
    parser.add_argument(
        "--benchmark", choices=sorted(BENCHMARK_SUBJECTS), default="toy"
    )
    parser.add_argument(
        "--split", choices=("train", "dev", "test", "all"), default="all"
    )
    parser.add_argument(
        "--candidate-json",
        type=Path,
        help="Evaluate a fixed candidate from best.json/report.json instead of evolving a population.",
    )
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--subjects", type=int, default=4)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--random-immigrant-rate", type=float, default=0.25)
    parser.add_argument(
        "--prompt-timeout",
        type=float,
        default=600.0,
        help="Seconds to wait for each Comfy prompt before failing the candidate.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed candidates from candidate_evaluations.jsonl in the run directory.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_fixed_candidate(args) if args.candidate_json else run_evolution(args)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
