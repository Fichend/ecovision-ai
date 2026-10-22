"""Download a small, mapped subset of TACO and save labeled object crops."""

from __future__ import annotations

import argparse
import io
import json
import random
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageOps


CATEGORY_MAPPING = {
    "Plastic film": "plastic_bag_film",
    "Other plastic wrapper": "plastic_bag_film",
    "Single-use carrier bag": "plastic_bag_film",
    "Garbage bag": "plastic_bag_film",
    "Polypropylene bag": "plastic_bag_film",
    "Crisp packet": "plastic_bag_film",
    "Styrofoam piece": "foam_disposable_tableware",
    "Disposable plastic cup": "foam_disposable_tableware",
    "Paper cup": "foam_disposable_tableware",
    "Disposable food container": "foam_disposable_tableware",
    "Plastic utensils": "foam_disposable_tableware",
    "Foam food container": "foam_disposable_tableware",
    "Foam cup": "foam_disposable_tableware",
    "Meal carton": "foam_disposable_tableware",
}


def fetch(url: str, attempts: int = 4) -> bytes:
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "EcoVision-AI/1.0"})
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read()
        except Exception as exc:  # pragma: no cover - network retry path
            error = exc
            time.sleep(1 + attempt)
    raise RuntimeError(f"Could not download {url}: {error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--hf-metadata", type=Path)
    parser.add_argument(
        "--hf-base-url",
        default="https://huggingface.co/datasets/pictograph/taco/resolve/main/train",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-per-class", type=int, default=600)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.hf_metadata:
        fetch_huggingface_crops(args)
        return
    if not args.annotations:
        parser.error("Provide --annotations or --hf-metadata")

    data = json.loads(args.annotations.read_text(encoding="utf-8"))
    category_names = {item["id"]: item["name"] for item in data["categories"]}
    images = {item["id"]: item for item in data["images"]}
    candidates: dict[str, list[dict[str, object]]] = defaultdict(list)
    for annotation in data["annotations"]:
        label = CATEGORY_MAPPING.get(category_names[annotation["category_id"]])
        if label:
            candidates[label].append(annotation)

    rng = random.Random(args.seed)
    report: dict[str, object] = {"counts": {}, "failures": []}
    for label, annotations in candidates.items():
        rng.shuffle(annotations)
        count = 0
        for annotation in annotations:
            if count >= args.max_per_class:
                break
            image_info = images[annotation["image_id"]]
            url = image_info.get("flickr_640_url") or image_info.get("flickr_url")
            if not url:
                continue
            try:
                source = ImageOps.exif_transpose(Image.open(io.BytesIO(fetch(str(url))))).convert("RGB")
                source_width = float(image_info["width"])
                source_height = float(image_info["height"])
                x, y, width, height = [float(value) for value in annotation["bbox"]]
                scale_x = source.width / source_width
                scale_y = source.height / source_height
                padding_x = width * 0.15
                padding_y = height * 0.15
                left = max(0, int((x - padding_x) * scale_x))
                top = max(0, int((y - padding_y) * scale_y))
                right = min(source.width, int((x + width + padding_x) * scale_x))
                bottom = min(source.height, int((y + height + padding_y) * scale_y))
                if right - left < 24 or bottom - top < 24:
                    continue
                crop = source.crop((left, top, right, bottom))
                crop.thumbnail((384, 384), Image.Resampling.LANCZOS)
                destination = args.output / label / (
                    f"taco-i{annotation['image_id']:06d}-a{annotation['id']:06d}.jpg"
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                crop.save(destination, "JPEG", quality=88, optimize=True)
                count += 1
            except Exception as exc:  # keep the rest of the public set usable
                report["failures"].append({"annotation_id": annotation["id"], "error": str(exc)})
        report["counts"][label] = count
    (args.output / "taco_fetch_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["counts"], indent=2))


def fetch_huggingface_crops(args: argparse.Namespace) -> None:
    rows = [json.loads(line) for line in args.hf_metadata.read_text(encoding="utf-8").splitlines() if line]
    candidates: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        objects = row["objects"]
        for annotation_index, (bbox, category_name) in enumerate(
            zip(objects["bbox"], objects["category_names"])
        ):
            label = CATEGORY_MAPPING.get(category_name)
            if label:
                candidates[label].append(
                    {"file_name": row["file_name"], "bbox": bbox, "annotation_index": annotation_index}
                )
    rng = random.Random(args.seed)
    selected: list[tuple[str, dict[str, object]]] = []
    for label, annotations in candidates.items():
        rng.shuffle(annotations)
        selected.extend((label, item) for item in annotations[: args.max_per_class])

    by_file: dict[str, list[tuple[str, dict[str, object]]]] = defaultdict(list)
    for label, item in selected:
        by_file[str(item["file_name"])].append((label, item))

    counts: dict[str, int] = defaultdict(int)
    failures: list[dict[str, str]] = []

    def process(file_name: str, items: list[tuple[str, dict[str, object]]]) -> list[str]:
        content = fetch(f"{args.hf_base_url}/{file_name}")
        source = ImageOps.exif_transpose(Image.open(io.BytesIO(content))).convert("RGB")
        saved: list[str] = []
        image_token = Path(file_name).stem
        for label, item in items:
            x, y, width, height = [float(value) for value in item["bbox"]]
            padding_x = width * 0.15
            padding_y = height * 0.15
            left = max(0, int(x - padding_x))
            top = max(0, int(y - padding_y))
            right = min(source.width, int(x + width + padding_x))
            bottom = min(source.height, int(y + height + padding_y))
            if right - left < 24 or bottom - top < 24:
                continue
            crop = source.crop((left, top, right, bottom))
            crop.thumbnail((384, 384), Image.Resampling.LANCZOS)
            destination = args.output / label / (
                f"taco-i{image_token}-a{int(item['annotation_index']):03d}.jpg"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            crop.save(destination, "JPEG", quality=88, optimize=True)
            saved.append(label)
        return saved

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process, file_name, items): file_name
            for file_name, items in by_file.items()
        }
        for future in as_completed(futures):
            file_name = futures[future]
            try:
                for label in future.result():
                    counts[label] += 1
            except Exception as exc:
                failures.append({"file_name": file_name, "error": str(exc)})
    report = {"counts": dict(counts), "failures": failures}
    (args.output / "taco_fetch_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["counts"], indent=2))


if __name__ == "__main__":
    main()
