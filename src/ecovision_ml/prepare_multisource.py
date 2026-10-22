"""Build a balanced 12-class manifest from public waste-image archives.

The preparation pipeline intentionally streams images from ZIP files, resizes them,
and writes only the selected training copies. It never expands a full source archive.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError

from .prepare_data import assign_splits, deduplicate, difference_hash, sha256_file


CLASS_NAMES = (
    "paper",
    "cardboard",
    "glass",
    "metal",
    "plastic_container",
    "plastic_bag_film",
    "food_organic",
    "battery",
    "electronic_waste",
    "textile",
    "foam_disposable_tableware",
    "general_waste",
)

HGI_MAPPING = {
    "applecore": "food_organic",
    "banana_peel": "food_organic",
    "battery": "battery",
    "battery_button": "battery",
    "button_battery": "battery",
    "book": "paper",
    "can": "metal",
    "capsule": "general_waste",
    "carton": "cardboard",
    "cigarette_butt": "general_waste",
    "cigarette_case": "general_waste",
    "glass_bottle": "glass",
    "lunch_box": "foam_disposable_tableware",
    "mask": "general_waste",
    "mercurial_thermometer": "general_waste",
    "mobilephone": "electronic_waste",
    "modulator_tube": "electronic_waste",
    "paper_cup": "foam_disposable_tableware",
    "pencil": "general_waste",
    "plastic_bottle": "plastic_container",
    "remote_control": "electronic_waste",
    "rice": "food_organic",
    "shoe": "textile",
    "t_shirt": "textile",
    "tea_leaf": "food_organic",
    "tin_can": "metal",
    "toothbrush": "general_waste",
    "trousers": "textile",
    "vegetable_leaf": "food_organic",
    "waste_paper": "paper",
    "watermelon_peel": "food_organic",
}

REALWASTE_MAPPING = {
    "cardboard": "cardboard",
    "food organics": "food_organic",
    "food_organics": "food_organic",
    "glass": "glass",
    "metal": "metal",
    "miscellaneous trash": "general_waste",
    "miscellaneous_trash": "general_waste",
    "paper": "paper",
    "plastic": "plastic_container",
    "textile trash": "textile",
    "textile_trash": "textile",
    "vegetation": "food_organic",
}

GARBAGE12_MAPPING = {
    "battery": "battery",
    "batteries": "battery",
    "biological": "food_organic",
    "brown_glass": "glass",
    "cardboard": "cardboard",
    "clothes": "textile",
    "green_glass": "glass",
    "metal": "metal",
    "paper": "paper",
    "plastic": "plastic_container",
    "shoes": "textile",
    "trash": "general_waste",
    "white_glass": "glass",
}

TRASHNET_MAPPING = {
    "paper": "paper",
    "cardboard": "cardboard",
    "glass": "glass",
    "metal": "metal",
    "plastic": "plastic_container",
    "trash": "general_waste",
}

CURATED_MAPPING = {name: name for name in CLASS_NAMES}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def normalized_token(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def infer_archive_label(name: str, mapping: dict[str, str]) -> str | None:
    parts = [normalized_token(part) for part in PurePosixPath(name).parts[:-1]]
    normalized_mapping = {normalized_token(key): value for key, value in mapping.items()}
    for part in reversed(parts):
        if part in normalized_mapping:
            return normalized_mapping[part]
    return None


def select_balanced_archive_entries(
    archive: zipfile.ZipFile,
    mapping: dict[str, str],
    maximum_per_class: int,
    seed: int,
) -> list[tuple[zipfile.ZipInfo, str]]:
    candidates: dict[str, list[zipfile.ZipInfo]] = defaultdict(list)
    for info in archive.infolist():
        if info.is_dir() or PurePosixPath(info.filename).suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = infer_archive_label(info.filename, mapping)
        if label:
            candidates[label].append(info)
    rng = random.Random(seed)
    selected: list[tuple[zipfile.ZipInfo, str]] = []
    for label in CLASS_NAMES:
        entries = candidates.get(label, [])
        rng.shuffle(entries)
        selected.extend((entry, label) for entry in entries[:maximum_per_class])
    return selected


def image_bytes_to_rgb(content: bytes) -> Image.Image:
    with Image.open(io.BytesIO(content)) as image:
        image.load()
        return ImageOps.exif_transpose(image).convert("RGB")


def save_training_copy(image: Image.Image, destination: Path, max_edge: int) -> None:
    image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, "JPEG", quality=88, optimize=True)


def curate_archive(
    archive_path: Path,
    source_name: str,
    mapping: dict[str, str],
    destination: Path,
    maximum_per_class: int,
    max_edge: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    records: list[dict[str, object]] = []
    rejected: list[dict[str, str]] = []
    with zipfile.ZipFile(archive_path) as archive:
        selected = select_balanced_archive_entries(
            archive, mapping, maximum_per_class, seed
        )
        for number, (info, label) in enumerate(selected):
            try:
                content = archive.read(info)
                image = image_bytes_to_rgb(content)
                digest = hashlib.sha256(content).hexdigest()
                output = destination / label / f"{source_name}-{digest[:16]}-{number:05d}.jpg"
                save_training_copy(image, output, max_edge)
                records.append(record_for_path(output, label, source_name, info.filename))
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                rejected.append({"source": info.filename, "reason": str(exc)})
    return records, rejected


def record_for_path(path: Path, label: str, source: str, original: str) -> dict[str, object]:
    with Image.open(path) as image:
        width, height = image.size
    return {
        "path": str(path.resolve()),
        "label": label,
        "sha256": sha256_file(path),
        "dhash": difference_hash(path),
        "width": width,
        "height": height,
        "source_dataset": source,
        "original_path": original,
        "source_group": original,
    }


def add_directory(
    source: Path,
    mapping: dict[str, str],
    source_name: str,
    maximum_per_class: int,
    seed: int,
) -> list[dict[str, object]]:
    by_label: dict[str, list[Path]] = defaultdict(list)
    normalized_mapping = {normalized_token(key): value for key, value in mapping.items()}
    for path in source.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = next(
            (
                normalized_mapping[token]
                for token in reversed([normalized_token(part) for part in path.parts[:-1]])
                if token in normalized_mapping
            ),
            None,
        )
        if label:
            by_label[label].append(path)
    rng = random.Random(seed)
    records: list[dict[str, object]] = []
    for label, paths in by_label.items():
        rng.shuffle(paths)
        for path in paths[:maximum_per_class]:
            detected_source = source_name
            if source_name == "additional_public_sources":
                if path.name.startswith("taco-i"):
                    detected_source = "taco"
                elif path.name.startswith("hf-") and label == "electronic_waste":
                    detected_source = "giz_ewaste_database"
            record = record_for_path(path, label, detected_source, str(path))
            if path.name.startswith("taco-i") and "-a" in path.stem:
                record["source_group"] = path.stem.split("-a", 1)[0]
            records.append(record)
    return records


def build_manifest(args: argparse.Namespace) -> Path:
    records: list[dict[str, object]] = []
    rejected: list[dict[str, str]] = []
    if args.hgi_zip:
        new_records, new_rejected = curate_archive(
            args.hgi_zip,
            "hgi30",
            HGI_MAPPING,
            args.curated_dir,
            args.max_per_source_class,
            args.max_edge,
            args.seed,
        )
        records.extend(new_records)
        rejected.extend(new_rejected)
    if args.realwaste_zip:
        new_records, new_rejected = curate_archive(
            args.realwaste_zip,
            "realwaste",
            REALWASTE_MAPPING,
            args.curated_dir,
            args.max_per_source_class,
            args.max_edge,
            args.seed + 1,
        )
        records.extend(new_records)
        rejected.extend(new_rejected)
    if args.garbage12_zip:
        new_records, new_rejected = curate_archive(
            args.garbage12_zip,
            "garbage12",
            GARBAGE12_MAPPING,
            args.curated_dir,
            args.max_per_source_class,
            args.max_edge,
            args.seed + 2,
        )
        records.extend(new_records)
        rejected.extend(new_rejected)
    if args.trashnet_dir:
        records.extend(
            add_directory(
                args.trashnet_dir,
                TRASHNET_MAPPING,
                "trashnet",
                args.max_per_source_class,
                args.seed + 3,
            )
        )
    if args.additional_dir:
        records.extend(
            add_directory(
                args.additional_dir,
                CURATED_MAPPING,
                "additional_public_sources",
                args.max_per_source_class,
                args.seed + 4,
            )
        )
    if not records:
        raise ValueError("No images were discovered from the configured sources")

    unique, duplicates, conflicts = deduplicate(records)
    missing = sorted(set(CLASS_NAMES) - set(unique["label"]))
    if missing:
        raise ValueError(f"The merged data is missing required classes: {missing}")
    manifest = assign_splits(unique, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False, quoting=csv.QUOTE_MINIMAL)
    summary = {
        "class_names": list(CLASS_NAMES),
        "records_before_deduplication": len(records),
        "records_after_deduplication": len(manifest),
        "exact_duplicates_removed": len(duplicates),
        "conflicting_duplicates_removed": len(conflicts),
        "rejected_images": rejected,
        "split_counts": manifest.groupby(["split", "label"]).size().unstack(fill_value=0).to_dict(orient="index"),
        "source_counts": manifest.groupby(["source_dataset", "label"]).size().unstack(fill_value=0).to_dict(orient="index"),
        "seed": args.seed,
        "temporary_images_may_be_deleted_after_training": True,
    }
    (args.output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({"manifest": str(manifest_path), "counts": Counter(manifest["label"])}, default=dict))
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hgi-zip", type=Path)
    parser.add_argument("--realwaste-zip", type=Path)
    parser.add_argument("--garbage12-zip", type=Path)
    parser.add_argument("--trashnet-dir", type=Path)
    parser.add_argument("--additional-dir", type=Path)
    parser.add_argument("--curated-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-per-source-class", type=int, default=1300)
    parser.add_argument("--max-edge", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    build_manifest(parse_args())


if __name__ == "__main__":
    main()
