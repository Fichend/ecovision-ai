"""Validate TrashNet images and create a reproducible stratified split."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image, UnidentifiedImageError

CLASS_NAMES = ("cardboard", "glass", "metal", "paper", "plastic", "trash")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_image(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            return rgb.size
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Invalid image: {path}") from exc


def difference_hash(path: Path) -> int:
    """Return a 64-bit perceptual hash used to keep near-duplicates together."""
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L").resize((9, 8)), dtype=np.uint8)
    differences = pixels[:, :-1] > pixels[:, 1:]
    value = 0
    for bit in differences.ravel():
        value = (value << 1) | int(bit)
    return value


def discover_images(source: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for label in CLASS_NAMES:
        class_dir = source / label
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                width, height = validate_image(path)
                records.append(
                    {
                        "path": str(path.resolve()),
                        "label": label,
                        "sha256": sha256_file(path),
                        "dhash": difference_hash(path),
                        "width": width,
                        "height": height,
                    }
                )
    if not records:
        raise ValueError(f"No supported images found under {source}")
    return records


def deduplicate(
    records: list[dict[str, object]],
) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    frame = pd.DataFrame.from_records(records)
    conflicting_counts = frame.groupby("sha256")["label"].nunique()
    conflicting_hashes = set(conflicting_counts[conflicting_counts > 1].index)
    conflict_mask = frame["sha256"].isin(conflicting_hashes)
    conflicts = frame.loc[conflict_mask].to_dict(orient="records")
    frame = frame.loc[~conflict_mask].copy()
    duplicate_mask = frame.duplicated(subset=["sha256"], keep="first")
    removed = frame.loc[duplicate_mask].to_dict(orient="records")
    return frame.loc[~duplicate_mask].reset_index(drop=True), removed, conflicts


def assign_similarity_groups(
    frame: pd.DataFrame, threshold: int = 4
) -> pd.DataFrame:
    """Group same-label images whose perceptual hashes are very close."""
    result = frame.copy()
    result["group_id"] = ""
    next_group = 0

    for label, label_frame in result.groupby("label"):
        indices = label_frame.index.tolist()
        parent = {index: index for index in indices}

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        hashes = {index: int(result.at[index, "dhash"]) for index in indices}
        for offset, left in enumerate(indices):
            for right in indices[offset + 1 :]:
                if (hashes[left] ^ hashes[right]).bit_count() <= threshold:
                    union(left, right)

        roots: dict[int, int] = {}
        for index in indices:
            root = find(index)
            if root not in roots:
                roots[root] = next_group
                next_group += 1
            result.at[index, "group_id"] = f"{label}-{roots[root]:04d}"
    return result


def assign_splits(
    frame: pd.DataFrame,
    seed: int = 42,
    test_fraction: float = 0.15,
    validation_fraction: float = 0.15,
) -> pd.DataFrame:
    if test_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("Validation and test fractions must be positive")
    if test_fraction + validation_fraction >= 1:
        raise ValueError("Validation and test fractions must sum to less than 1")

    grouped = assign_similarity_groups(frame)
    grouped["split"] = ""
    train_fraction = 1.0 - validation_fraction - test_fraction

    for class_offset, (label, label_frame) in enumerate(grouped.groupby("label")):
        groups = [group.index.tolist() for _, group in label_frame.groupby("group_id")]
        if len(groups) < 3:
            raise ValueError(f"Class {label} has fewer than three independent image groups")
        rng = random.Random(seed + class_offset)
        rng.shuffle(groups)
        groups.sort(key=len, reverse=True)
        total = len(label_frame)
        targets = {
            "train": total * train_fraction,
            "validation": total * validation_fraction,
            "test": total * test_fraction,
        }
        counts = {split: 0 for split in targets}
        for group_indices in groups:
            size = len(group_indices)

            def score(candidate: str) -> float:
                prospective = dict(counts)
                prospective[candidate] += size
                return sum(
                    ((prospective[split] - targets[split]) / max(targets[split], 1.0)) ** 2
                    for split in targets
                )

            chosen = min(targets, key=score)
            grouped.loc[group_indices, "split"] = chosen
            counts[chosen] += size

    if set(grouped["split"]) != {"train", "validation", "test"}:
        raise RuntimeError("Grouped split did not populate all required partitions")
    return grouped.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def prepare(source: Path, output: Path, seed: int = 42) -> Path:
    random.seed(seed)
    output.mkdir(parents=True, exist_ok=True)
    discovered = discover_images(source)
    unique, removed, conflicts = deduplicate(discovered)
    manifest = assign_splits(unique, seed=seed)
    manifest_path = output / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    counts = {
        split: dict(Counter(manifest.loc[manifest["split"] == split, "label"]))
        for split in ("train", "validation", "test")
    }
    summary = {
        "source": str(source.resolve()),
        "seed": seed,
        "discovered_images": len(discovered),
        "unique_images": len(unique),
        "exact_duplicates_removed": len(removed),
        "conflicting_images_removed": len(conflicts),
        "perceptual_hash": "64-bit dHash",
        "near_duplicate_hamming_threshold": 4,
        "similarity_groups": int(manifest["group_id"].nunique()),
        "class_names": list(CLASS_NAMES),
        "split_counts": counts,
        "duplicate_records": removed,
        "conflicting_records": conflicts,
    }
    (output / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = prepare(args.source, args.output, args.seed)
    print(f"Manifest written to {manifest}")


if __name__ == "__main__":
    main()
