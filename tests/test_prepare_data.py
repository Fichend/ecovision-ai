from pathlib import Path

import numpy as np
from PIL import Image

from ecovision_ml.prepare_data import CLASS_NAMES, deduplicate, prepare


def test_prepare_creates_stratified_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source"
    for class_index, label in enumerate(CLASS_NAMES):
        class_dir = source / label
        class_dir.mkdir(parents=True)
        for image_index in range(10):
            generator = np.random.default_rng(class_index * 100 + image_index)
            pixels = generator.integers(0, 256, size=(24, 24, 3), dtype=np.uint8)
            image = Image.fromarray(pixels, mode="RGB")
            image.save(class_dir / f"{label}{image_index}.jpg")

    manifest_path = prepare(source, tmp_path / "processed", seed=7)
    assert manifest_path.exists()

    import pandas as pd

    manifest = pd.read_csv(manifest_path)
    assert len(manifest) == 60
    assert set(manifest["split"]) == {"train", "validation", "test"}
    assert set(manifest["label"]) == set(CLASS_NAMES)
    assert manifest["sha256"].is_unique


def test_deduplicate_excludes_conflicting_labels() -> None:
    records = [
        {"path": "/a.jpg", "label": "glass", "sha256": "same", "width": 1, "height": 1},
        {"path": "/b.jpg", "label": "plastic", "sha256": "same", "width": 1, "height": 1},
        {"path": "/c.jpg", "label": "paper", "sha256": "unique", "width": 1, "height": 1},
    ]
    unique, duplicates, conflicts = deduplicate(records)
    assert unique["path"].tolist() == ["/c.jpg"]
    assert duplicates == []
    assert {item["path"] for item in conflicts} == {"/a.jpg", "/b.jpg"}
