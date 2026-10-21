"""Evaluate user-supplied real-world photos without mixing them into training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score

from ecovision_api.config import settings
from ecovision_api.inference import ModelService

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def evaluate(source: Path, output: Path) -> None:
    files = sorted(
        path for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not files:
        raise ValueError(f"No images found under {source}")
    model = ModelService(settings.model_path, settings.metadata_path)
    rows: list[dict[str, object]] = []
    for path in files:
        with Image.open(path) as image:
            result = model.classify(image.convert("RGB"), include_gradcam=False)
        expected = path.parent.name if path.parent.name in model.class_names else None
        rows.append(
            {
                "path": str(path.resolve()),
                "expected_label": expected,
                "predicted_label": result["predicted_class"],
                "confidence": result["confidence"],
                "confidence_level": result["confidence_level"],
                "correct": expected == result["predicted_class"] if expected else None,
            }
        )
        print(f"{path.name}: {result['predicted_class']} ({result['confidence']:.1%})", flush=True)

    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "external_predictions.csv", index=False)
    labelled = frame.dropna(subset=["expected_label"])
    summary: dict[str, object] = {
        "images": len(frame),
        "labelled_images": len(labelled),
        "model_version": model.model_version,
        "device": str(model.device),
    }
    if not labelled.empty:
        targets = labelled["expected_label"]
        predictions = labelled["predicted_label"]
        summary.update(
            {
                "accuracy": accuracy_score(targets, predictions),
                "macro_f1": f1_score(targets, predictions, average="macro", zero_division=0),
                "classification_report": classification_report(
                    targets, predictions, output_dict=True, zero_division=0
                ),
            }
        )
    (output / "external_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/external"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/external_evaluation"))
    args = parser.parse_args()
    evaluate(args.source, args.output)


if __name__ == "__main__":
    main()
