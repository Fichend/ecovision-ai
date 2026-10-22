"""Download a bounded image sample from a public Hugging Face dataset."""

from __future__ import annotations

import argparse
import io
import json
import random
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "EcoVision-AI/1.0"})
    with urllib.request.urlopen(request, timeout=45) as response:
        return response.read()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=350)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    api_url = f"https://huggingface.co/api/datasets/{args.dataset}"
    metadata = json.loads(fetch(api_url))
    files = [
        item["rfilename"]
        for item in metadata.get("siblings", [])
        if Path(item["rfilename"]).suffix.lower() in IMAGE_SUFFIXES
    ]
    random.Random(args.seed).shuffle(files)
    files = files[: args.limit]
    destination_dir = args.output / args.label
    destination_dir.mkdir(parents=True, exist_ok=True)

    def process(index: int, filename: str) -> str:
        encoded = urllib.parse.quote(filename, safe="/")
        url = f"https://huggingface.co/datasets/{args.dataset}/resolve/main/{encoded}"
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(fetch(url)))).convert("RGB")
        image.thumbnail((384, 384), Image.Resampling.LANCZOS)
        destination = destination_dir / f"hf-{index:05d}-{Path(filename).stem}.jpg"
        image.save(destination, "JPEG", quality=88, optimize=True)
        return str(destination)

    failures: list[dict[str, str]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process, index, filename): filename
            for index, filename in enumerate(files)
        }
        for future in as_completed(futures):
            try:
                future.result()
                completed += 1
            except Exception as exc:
                failures.append({"file": futures[future], "error": str(exc)})
    report = {
        "dataset": args.dataset,
        "license": metadata.get("cardData", {}).get("license"),
        "label": args.label,
        "requested": len(files),
        "completed": completed,
        "failures": failures,
    }
    (args.output / f"{args.label}_fetch_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
