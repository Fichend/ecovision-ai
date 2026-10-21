"""Train and evaluate a MobileNetV2 TrashNet classifier."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


class ManifestDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        class_to_index: dict[str, int],
        transform: transforms.Compose,
    ) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.class_to_index = class_to_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.manifest.iloc[index]
        with Image.open(row["path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.class_to_index[row["label"]]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    normalization = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
            transforms.ToTensor(),
            normalization,
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalization,
        ]
    )
    return train_transform, evaluation_transform


def create_loaders(
    manifest: pd.DataFrame,
    class_to_index: dict[str, int],
    batch_size: int,
    image_size: int,
    workers: int,
) -> dict[str, DataLoader]:
    train_transform, evaluation_transform = build_transforms(image_size)
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "validation", "test"):
        split_frame = manifest.loc[manifest["split"] == split]
        dataset = ManifestDataset(
            split_frame,
            class_to_index,
            train_transform if split == "train" else evaluation_transform,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, object]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    targets: list[int] = []
    predictions: list[int] = []
    started = time.perf_counter()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())

    return {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy_score(targets, predictions),
        "macro_f1": f1_score(targets, predictions, average="macro", zero_division=0),
        "macro_precision": precision_score(
            targets, predictions, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            targets, predictions, average="macro", zero_division=0
        ),
        "seconds": time.perf_counter() - started,
        "targets": targets,
        "predictions": predictions,
    }


def compact(metrics: dict[str, object]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if key not in {"targets", "predictions"}
    }


def plot_confusion(matrix: np.ndarray, class_names: list[str], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Greens")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="TrashNet test confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_history(history: list[dict[str, object]], output: Path) -> None:
    epochs = [int(record["epoch"]) for record in history]
    train_loss = [float(record["train"]["loss"]) for record in history]
    validation_loss = [float(record["validation"]["loss"]) for record in history]
    train_f1 = [float(record["train"]["macro_f1"]) for record in history]
    validation_f1 = [float(record["validation"]["macro_f1"]) for record in history]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, train_loss, marker="o", label="Train")
    axes[0].plot(epochs, validation_loss, marker="o", label="Validation")
    axes[0].set(title="Cross-entropy loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(epochs, train_f1, marker="o", label="Train")
    axes[1].plot(epochs, validation_f1, marker="o", label="Validation")
    axes[1].set(title="Macro-F1", xlabel="Epoch", ylabel="Score", ylim=(0, 1))
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.suptitle("MobileNetV2 TrashNet training history")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    required_columns = {"path", "label", "split", "sha256"}
    if not required_columns.issubset(manifest.columns):
        raise ValueError(f"Manifest must contain {sorted(required_columns)}")
    if set(manifest["split"]) != {"train", "validation", "test"}:
        raise ValueError("Manifest must contain train, validation, and test splits")

    class_names = sorted(manifest["label"].unique().tolist())
    class_to_index = {name: index for index, name in enumerate(class_names)}
    device = choose_device(args.device)
    loaders = create_loaders(
        manifest,
        class_to_index,
        args.batch_size,
        args.image_size,
        args.workers,
    )

    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
    model.features.requires_grad_(False)
    model = model.to(device)

    train_labels = manifest.loc[manifest["split"] == "train", "label"]
    counts = Counter(train_labels)
    class_weights = torch.tensor(
        [len(train_labels) / (len(class_names) * counts[name]) for name in class_names],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )

    history: list[dict[str, object]] = []
    best_score = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    print(f"device={device} samples={len(manifest)} classes={class_names}", flush=True)

    for epoch in range(args.epochs):
        if epoch == args.freeze_epochs:
            model.features.requires_grad_(True)
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.features.parameters(), "lr": args.learning_rate * 0.1},
                    {"params": model.classifier.parameters(), "lr": args.learning_rate},
                ],
                weight_decay=1e-4,
            )
            print("unfroze MobileNetV2 feature extractor", flush=True)

        train_metrics = run_epoch(model, loaders["train"], criterion, device, optimizer)
        validation_metrics = run_epoch(
            model, loaders["validation"], criterion, device, optimizer=None
        )
        record = {
            "epoch": epoch + 1,
            "train": compact(train_metrics),
            "validation": compact(validation_metrics),
        }
        history.append(record)
        print(
            f"epoch={epoch + 1}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_f1={train_metrics['macro_f1']:.4f} "
            f"val_loss={validation_metrics['loss']:.4f} "
            f"val_f1={validation_metrics['macro_f1']:.4f} "
            f"seconds={train_metrics['seconds'] + validation_metrics['seconds']:.1f}",
            flush=True,
        )
        if float(validation_metrics["macro_f1"]) > best_score:
            best_score = float(validation_metrics["macro_f1"])
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, output / "best_model.pt")
        (output / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

    plot_history(history, output / "training_history.png")

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    test_metrics = run_epoch(model, loaders["test"], criterion, device, optimizer=None)
    targets = test_metrics.pop("targets")
    predictions = test_metrics.pop("predictions")
    labels = list(range(len(class_names)))
    matrix = confusion_matrix(targets, predictions, labels=labels)
    report = classification_report(
        targets,
        predictions,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    np.savetxt(output / "confusion_matrix.csv", matrix, fmt="%d", delimiter=",")
    plot_confusion(matrix, class_names, output / "confusion_matrix.png")
    (output / "classification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    metadata = {
        "architecture": "mobilenet_v2",
        "weights": str(weights),
        "model_version": "mobilenet_v2-trashnet-v1",
        "class_names": class_names,
        "class_to_index": class_to_index,
        "image_size": args.image_size,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "seed": args.seed,
        "epochs": args.epochs,
        "freeze_epochs": args.freeze_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "device": str(device),
        "best_validation_macro_f1": best_score,
        "test_metrics": compact(test_metrics),
        "dataset_split_counts": manifest.groupby(["split", "label"]).size().unstack(fill_value=0).to_dict(orient="index"),
    }
    (output / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        f"test_accuracy={test_metrics['accuracy']:.4f} "
        f"test_macro_f1={test_metrics['macro_f1']:.4f}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--freeze-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if not 0 <= args.freeze_epochs <= args.epochs:
        parser.error("--freeze-epochs must be between 0 and --epochs")
    return args


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
