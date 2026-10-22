"""Train and evaluate a resource-conscious transfer-learning classifier."""

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
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V2_Weights,
    MobileNet_V3_Large_Weights,
    efficientnet_b0,
    mobilenet_v2,
    mobilenet_v3_large,
)


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
    logits_batches: list[torch.Tensor] = []
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
        logits_batches.append(logits.detach().cpu())

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
        "logits": torch.cat(logits_batches) if logits_batches else torch.empty((0, 0)),
    }


def compact(metrics: dict[str, object]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if key not in {"targets", "predictions", "logits"}
    }


def plot_confusion(matrix: np.ndarray, class_names: list[str], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(11, 9))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Greens")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="EcoVision 12-class test confusion matrix",
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
    figure.suptitle("EcoVision transfer-learning history")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def build_model(
    architecture: str, class_count: int
) -> tuple[nn.Module, object, nn.Module, nn.Module]:
    if architecture == "mobilenet_v2":
        weights = MobileNet_V2_Weights.DEFAULT
        model = mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.last_channel, class_count)
        return model, weights, model.features, model.classifier
    if architecture == "mobilenet_v3_large":
        weights = MobileNet_V3_Large_Weights.DEFAULT
        model = mobilenet_v3_large(weights=weights)
        final = model.classifier[-1]
        model.classifier[-1] = nn.Linear(final.in_features, class_count)
        return model, weights, model.features, model.classifier
    if architecture == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT
        model = efficientnet_b0(weights=weights)
        final = model.classifier[-1]
        model.classifier[-1] = nn.Linear(final.in_features, class_count)
        return model, weights, model.features, model.classifier
    raise ValueError(f"Unsupported architecture: {architecture}")


def fit_temperature(logits: torch.Tensor, targets: list[int]) -> float:
    if logits.numel() == 0:
        return 1.0
    log_temperature = torch.zeros((), requires_grad=True)
    target_tensor = torch.tensor(targets, dtype=torch.long)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.25, 8.0)
        loss = nn.functional.cross_entropy(logits / temperature, target_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.25, 8.0))


def choose_rejection_policy(
    logits: torch.Tensor,
    targets: list[int],
    temperature: float,
    minimum_accuracy: float = 0.85,
) -> dict[str, float]:
    probabilities = torch.softmax(logits / temperature, dim=1)
    values, indices = probabilities.topk(k=2, dim=1)
    correct = indices[:, 0].eq(torch.tensor(targets))
    best = {"confidence_threshold": 0.55, "margin_threshold": 0.10, "coverage": 0.0, "accepted_accuracy": 0.0}
    for confidence in np.arange(0.35, 0.86, 0.05):
        for margin in np.arange(0.05, 0.31, 0.05):
            accepted = (values[:, 0] >= confidence) & ((values[:, 0] - values[:, 1]) >= margin)
            count = int(accepted.sum())
            if not count:
                continue
            accuracy = float(correct[accepted].float().mean())
            coverage = count / len(targets)
            if accuracy >= minimum_accuracy and coverage > best["coverage"]:
                best = {
                    "confidence_threshold": round(float(confidence), 3),
                    "margin_threshold": round(float(margin), 3),
                    "coverage": coverage,
                    "accepted_accuracy": accuracy,
                }
    return best


def evaluate_rejection_policy(
    logits: torch.Tensor,
    targets: list[int],
    temperature: float,
    policy: dict[str, float],
) -> dict[str, float | int]:
    probabilities = torch.softmax(logits / temperature, dim=1)
    values, indices = probabilities.topk(k=2, dim=1)
    accepted = (
        (values[:, 0] >= policy["confidence_threshold"])
        & ((values[:, 0] - values[:, 1]) >= policy["margin_threshold"])
    )
    accepted_count = int(accepted.sum())
    correct = indices[:, 0].eq(torch.tensor(targets))
    return {
        "coverage": accepted_count / max(len(targets), 1),
        "accepted_accuracy": (
            float(correct[accepted].float().mean()) if accepted_count else 0.0
        ),
        "accepted_count": accepted_count,
        "rejected_count": len(targets) - accepted_count,
    }


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    torch.set_num_threads(args.cpu_threads)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    required_columns = {"path", "label", "split", "sha256"}
    if not required_columns.issubset(manifest.columns):
        raise ValueError(f"Manifest must contain {sorted(required_columns)}")
    if set(manifest["split"]) != {"train", "validation", "test"}:
        raise ValueError("Manifest must contain train, validation, and test splits")
    audit_manifest = manifest.drop(columns=["path"], errors="ignore").copy()
    if "original_path" in audit_manifest.columns:
        audit_manifest["original_path"] = audit_manifest["original_path"].map(
            lambda value: Path(str(value)).name
        )
    audit_manifest.to_csv(output / "dataset_manifest_audit.csv", index=False)

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

    model, weights, feature_extractor, classifier = build_model(
        args.architecture, len(class_names)
    )
    model = model.to(device)

    history: list[dict[str, object]] = []
    best_score = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    start_epoch = 0
    if args.resume and (output / "best_model.pt").exists() and (output / "history.json").exists():
        history = json.loads((output / "history.json").read_text(encoding="utf-8"))
        start_epoch = len(history)
        if start_epoch >= args.epochs:
            raise ValueError(
                f"Checkpoint already contains {start_epoch} epochs; --epochs is {args.epochs}"
            )
        state = torch.load(output / "best_model.pt", map_location=device, weights_only=True)
        model.load_state_dict(state)
        best_state = copy.deepcopy(model.state_dict())
        best_score = max(
            float(record["validation"]["macro_f1"]) for record in history
        )
        print(
            f"resuming_from_epoch={start_epoch + 1} best_validation_macro_f1={best_score:.4f}",
            flush=True,
        )

    feature_extractor.requires_grad_(start_epoch >= args.freeze_epochs)

    train_labels = manifest.loc[manifest["split"] == "train", "label"]
    counts = Counter(train_labels)
    class_weights = torch.tensor(
        [len(train_labels) / (len(class_names) * counts[name]) for name in class_names],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    if start_epoch >= args.freeze_epochs:
        optimizer = torch.optim.AdamW(
            [
                {"params": feature_extractor.parameters(), "lr": args.learning_rate * 0.1},
                {"params": classifier.parameters(), "lr": args.learning_rate},
            ],
            weight_decay=1e-4,
        )
    else:
        optimizer = torch.optim.AdamW(
            classifier.parameters(), lr=args.learning_rate, weight_decay=1e-4
        )

    epochs_without_improvement = 0
    print(f"device={device} samples={len(manifest)} classes={class_names}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        if epoch == args.freeze_epochs and start_epoch < args.freeze_epochs:
            feature_extractor.requires_grad_(True)
            optimizer = torch.optim.AdamW(
                [
                    {"params": feature_extractor.parameters(), "lr": args.learning_rate * 0.1},
                    {"params": classifier.parameters(), "lr": args.learning_rate},
                ],
                weight_decay=1e-4,
            )
            print(f"unfroze {args.architecture} feature extractor", flush=True)

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
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        (output / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        if epoch + 1 > args.freeze_epochs and epochs_without_improvement >= args.patience:
            print(f"early_stopping patience={args.patience}", flush=True)
            break

    plot_history(history, output / "training_history.png")

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    test_metrics = run_epoch(model, loaders["test"], criterion, device, optimizer=None)
    targets = test_metrics["targets"]
    predictions = test_metrics["predictions"]
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
    validation_final = run_epoch(
        model, loaders["validation"], criterion, device, optimizer=None
    )
    temperature = fit_temperature(
        validation_final["logits"], validation_final["targets"]
    )
    rejection_policy = choose_rejection_policy(
        validation_final["logits"], validation_final["targets"], temperature
    )
    rejection_test_metrics = evaluate_rejection_policy(
        test_metrics["logits"], targets, temperature, rejection_policy
    )
    metadata = {
        "architecture": args.architecture,
        "weights": str(weights),
        "model_version": f"{args.architecture}-ecovision-12class-v1",
        "class_names": class_names,
        "class_to_index": class_to_index,
        "image_size": args.image_size,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "early_stopped": len(history) < args.epochs,
        "freeze_epochs": args.freeze_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "temperature": temperature,
        "rejection_policy": rejection_policy,
        "rejection_test_metrics": rejection_test_metrics,
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
    parser.add_argument("--architecture", choices=("mobilenet_v2", "mobilenet_v3_large", "efficientnet_b0"), default="mobilenet_v3_large")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--freeze-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from best_model.pt and history.json in the output directory.",
    )
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
