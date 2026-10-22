from __future__ import annotations

import base64
import io
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, mobilenet_v2, mobilenet_v3_large


class InvalidImageError(ValueError):
    pass


class ModelService:
    def __init__(self, model_path: Path, metadata_path: Path, device: str = "auto") -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {model_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Model metadata not found: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.class_names: list[str] = list(self.metadata["class_names"])
        self.model_version = str(self.metadata["model_version"])
        self.image_size = int(self.metadata["image_size"])
        self.temperature = float(self.metadata.get("temperature", 1.0))
        policy = self.metadata.get("rejection_policy", {})
        self.confidence_threshold = float(policy.get("confidence_threshold", 0.50))
        self.margin_threshold = float(policy.get("margin_threshold", 0.10))
        self.device = self._choose_device(device)
        architecture = str(self.metadata.get("architecture", "mobilenet_v2"))
        if architecture == "mobilenet_v2":
            self.model = mobilenet_v2(weights=None)
            self.model.classifier[1] = nn.Linear(self.model.last_channel, len(self.class_names))
        elif architecture == "mobilenet_v3_large":
            self.model = mobilenet_v3_large(weights=None)
            final = self.model.classifier[-1]
            self.model.classifier[-1] = nn.Linear(final.in_features, len(self.class_names))
        elif architecture == "efficientnet_b0":
            self.model = efficientnet_b0(weights=None)
            final = self.model.classifier[-1]
            self.model.classifier[-1] = nn.Linear(final.in_features, len(self.class_names))
        else:
            raise ValueError(f"Unsupported model architecture: {architecture}")
        state = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.metadata["normalization"]["mean"],
                    std=self.metadata["normalization"]["std"],
                ),
            ]
        )
        self._lock = threading.Lock()

    @staticmethod
    def _choose_device(requested: str) -> torch.device:
        if requested != "auto":
            return torch.device(requested)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @staticmethod
    def decode_image(content: bytes) -> Image.Image:
        try:
            with Image.open(io.BytesIO(content)) as candidate:
                candidate.verify()
            with Image.open(io.BytesIO(content)) as candidate:
                # Pillow reports multi-picture JPEGs from many phones as MPO.
                # The classifier expects one image, so use the primary frame.
                if candidate.format not in {"JPEG", "MPO", "PNG", "WEBP"}:
                    raise InvalidImageError(
                        f"Unsupported decoded image format: {candidate.format}"
                    )
                if candidate.format == "MPO":
                    candidate.seek(0)
                image = candidate.convert("RGB")
                if image.width < 16 or image.height < 16:
                    raise InvalidImageError("Image dimensions must be at least 16×16")
                if image.width * image.height > 40_000_000:
                    raise InvalidImageError("Image dimensions are too large")
                return image.copy()
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidImageError("The uploaded file is not a valid image") from exc

    @staticmethod
    def confidence_details(confidence: float) -> tuple[str, str]:
        if confidence >= 0.75:
            return "high", "The model is comparatively confident, but the result is not proof of material composition or recyclability."
        if confidence >= 0.50:
            return "medium", "The classes are not clearly separated. Check the item material and local recycling guidance before disposal."
        return "low", "Low-confidence result. The image may contain multiple materials or differ from the training data; do not rely on this prediction alone."

    @staticmethod
    def visual_quality_reason(image: Image.Image) -> str | None:
        """Reject blank or nearly uniform frames before trusting softmax scores."""
        luminance = np.asarray(image.convert("L").resize((64, 64)), dtype=np.float32)
        if float(luminance.std()) < 3.0:
            return "low_visual_information"
        return None

    def classify(self, image: Image.Image, include_gradcam: bool = True) -> dict[str, object]:
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        started = time.perf_counter()
        with self._lock:
            with torch.inference_mode():
                logits = self.model(tensor)
                probabilities = torch.softmax(logits / self.temperature, dim=1)[0].detach().cpu()
            top_probabilities, top_indices = torch.topk(probabilities, k=3)
            predicted_index = int(top_indices[0])
            gradcam = self._gradcam_data_url(image, tensor, predicted_index) if include_gradcam else None
        elapsed_ms = (time.perf_counter() - started) * 1000
        confidence = float(top_probabilities[0])
        margin = float(top_probabilities[0] - top_probabilities[1])
        quality_reason = self.visual_quality_reason(image)
        accepted = (
            quality_reason is None
            and confidence >= self.confidence_threshold
            and margin >= self.margin_threshold
        )
        confidence_level, uncertainty = self.confidence_details(confidence)
        raw_prediction = self.class_names[predicted_index]
        if not accepted:
            confidence_level = "low"
            uncertainty = (
                "The image contains too little visual detail for reliable recognition. "
                "Retake it with the item clearly visible."
                if quality_reason
                else "The model cannot identify this item reliably. Retake the photo with one "
                "clear item, better lighting, and a simpler background."
            )
        return {
            "predicted_class": raw_prediction if accepted else "unknown",
            "raw_predicted_class": raw_prediction,
            "top_predictions": [
                {
                    "label": self.class_names[int(index)],
                    "probability": float(probability),
                }
                for probability, index in zip(top_probabilities, top_indices)
            ],
            "confidence": confidence,
            "confidence_margin": margin,
            "confidence_level": confidence_level,
            "accepted": accepted,
            "rejection_reason": (
                None if accepted else quality_reason or "low_confidence_or_ambiguous"
            ),
            "uncertainty_message": uncertainty,
            "inference_time_ms": elapsed_ms,
            "model_version": self.model_version,
            "gradcam_available": gradcam is not None,
            "gradcam_data_url": gradcam,
        }

    def _gradcam_data_url(
        self, original: Image.Image, tensor: torch.Tensor, class_index: int
    ) -> str | None:
        activations: list[torch.Tensor] = []
        gradients: list[torch.Tensor] = []

        def capture(_module: nn.Module, _inputs: object, output: torch.Tensor) -> None:
            activations.append(output)
            output.register_hook(lambda gradient: gradients.append(gradient))

        handle = self.model.features[-1].register_forward_hook(capture)
        try:
            self.model.zero_grad(set_to_none=True)
            logits = self.model(tensor)
            logits[0, class_index].backward()
            if not activations or not gradients:
                return None
            activation = activations[0][0].detach()
            gradient = gradients[0][0].detach()
            weights = gradient.mean(dim=(1, 2), keepdim=True)
            heatmap = torch.relu((weights * activation).sum(dim=0))
            maximum = float(heatmap.max().detach().cpu())
            if maximum <= 0:
                return None
            heatmap = (heatmap / maximum).cpu().numpy()
            heatmap = cv2.resize(heatmap, original.size, interpolation=cv2.INTER_CUBIC)
            heatmap_rgb = cv2.cvtColor(
                cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET),
                cv2.COLOR_BGR2RGB,
            )
            base = np.asarray(original, dtype=np.float32)
            overlay = np.uint8(np.clip(base * 0.58 + heatmap_rgb * 0.42, 0, 255))
            buffer = io.BytesIO()
            Image.fromarray(overlay).save(buffer, format="PNG", optimize=True)
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            return f"data:image/png;base64,{encoded}"
        finally:
            handle.remove()
            self.model.zero_grad(set_to_none=True)
