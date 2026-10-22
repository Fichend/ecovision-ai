from __future__ import annotations

from dataclasses import replace

from fastapi.testclient import TestClient

import ecovision_api.main as main_module
from ecovision_api.inference import InvalidImageError
from ecovision_api.main import app, get_model_service


class FakeModelService:
    model_version = "test-model"
    device = "cpu"

    @staticmethod
    def decode_image(content: bytes) -> object:
        if content != b"valid-image":
            raise InvalidImageError("invalid test image")
        return object()

    @staticmethod
    def classify(_image: object) -> dict[str, object]:
        return {
            "predicted_class": "paper",
            "top_predictions": [
                {"label": "paper", "probability": 0.8},
                {"label": "cardboard", "probability": 0.15},
                {"label": "plastic", "probability": 0.05},
            ],
            "confidence": 0.8,
            "confidence_level": "high",
            "uncertainty_message": "test uncertainty",
            "inference_time_ms": 12.0,
            "model_version": "test-model",
            "gradcam_available": False,
            "gradcam_data_url": None,
        }


def client() -> TestClient:
    app.dependency_overrides[get_model_service] = lambda: FakeModelService()
    return TestClient(app)


def test_health_reports_model_and_knowledge() -> None:
    with client() as test_client:
        response = test_client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["model_ready"] is True


def test_classify_validates_and_returns_contract() -> None:
    with client() as test_client:
        response = test_client.post(
            "/api/classify",
            files={"image": ("item.png", b"valid-image", "image/png")},
        )
    assert response.status_code == 200
    assert response.json()["predicted_class"] == "paper"
    assert len(response.json()["top_predictions"]) == 3


def test_classify_rejects_unsupported_mime_type() -> None:
    with client() as test_client:
        response = test_client.post(
            "/api/classify",
            files={"image": ("item.gif", b"gif", "image/gif")},
        )
    assert response.status_code == 415


def test_chat_uses_grounded_fallback_without_api_key(monkeypatch) -> None:
    monkeypatch.setattr(
        main_module,
        "settings",
        replace(main_module.settings, deepseek_api_key=None),
    )
    payload = {
        "conversation_history": [],
        "classification": {
            "predicted_class": "glass",
            "top_predictions": [
                {"label": "glass", "probability": 0.8},
                {"label": "plastic", "probability": 0.15},
                {"label": "metal", "probability": 0.05}
            ],
            "confidence": 0.8,
            "confidence_level": "high",
            "uncertainty_message": "Visual estimate only.",
            "model_version": "test-model"
        },
        "question": "What should I do?",
        "location": "hong_kong"
    }
    with client() as test_client:
        response = test_client.post("/api/chat", json=payload)
    assert response.status_code == 200
    assert response.json()["provider"] == "grounded_fallback"
    assert response.json()["sources"]


def test_chat_does_not_treat_rejected_guess_as_classification(monkeypatch) -> None:
    monkeypatch.setattr(
        main_module,
        "settings",
        replace(main_module.settings, deepseek_api_key=None),
    )
    payload = {
        "classification": {
            "predicted_class": "unknown",
            "raw_predicted_class": "battery",
            "top_predictions": [
                {"label": "battery", "probability": 0.39},
                {"label": "metal", "probability": 0.34},
                {"label": "general_waste", "probability": 0.27},
            ],
            "confidence": 0.39,
            "confidence_margin": 0.05,
            "confidence_level": "low",
            "accepted": False,
            "rejection_reason": "low_confidence_or_ambiguous",
            "uncertainty_message": "The result was rejected.",
            "model_version": "test-model",
        },
        "question": "What should I do?",
        "location": "hong_kong",
    }
    with client() as test_client:
        response = test_client.post("/api/chat", json=payload)
    assert response.status_code == 200
    answer = response.json()["response"]
    assert "could not identify" in answer
    assert "best visual match was battery" in answer
    assert "describe the material" in answer
