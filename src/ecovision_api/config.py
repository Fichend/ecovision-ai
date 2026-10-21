from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def project_path(environment_name: str, default: str) -> Path:
    configured = Path(os.getenv(environment_name, default))
    return configured if configured.is_absolute() else PROJECT_ROOT / configured


@dataclass(frozen=True)
class Settings:
    model_path: Path = project_path(
        "ECOVISION_MODEL_PATH",
        "artifacts/mobilenet_v2_grouped_baseline/best_model.pt",
    )
    metadata_path: Path = project_path(
        "ECOVISION_MODEL_METADATA_PATH",
        "artifacts/mobilenet_v2_grouped_baseline/model_metadata.json",
    )
    knowledge_path: Path = project_path(
        "ECOVISION_KNOWLEDGE_PATH", "knowledge/recycling_guidance.json"
    )
    max_upload_bytes: int = int(os.getenv("ECOVISION_MAX_UPLOAD_MB", "10")) * 1024 * 1024
    deepseek_api_key: str | None = os.getenv("DEEPSEEK_API_KEY") or None
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")


settings = Settings()
