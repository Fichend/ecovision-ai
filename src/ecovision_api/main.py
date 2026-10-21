from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .deepseek import DeepSeekService, fallback_answer
from .inference import InvalidImageError, ModelService
from .knowledge import KnowledgeBase
from .schemas import (
    ChatRequest,
    ChatResponse,
    ClassificationResponse,
    HealthResponse,
    SourceReference,
)

app = FastAPI(
    title="EcoVision AI API",
    version="0.1.0",
    description="Waste classification, Grad-CAM, and grounded recycling guidance.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


@lru_cache(maxsize=1)
def get_model_service() -> ModelService:
    return ModelService(settings.model_path, settings.metadata_path)


@lru_cache(maxsize=1)
def get_knowledge_base() -> KnowledgeBase:
    return KnowledgeBase(settings.knowledge_path)


@app.get("/api/health", response_model=HealthResponse)
def health(
    model: ModelService = Depends(get_model_service),
    knowledge: KnowledgeBase = Depends(get_knowledge_base),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_ready=True,
        model_version=model.model_version,
        device=str(model.device),
        deepseek_configured=bool(settings.deepseek_api_key),
        knowledge_version=knowledge.version,
    )


@app.post("/api/classify", response_model=ClassificationResponse)
async def classify(
    image: UploadFile = File(...),
    model: ModelService = Depends(get_model_service),
) -> ClassificationResponse:
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Supported image types are JPEG, PNG, and WebP.",
        )
    content = await image.read(settings.max_upload_bytes + 1)
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="The image exceeds the upload limit.")
    try:
        decoded = model.decode_image(content)
        return ClassificationResponse.model_validate(model.classify(decoded))
    except InvalidImageError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    knowledge: KnowledgeBase = Depends(get_knowledge_base),
) -> ChatResponse:
    guidance = knowledge.retrieve(
        request.location,
        request.classification.predicted_class,
        request.city,
    )
    sources = [SourceReference.model_validate(source) for source in guidance.sources]
    if not settings.deepseek_api_key:
        return ChatResponse(
            response=fallback_answer(request, guidance),
            sources=sources,
            uncertainty_information=request.classification.uncertainty_message,
            provider="grounded_fallback",
        )
    try:
        service = DeepSeekService(
            settings.deepseek_api_key,
            settings.deepseek_base_url,
            settings.deepseek_model,
        )
        response = await service.answer(request, guidance)
    except Exception:
        return ChatResponse(
            response=fallback_answer(
                request,
                guidance,
                "DeepSeek is currently unavailable, so this response uses only the "
                "curated knowledge base.",
            ),
            sources=sources,
            uncertainty_information=request.classification.uncertainty_message,
            provider="grounded_fallback",
        )
    return ChatResponse(
        response=response,
        sources=sources,
        uncertainty_information=request.classification.uncertainty_message,
        provider="deepseek",
        model=settings.deepseek_model,
    )


def run() -> None:
    import uvicorn

    uvicorn.run("ecovision_api.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    run()
