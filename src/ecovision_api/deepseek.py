from __future__ import annotations

import asyncio
import json

from openai import OpenAI

from .knowledge import Guidance
from .schemas import ChatRequest


SYSTEM_PROMPT = """You are EcoVision AI's recycling assistant. Respond in concise English.
The CNN prediction is immutable evidence from a visual classifier: never replace it or claim
that it proves material composition. Explain uncertainty plainly. Use only the supplied
knowledge excerpts for local disposal guidance and cite source titles inline. If the city or
material is insufficient, say what the user should verify. When accepted is false or the
predicted class is unknown, do not turn raw_predicted_class into a disposal conclusion.
Explain that it is only the rejected best visual match, ask for a clearer photo or material
description, and keep any interim advice general. Never invent a recycling rule."""


class DeepSeekService:
    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=45.0)
        self.model = model

    async def answer(self, request: ChatRequest, guidance: Guidance) -> str:
        evidence = {
            "classification": request.classification.model_dump(),
            "location": request.location,
            "city": request.city,
            "knowledge_version": guidance.knowledge_version,
            "knowledge_statements": guidance.statements,
            "sources": guidance.sources,
        }
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "system",
                "content": "Grounding evidence:\n" + json.dumps(evidence, ensure_ascii=False),
            },
        ]
        messages.extend(message.model_dump() for message in request.conversation_history[-8:])
        messages.append(
            {
                "role": "user",
                "content": request.question
                or "Explain the classification and give a safe recycling recommendation.",
            }
        )

        def complete() -> str:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=700,
            )
            return response.choices[0].message.content or "No response was generated."

        return await asyncio.to_thread(complete)


def fallback_answer(
    request: ChatRequest,
    guidance: Guidance,
    service_note: str = (
        "DeepSeek is not configured yet, so this response uses only the curated "
        "knowledge base."
    ),
) -> str:
    prediction = request.classification.predicted_class
    confidence = request.classification.confidence * 100
    statements = " ".join(guidance.statements)
    location_note = ""
    if request.location == "mainland_china" and not request.city:
        location_note = " Please provide your city because local collection rules differ."
    if not request.classification.accepted or prediction == "unknown":
        candidate = request.classification.raw_predicted_class
        candidate_note = (
            f" The best visual match was {candidate}, but it did not pass the acceptance threshold."
            if candidate
            else ""
        )
        return (
            f"The classifier could not identify this item reliably ({confidence:.1f}% raw "
            f"softmax confidence).{candidate_note} Please upload a clearer photo or describe "
            f"the material before using item-specific disposal advice. {statements}"
            f"{location_note} {service_note}"
        )
    return (
        f"The CNN predicts {prediction} with {confidence:.1f}% raw softmax confidence. "
        f"This is a visual estimate, not proof of the item's material. {statements}"
        f"{location_note} {service_note}"
    )
