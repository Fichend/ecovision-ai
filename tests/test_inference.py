from __future__ import annotations

import io

from PIL import Image

from ecovision_api.inference import ModelService


def test_decode_image_accepts_mpo_and_uses_primary_frame() -> None:
    buffer = io.BytesIO()
    primary = Image.new("RGB", (32, 24), "red")
    secondary = Image.new("RGB", (32, 24), "blue")
    primary.save(
        buffer,
        format="MPO",
        save_all=True,
        append_images=[secondary],
    )

    decoded = ModelService.decode_image(buffer.getvalue())

    assert decoded.mode == "RGB"
    assert decoded.size == (32, 24)
    red, green, blue = decoded.getpixel((0, 0))
    assert red > 200
    assert green < 40
    assert blue < 40


def test_visual_quality_rejects_uniform_frame() -> None:
    assert ModelService.visual_quality_reason(Image.new("RGB", (224, 224), "white")) == "low_visual_information"


def test_visual_quality_accepts_detailed_frame() -> None:
    image = Image.effect_noise((224, 224), 32).convert("RGB")
    assert ModelService.visual_quality_reason(image) is None
