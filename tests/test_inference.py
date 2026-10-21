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
