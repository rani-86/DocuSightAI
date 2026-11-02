import random

import cv2
import numpy as np
from PIL import Image, ImageDraw

from docusight.preprocessing import estimate_skew, find_document_quad, preprocess, rotate
from docusight.synth import Gen, degrade


def _page():
    img = Image.new("L", (900, 1100), 255)
    d = ImageDraw.Draw(img)
    for i in range(25):
        d.rectangle([80, 80 + i * 38, 80 + 300 + (i * 37) % 450, 96 + i * 38], fill=0)
    return np.array(img)


def test_deskew_recovers_rotation():
    page = _page()
    skewed = rotate(page, 3.0)
    assert abs(estimate_skew(skewed) + 3.0) < 0.5


def test_perspective_quad_found_for_photo():
    gen = Gen(random.Random(1), "train")
    canvas, _, _ = gen.invoice(0, None)
    rng = random.Random(3)
    for _ in range(20):  # draw until the degrade picks the "photo on background" branch
        photo = degrade(canvas.img, rng, 1.0)
        if photo.size != canvas.img.size and photo.size[0] > canvas.img.size[0] * 1.1:
            break
    gray = cv2.cvtColor(np.array(photo), cv2.COLOR_RGB2GRAY)
    assert find_document_quad(gray) is not None


def test_preprocess_outputs_and_steps():
    res = preprocess(cv2.cvtColor(rotate(_page(), -2.0), cv2.COLOR_GRAY2BGR))
    assert res.image.ndim == 3 and res.binary.dtype == np.uint8
    assert any(s.startswith("deskew") for s in res.steps)
