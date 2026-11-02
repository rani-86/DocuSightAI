"""DIQA tests: estimators vs known ground truth, decision logic, thresholds loading, pipeline integration. No OCR."""
import json
import random

import cv2
import numpy as np
import pytest

from conftest import FakeEngine
from docusight import quality as Q
from docusight import synth
from docusight.classifier import KeywordDocClassifier
from docusight.observability import Metrics
from docusight.pipeline import DocumentPipeline
from docusight.preprocessing import denoise, preprocess


@pytest.fixture(scope="module")
def page():
    """A clean ruled-table page; paper at ~225 gray so noise is not clipped away."""
    gen = synth.Gen(random.Random(11), "calib")
    canvas, _, _ = gen.invoice(1, None)
    return np.array(canvas.img.convert("L"), np.float32) * 0.88


@pytest.fixture(scope="module")
def dense():
    gen = synth.Gen(random.Random(12), "calib")
    canvas, _, _ = synth.dense_text_page(gen, 15)
    return np.array(canvas.img.convert("L"), np.float32) * 0.88


def degrade(base, *ops, seed=0):
    return synth.apply_controlled(base, list(ops), seed)


# ---------------------------------------------------------------------------------- noise

@pytest.mark.parametrize("sigma", [4, 8, 14])
def test_noise_sigma_recovers_known_sigma(page, sigma):
    est = Q.noise_sigma(degrade(page, ("gauss_noise", {"sigma": sigma})))
    assert est.measurable
    assert abs(est.sigma - sigma) < 0.6
    assert abs(est.sigma_immerkaer - sigma) < 0.8


def test_clean_text_dense_page_is_not_noise(dense):
    """Text strokes must not be mistaken for noise, even when text covers most of the page."""
    est = Q.noise_sigma(degrade(dense))
    assert est.measurable and est.sigma < 0.5
    noisy = Q.noise_sigma(degrade(dense, ("gauss_noise", {"sigma": 6})))
    assert abs(noisy.sigma - 6) < 0.7


def test_noise_estimate_ignores_blur_and_jpeg(page):
    assert Q.noise_sigma(degrade(page, ("gauss_blur", {"sigma": 2.0}))).sigma < 0.5
    assert Q.noise_sigma(degrade(page, ("jpeg", {"quality": 30}))).sigma < 0.5


def test_saturated_page_is_reported_not_measurable():
    """Noise cannot be observed on a blown-out page: say 'unknown' instead of a confident 0."""
    img = np.full((400, 400), 255, np.uint8)
    assert not Q.noise_sigma(img).measurable


def test_salt_and_pepper_detected_and_not_counted_as_gaussian(page):
    img = degrade(page, ("salt_pepper", {"density": 0.01}))
    flat = Q.flat_tiles(img)
    frac, _ = Q.impulse_fraction(flat, img.shape)
    assert 0.005 < frac < 0.02  # tracks the injected density
    assert Q.noise_sigma(img, flat).sigma < 0.5
    clean_frac, _ = Q.impulse_fraction(Q.flat_tiles(degrade(page)), page.shape)
    assert clean_frac < 0.0005  # thin strokes and table rules are not impulses


def test_jpeg_blockiness_separates_heavy_compression(page):
    clean = Q.jpeg_blockiness(degrade(page))
    heavy = Q.jpeg_blockiness(degrade(page, ("jpeg", {"quality": 15})))
    assert clean < 1.08 and heavy > clean + 0.25


def test_periodic_noise_detected(page):
    base = Q.periodic_peak(degrade(page), Q.flat_tiles(degrade(page)))
    img = degrade(page, ("periodic", {"amplitude": 4, "period_px": 6.0, "angle_deg": 0.0}))
    peak = Q.periodic_peak(img, Q.flat_tiles(img))
    assert peak is not None and peak > 100 * max(base or 1.0, 1.0)


def test_suspected_noise_types_uses_thresholds(page):
    th = Q.Thresholds(hint_impulse=0.001, hint_jpeg=1.15, hint_periodic=1000.0)
    img = degrade(page, ("salt_pepper", {"density": 0.01}), ("gauss_noise", {"sigma": 5}))
    flat = Q.flat_tiles(img)
    n = Q.noise_sigma(img, flat)
    types = Q.suspected_noise_types(n.sigma, Q.noise_hints(img, flat), th)
    assert "salt_pepper" in types and "gaussian" in types
    assert Q.suspected_noise_types(0.0, Q.noise_hints(degrade(page)), th) == ["none"]


# ---------------------------------------------------------------------------------- blur

def test_blur_score_is_monotone_in_true_blur(page):
    ew = [Q.blur_metrics(degrade(page, ("gauss_blur", {"sigma": b}))).edge_width for b in (1.0, 1.5, 2.0, 3.0)]
    assert all(b > a for a, b in zip(ew, ew[1:])), ew


def test_blur_score_stays_monotone_under_noise_after_denoising(page):
    """The whole point of measuring after denoising: noise must not scramble the blur ordering."""
    ew = []
    for b in (1.0, 2.0, 3.0):
        img = degrade(page, ("gauss_blur", {"sigma": b}), ("gauss_noise", {"sigma": 12}))
        sig = Q.noise_sigma(img).sigma
        ew.append(Q.blur_metrics(denoise(img, sig)).edge_width)
    assert ew[0] < ew[1] < ew[2], ew


def test_legacy_laplacian_variance_is_inflated_by_noise(page):
    """Documents the failure that motivated the new estimator (and guards against 'fixing' it by accident)."""
    from docusight.preprocessing import blur_score

    blurry = degrade(page, ("gauss_blur", {"sigma": 3.0}))
    noisy_blurry = degrade(page, ("gauss_blur", {"sigma": 3.0}), ("gauss_noise", {"sigma": 16}))
    assert blur_score(noisy_blurry) > 100 * blur_score(blurry)


def test_text_height_tracks_resolution(page):
    h = [Q.text_height(degrade(page, ("lowres", {"scale": s}))).text_height_px for s in (1.0, 0.5, 0.35)]
    assert h[0] > h[1] > h[2]


# ---------------------------------------------------------------------------------- exposure

def test_exposure_detects_shadow_and_glare(page):
    even = Q.exposure_metrics(degrade(page))
    shadow = Q.exposure_metrics(degrade(page, ("shadow", {"strength": 0.6, "angle_deg": 0.0})))
    glare = Q.exposure_metrics(degrade(page, ("glare", {"cx": 0.5, "cy": 0.4, "radius_frac": 0.12, "strength": 2.0})))
    assert shadow.unevenness > even.unevenness + 0.5
    assert glare.glare_fraction > 0.1 and even.glare_fraction == 0.0
    assert glare.glare_edge_ratio is not None and glare.glare_edge_ratio < 0.3


# ---------------------------------------------------------------------------------- decision logic

def _meas(sigma=0.0, edge=0.8, impulse=0.0, glare=0.0, unev=1.05, paper=225.0, measurable=True):
    return (Q.NoiseEstimate(sigma, sigma, sigma, 0.5, 100, measurable), Q.NoiseHints(impulse, 1.0, None, 1000),
            Q.BlurMetrics(edge, 0, 0, 0, 0.05, 1000), Q.ExposureMetrics(200, 100, paper, 0.5, 0, unev, 0, glare, None))


TH = Q.Thresholds(noise_enhance=2.0, noise_retake=18.0, blur_enhance=1.1, blur_retake=2.2, text_height_min=7.0, glare_retake=0.05,
                  unevenness_retake=2.5, impulse_enhance=0.002, dark_paper_min=90.0)


def test_decision_accept_enhance_retake():
    assert Q.decide(*_meas(), 13.0, TH).decision == "accept"
    d = Q.decide(*_meas(sigma=6.0), 13.0, TH)
    assert d.decision == "enhance" and any("denoise" in r for r in d.reasons)
    d = Q.decide(*_meas(edge=1.6), 13.0, TH)
    assert d.decision == "enhance" and any("deblur" in r for r in d.reasons)
    d = Q.decide(*_meas(impulse=0.01), 13.0, TH)
    assert d.decision == "enhance" and any("despike" in r for r in d.reasons)


@pytest.mark.parametrize("kwargs,text_h,needle", [
    ({}, 5.0, "text too small"), ({"glare": 0.2}, 13.0, "glare"), ({"unev": 3.0}, 13.0, "uneven lighting"),
    ({"paper": 60.0}, 13.0, "too dark"), ({"sigma": 25.0}, 13.0, "heavy noise"), ({"edge": 2.6}, 13.0, "too blurry")])
def test_decision_retake_reasons(kwargs, text_h, needle):
    d = Q.decide(*_meas(**kwargs), text_h, TH)
    assert d.decision == "retake" and any(needle in r for r in d.reasons)


def test_unmeasurable_noise_never_triggers_noise_rules():
    d = Q.decide(*_meas(sigma=0.0, measurable=False), 13.0, TH)
    assert d.decision == "accept"


def test_none_thresholds_disable_their_checks():
    d = Q.decide(*_meas(sigma=30.0, edge=3.0, glare=0.5), 3.0, Q.Thresholds())
    assert d.decision == "accept"


# ---------------------------------------------------------------------------------- thresholds file

def _write_thresholds(path, **over):
    j = {"noise_sigma": {"enhance_above": 2.5, "retake_above": 20.0}, "blur_edge_width": {"enhance_above": 1.1, "retake_above": 2.5},
         "text_height_px": {"retake_below": 6.0}, "glare_fraction": {"retake_above": 0.05}, "illumination_unevenness": {"retake_above": None},
         "impulse_fraction": {"enhance_above": 0.002}, "hints": {"impulse_fraction": 0.001, "jpeg_blockiness": 1.15, "periodic_peak": 1700, "sigma_visible": 1.5}}
    j.update(over)
    path.write_text(json.dumps(j))
    return path


def test_load_thresholds_and_env_style_overrides(tmp_path):
    p = _write_thresholds(tmp_path / "t.json")
    th = Q.load_thresholds(p)
    assert th.noise_enhance == 2.5 and th.text_height_min == 6.0 and th.unevenness_retake is None and th.hint_periodic == 1700
    th = Q.load_thresholds(p, {"noise_enhance": 9.0, "glare_retake": None})
    assert th.noise_enhance == 9.0 and th.glare_retake is None
    with pytest.raises(KeyError):
        Q.load_thresholds(p, {"no_such_threshold": 1})
    assert Q.load_thresholds(tmp_path / "missing.json") is None


# ---------------------------------------------------------------------------------- pipeline integration

def _bgr(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def test_preprocess_v2_order_and_steps(page, tmp_path):
    th = Q.load_thresholds(_write_thresholds(tmp_path / "t.json"))
    clean = preprocess(_bgr(degrade(page)), thresholds=th, mode="v2", disable=())
    assert clean.qinfo["decision"] == "accept"
    assert not {"denoise", "deblur", "despike"} & set(clean.steps)

    noisy = preprocess(_bgr(degrade(page, ("gauss_noise", {"sigma": 8}))), thresholds=th, mode="v2", disable=())
    assert "denoise" in noisy.steps and noisy.qinfo["decision"] == "enhance"
    assert "gaussian" in noisy.qinfo["noise_types"] and abs(noisy.qinfo["scores"]["noise_sigma"] - 8) < 0.7

    blurry = preprocess(_bgr(degrade(page, ("gauss_blur", {"sigma": 1.5}))), thresholds=th, mode="v2", disable=())
    assert "deblur" in blurry.steps

    sp = preprocess(_bgr(degrade(page, ("salt_pepper", {"density": 0.01}))), thresholds=th, mode="v2", disable=())
    assert "despike" in sp.steps and "denoise" not in sp.steps  # impulses are removed, not smoothed as Gaussian noise


def test_preprocess_disable_flag_and_legacy_mode(page, tmp_path):
    th = Q.load_thresholds(_write_thresholds(tmp_path / "t.json"))
    img = _bgr(degrade(page, ("gauss_noise", {"sigma": 8})))
    off = preprocess(img, thresholds=th, mode="v2", disable=("denoise",))
    assert "denoise" not in off.steps
    legacy = preprocess(img, mode="legacy", disable=())
    assert legacy.qinfo is None and "adaptive_threshold+morph_open" in legacy.steps
    with pytest.raises(ValueError):
        from docusight.config import _disabled_steps  # noqa: F401
        import os
        os.environ["DOCUSIGHT_PREPROCESS_DISABLE"] = "nonsense"
        try:
            _disabled_steps()
        finally:
            del os.environ["DOCUSIGHT_PREPROCESS_DISABLE"]


def test_retake_sets_review_flag_without_blocking_extraction(monkeypatch, invoice_tokens, page):
    monkeypatch.setattr(Q, "load_thresholds", lambda *a, **k: Q.Thresholds(text_height_min=1000.0))  # forces a retake
    pipe = DocumentPipeline(engine=FakeEngine(invoice_tokens), classifier=KeywordDocClassifier())
    res = pipe.run(_bgr(degrade(page)))
    assert res.quality.decision == "retake" and "text too small" in res.quality.reasons[0]
    assert res.needs_review and "IMAGE_QUALITY" in res.review_reasons
    assert res.fields["total_amount"].value == "14868.00"  # still extracted
    m = Metrics()
    m.record(res)
    assert 'docusight_quality_decisions_total{decision="retake"' in m.render()


def test_api_response_has_additive_quality_object(monkeypatch, invoice_tokens, page):
    from fastapi.testclient import TestClient

    from docusight import api

    monkeypatch.setattr(Q, "load_thresholds", lambda *a, **k: Q.Thresholds(noise_enhance=2.0, text_height_min=1.0))
    monkeypatch.setattr(api, "DocumentPipeline", lambda: DocumentPipeline(engine=FakeEngine(invoice_tokens), classifier=KeywordDocClassifier()))
    png = cv2.imencode(".png", _bgr(degrade(page)))[1].tobytes()
    with TestClient(api.app) as client:
        body = client.post("/v1/documents/extract", files={"file": ("p.png", png, "image/png")}).json()
    q = body["quality"]
    assert q["decision"] in ("accept", "enhance") and "noise_sigma" in q["scores"] and q["noise_types"]
    assert body["doc_type"] == "invoice"  # original contract untouched


def test_glare_retake_requires_text_loss_evidence():
    th = Q.Thresholds(glare_retake=0.2, glare_ratio_below=0.25)

    def exp(ratio):
        return Q.ExposureMetrics(200, 100, 225.0, 0.5, 0, 1.05, 0, 0.3, ratio)

    n, h, b = _meas()[:3]
    assert Q.decide(n, h, b, exp(0.05), 13.0, th).decision == "retake"  # blob with the text wiped out
    assert Q.decide(n, h, b, exp(0.9), 13.0, th).decision == "accept"  # blown-out paper, text intact
    assert Q.decide(n, h, b, exp(None), 13.0, th).decision == "accept"  # nothing around it to compare with: never a retake
