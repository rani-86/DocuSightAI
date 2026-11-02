"""Document image quality assessment (DIQA): pure numpy / OpenCV estimators.

Every estimator takes a single-channel uint8 image and returns a small dataclass. The docstrings say what the
number means, why it is computed that way, and where it is known to fail; `scripts/validate_quality.py` measures
each of them against known ground truth (data/quality_calib).

Design rule shared by all estimators: a document page is mostly flat paper plus sparse high-contrast strokes. A
global statistic (variance of the Laplacian, MAD of a high-pass residual) mixes the two populations, so it measures
*the page content* as much as the *image defect*. We therefore first find the pixels that are paper (flat tiles) or
the pixels that are text edges, and measure each defect only where it is observable.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

TILE = 32  # noise tile size in px
_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
_IMMERKAER_NORM = 6.0  # sqrt(sum(kernel^2)) = sqrt(36): response std = 6 * sigma for white noise
_SQRT_HALF_PI = float(np.sqrt(np.pi / 2))


def _gray_u8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------- flat tiles

def _tile_view(a: np.ndarray, tile: int) -> np.ndarray:
    rows, cols = a.shape[0] // tile, a.shape[1] // tile
    return a[: rows * tile, : cols * tile].reshape(rows, tile, cols, tile)


def _immerkaer_abs(gray_f32: np.ndarray) -> np.ndarray:
    return np.abs(cv2.filter2D(gray_f32, cv2.CV_32F, _IMMERKAER, borderType=cv2.BORDER_REFLECT))


def _tile_sigma(absresp: np.ndarray, tile: int) -> np.ndarray:
    """Robust per-tile noise sigma: median |I*N| / (0.6745 * 6). Strokes covering < 50% of a tile do not move the median."""
    t = _tile_view(absresp, tile).transpose(0, 2, 1, 3)  # (rows, cols, tile, tile)
    return np.median(t.reshape(t.shape[0], t.shape[1], -1), axis=2) / (0.6745 * _IMMERKAER_NORM)


def despike(gray: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Find isolated single-pixel outliers (salt-and-pepper) and replace them with the 3x3 median.

    A pixel is an impulse if it differs from its 3x3 median by more than max(25, 6*sigma) AND at least 7 of its 8
    neighbours agree with that median (within max(25, 3.5*sigma)). The second condition is what separates an impulse from the
    end/corner of a 1-px text stroke or table rule (which has dark neighbours along the stroke).
    Fails: on a mid-grey paper level a *salt* impulse (255 on ~220 paper) deviates by only ~35 and is
    indistinguishable from a 1.5-sigma noise sample once sigma > ~6, so only the *pepper* half of the impulses is reliably
    found (on dark paper the reverse holds); clusters of impulses (density > ~5%) defeat the neighbour-agreement test.
    """
    med = cv2.medianBlur(gray, 3)
    g, m = gray.astype(np.int16), med.astype(np.int16)
    dev = np.abs(g - m) > max(25.0, 6.0 * sigma)
    pad = np.pad(g, 1, mode="edge")
    h, w = g.shape
    nb_thr = max(25.0, 3.5 * sigma)
    agree = np.zeros(g.shape, np.int8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                agree += (np.abs(pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] - m) < nb_thr)
    imp = dev & (agree >= 7)
    return imp, np.where(imp, med, gray)


@dataclass
class FlatMap:
    """Which TILE x TILE tiles are paper (no text edges, no texture) and the per-tile quantities used to decide."""

    mask: np.ndarray  # bool (rows, cols)
    tile: int
    grad_max: np.ndarray  # per-tile max smoothed gradient (gray levels / px)
    sigma_tile: np.ndarray  # per-tile robust noise sigma
    impulses: np.ndarray  # bool full-resolution mask of isolated impulse pixels
    despiked: np.ndarray  # uint8 image with impulses replaced by the 3x3 median
    min_tiles: int = 12

    @property
    def fraction(self) -> float:
        return float(self.mask.mean()) if self.mask.size else 0.0

    @property
    def measurable(self) -> bool:
        return int(self.mask.sum()) >= self.min_tiles

    def pixel_mask(self, shape: tuple[int, int]) -> np.ndarray:
        m = np.kron(self.mask, np.ones((self.tile, self.tile), bool))
        out = np.zeros(shape, bool)
        out[: m.shape[0], : m.shape[1]] = m[: shape[0], : shape[1]]
        return out


def flat_tiles(gray: np.ndarray, tile: int = TILE, min_tiles: int = 12) -> FlatMap:
    """Select tiles that contain only paper (plus noise), i.e. where any variation is defect rather than content.

    A tile is flat if, after removing isolated impulses, BOTH hold (sigma_t = the tile's own robust noise sigma):
      * its largest Gaussian(1)-smoothed gradient is below 6 + 0.35*sigma_t gray levels/px  (no stroke edge), and
      * the 4x4 grid of 8x8 block means, after subtracting their best-fit plane (so illumination ramps are allowed), spans
        less than 1.5 + 0.7*sigma_t levels  (no low-frequency structure: a noise-only tile's block means span ~0.45*sigma_t).
    Why adaptive: noise itself raises gradients, so fixed thresholds throw away every tile at sigma >= 14; scaling with
    the tile's *robust* sigma (median, immune to a few stroke pixels) keeps them. Why the block-mean test: blurred or
    down-scaled dense text is smooth enough to pass a gradient test but still has structure at the 8-32 px scale.
    Impulses are removed first because a median pre-filter (the obvious alternative) also erases 1-px strokes and rules and
    would declare tiles containing them "flat".
    Fails: pages where text covers almost every tile (then `measurable` is False and callers must treat the page's noise as
    unknown rather than use a biased number); halftone / textured paper (rejected as non-flat); a tile of uniform dark ink.
    """
    g = _gray_u8(gray)
    a0 = _immerkaer_abs(g.astype(np.float32))
    sigma0 = float(np.percentile(_tile_sigma(a0, tile), 25)) if a0.shape[0] >= tile and a0.shape[1] >= tile else 0.0
    imp, dsp = despike(g, sigma0)
    a = _immerkaer_abs(dsp.astype(np.float32))
    sig_t = _tile_sigma(a, tile)
    sm = cv2.GaussianBlur(dsp, (0, 0), 1.0).astype(np.float32)
    mag = np.hypot(cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3)) / 8.0
    gmax = _tile_view(mag, tile).max(axis=(1, 3))
    rows, cols = gmax.shape
    k = tile // 8
    bm = cv2.resize(sm[: rows * tile, : cols * tile], (cols * k, rows * k), interpolation=cv2.INTER_AREA)
    bm = bm.reshape(rows, k, cols, k).transpose(0, 2, 1, 3).reshape(rows, cols, k * k)
    yy, xx = np.mgrid[:k, :k]
    A = np.stack([xx.ravel(), yy.ravel(), np.ones(k * k)], 1).astype(np.float32)
    resid = bm - (bm @ np.linalg.pinv(A).T) @ A.T
    span = resid.max(axis=2) - resid.min(axis=2)
    mask = (gmax < 6.0 + 0.35 * sig_t) & (span < 1.5 + 0.7 * sig_t)
    return FlatMap(mask=mask, tile=tile, grad_max=gmax, sigma_tile=sig_t, impulses=imp, despiked=dsp, min_tiles=min_tiles)


# ----------------------------------------------------------------------------- noise sigma

@dataclass
class NoiseEstimate:
    sigma: float  # the value to use (== sigma_mad); 0.0 when not measurable
    sigma_immerkaer: float
    sigma_mad: float
    flat_fraction: float  # fraction of all tiles that were used
    n_tiles: int
    measurable: bool  # False: too few paper tiles (dense page) or all clipped - the noise level is UNKNOWN, not zero


def noise_sigma(gray: np.ndarray, flat: Optional[FlatMap] = None, drop_clipped: bool = True) -> NoiseEstimate:
    """Noise standard deviation (gray levels) of the white-noise component, estimated on paper-only tiles.

    What: both estimators use the Immerkaer 3x3 kernel N = [1 -2 1; -2 4 -2; 1 -2 1], which cancels any locally planar
    signal and leaves noise with std 6*sigma. They run on the *despiked* image (isolated impulses are reported separately by
    `impulse_fraction`) and on flat tiles only:
      (a) Immerkaer's fast estimator: sigma = sqrt(pi/2)/6 * mean|I*N|   (efficient under Gaussian noise).
      (b) MAD of the same residual: sigma = 1.4826 * MAD(I*N) / 6         (robust to heavy tails).
    Both are computed per tile and aggregated by the median over tiles. `sigma` returns (b).
    Why not the whole page: text strokes put huge values in I*N, and even the *median* of the full-page residual is 0 when
    more than half of the pixels are exactly-flat paper, which is why the legacy estimator reported 0.0 on most test pages.
    Fails: (1) tiles with >50% saturated pixels are skipped, because clipping hides noise - a page whose paper is blown out to
    255 reports 0 although a sensor noise may have existed before clipping; (2) spatially correlated noise (JPEG, scanner
    sharpening, resampled images) is under-estimated by a 3x3 high-pass: this is the *white* component only; (3) pages
    with fewer than `min_tiles` paper tiles return measurable=False.
    """
    flat = flat or flat_tiles(gray)
    g = _gray_u8(gray)
    if not flat.measurable:
        return NoiseEstimate(0.0, 0.0, 0.0, flat.fraction, 0, False)
    resp = cv2.filter2D(flat.despiked.astype(np.float32), cv2.CV_32F, _IMMERKAER, borderType=cv2.BORDER_REFLECT)
    t = flat.tile
    imm, mad = [], []
    for r, c in zip(*np.nonzero(flat.mask)):
        patch = g[r * t:(r + 1) * t, c * t:(c + 1) * t]
        if drop_clipped and ((patch >= 255).mean() > 0.5 or (patch <= 0).mean() > 0.5):
            continue
        x = resp[r * t + 1:(r + 1) * t - 1, c * t + 1:(c + 1) * t - 1].ravel()  # drop 1px border (filter support)
        imm.append(_SQRT_HALF_PI / _IMMERKAER_NORM * np.abs(x).mean())
        mad.append(1.4826 * np.median(np.abs(x - np.median(x))) / _IMMERKAER_NORM)
    if not mad:
        return NoiseEstimate(0.0, 0.0, 0.0, flat.fraction, 0, False)
    return NoiseEstimate(sigma=float(np.median(mad)), sigma_immerkaer=float(np.median(imm)), sigma_mad=float(np.median(mad)),
                         flat_fraction=len(mad) / flat.mask.size, n_tiles=len(mad), measurable=True)


# ----------------------------------------------------------------------------- noise type hints

@dataclass
class NoiseHints:
    impulse_fraction: float  # salt-and-pepper
    jpeg_blockiness: float  # 8x8 boundary / interior second-difference ratio, rules excluded (1.0 = no grid)
    periodic_peak: Optional[float]  # FFT peak-to-background ratio (None if not measurable)
    n_flat_pixels: int


def impulse_fraction(flat: FlatMap, shape: tuple[int, int]) -> tuple[float, int]:
    """Fraction of paper pixels that are isolated impulses (see `despike`).

    What/why: salt-and-pepper pixels are isolated near-0/255 outliers; Gaussian noise rarely deviates by more than 6 sigma, and
    text strokes are excluded by the neighbour-agreement rule. Counted on flat tiles only.
    Fails: only the impulses far from the paper level are found (pepper on light paper, salt on dark paper), so the number is
    roughly half the true density on a light page; dust/specks on real paper count too (arguably correct); returns 0 when no
    paper tiles exist.
    """
    pm = flat.pixel_mask(shape)
    if not pm.any():
        return 0.0, 0
    return float(flat.impulses[pm].mean()), int(pm.sum())


def jpeg_blockiness(gray: np.ndarray, eps: float = 0.5) -> float:
    """Strength of an 8x8 compression grid: |second difference| at block-boundary pixels vs interior pixels.

    What/why: JPEG quantises 8x8 DCT blocks independently, which adds a discontinuity at pixel pairs straddling a block border
    (phases 7|0). The second difference centred on phase 0 or 7 is compared with the one centred on phases 2-5. Text edges
    fall uniformly on all 8 phases, so the ratio is ~1 without JPEG. Long table rules / ruled lines are masked out first,
    because a rule sitting on a fixed phase is grid-aligned content that would fake blockiness (the clean ruled invoice
    scored 0.82 vs 1.12 for a clean unruled page before masking).
    Fails: (1) evidence is only strong at JPEG quality <~30 on text pages - at q 50-75 it is smaller than the page-to-page
    variation, and the measure cannot tell them from clean; (2) any resampling (resize, rotation, perspective) destroys the 8x8
    grid, so measure it on the decoded input before geometric correction; (3) a perfectly flat region shows no blocking even
    at q=15, so pages with almost no content give ~1.0.
    """
    g = gray.astype(np.float32)
    binv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    lines = cv2.morphologyEx(binv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 1))) | \
        cv2.morphologyEx(binv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 31)))
    keep = cv2.dilate(lines, np.ones((9, 9), np.uint8)) == 0
    ratios = []
    for axis in (1, 0):
        d2 = np.abs(np.diff(g, 2, axis=axis))
        n = d2.shape[axis]
        ph = (np.arange(n) + 1) % 8
        valid = keep[:, 1:-1] if axis == 1 else keep[1:-1, :]
        sh = (1, n) if axis == 1 else (n, 1)
        bnd = np.isin(ph, (0, 7)).reshape(sh) & valid
        inn = np.isin(ph, (2, 3, 4, 5)).reshape(sh) & valid
        if bnd.sum() > 500 and inn.sum() > 500:
            ratios.append((d2[bnd].mean() + eps) / (d2[inn].mean() + eps))
    return float(np.mean(ratios)) if ratios else 1.0


def periodic_peak(gray: np.ndarray, flat: FlatMap, block: int = 32, max_blocks: int = 48) -> Optional[float]:
    """Strength of the strongest isolated spectral peak in paper regions (periodic / moire / scan-line noise).

    What/why: a sinusoidal or line pattern concentrates energy in one or two FFT bins while white noise spreads it evenly.
    Up to `max_blocks` flat 32x32 tiles are plane-detrended (so an illumination ramp does not leak into low frequencies),
    Hann-windowed and their power spectra averaged; the score is max(P) / (median(P) + floor) over frequencies outside the lowest 3
    bins, where floor = the power of 8-bit quantisation noise (sigma_q^2 = 1/12 per pixel). Without the floor the ratio
    explodes on noise-free paper (median ~ 0): this was the bug in the first version (a fitted threshold of 5e12).
    Fails: returns None when there is no flat tile (very dense pages); a pattern with period close to the 32 px block leaks across
    the Hann window; real paper texture or halftone raises the baseline; a strong smooth glare/shadow edge in a tile
    can produce a mid-sized score (see validation: shadow 0.7 scores ~100 vs 3000+ for the weakest periodic pattern).
    """
    if not flat.measurable:
        return None
    t = flat.tile
    win = np.outer(np.hanning(block), np.hanning(block)).astype(np.float32)
    yy, xx = np.mgrid[:block, :block].astype(np.float32)
    A = np.stack([xx.ravel(), yy.ravel(), np.ones(block * block, np.float32)], 1)
    pinv = np.linalg.pinv(A)
    spec, n = None, 0
    for r, c in zip(*np.nonzero(flat.mask)):
        p = gray[r * t:r * t + block, c * t:c * t + block].astype(np.float32)
        p = (p - (A @ (pinv @ p.ravel())).reshape(block, block)) * win
        ps = np.abs(np.fft.fftshift(np.fft.fft2(p))) ** 2
        spec = ps if spec is None else spec + ps
        n += 1
        if n >= max_blocks:
            break
    spec /= n
    Y, X = np.mgrid[:block, :block]
    vals = spec[np.hypot(Y - block // 2, X - block // 2) > 3]
    floor = (1.0 / 12.0) * float((win ** 2).sum())
    return float(vals.max() / (np.median(vals) + floor))


def noise_hints(gray: np.ndarray, flat: Optional[FlatMap] = None) -> NoiseHints:
    g = _gray_u8(gray)
    flat = flat or flat_tiles(g)
    imp, npx = impulse_fraction(flat, g.shape)
    return NoiseHints(impulse_fraction=imp, jpeg_blockiness=jpeg_blockiness(g), periodic_peak=periodic_peak(g, flat), n_flat_pixels=npx)


# ----------------------------------------------------------------------------- blur

@dataclass
class BlurMetrics:
    edge_width: float  # Gaussian-equivalent edge sigma in px (bigger = blurrier); nan if no edges
    lap_var_edge: float  # mean squared Laplacian over edge regions
    tenengrad: float  # mean squared Sobel magnitude over edge pixels
    tenengrad_norm: float  # tenengrad / (mean local edge contrast)^2 : contrast-invariant sharpness
    edge_density: float  # fraction of pixels that are edge pixels
    n_edge_pixels: int


def blur_metrics(gray: np.ndarray, min_edges: int = 200) -> BlurMetrics:
    """Sharpness measured only on text edges, of an image that has already been denoised.

    What: Sobel gradient magnitude g (gray/px) and Laplacian L are computed; edge pixels are those with
    g > max(10, 0.3 * p99.5(g)) (relative, so a faint but sharp page still has edges). Returned:
      * edge_width : for each edge pixel that is a local gradient maximum, width = local range / (sqrt(2*pi) * g). For a
        step of contrast C blurred by a Gaussian of std s, the peak gradient is C/(s*sqrt(2*pi)), so this recovers s in
        px and is independent of contrast and of how much text is on the page. Median over edge pixels.
      * lap_var_edge : Laplacian energy restricted to edge regions (plain Laplacian variance divided by edge density
        up to the mean term), so blank paper no longer dilutes it. Still scales with contrast^2.
      * tenengrad / tenengrad_norm : mean g^2 on edge pixels, and the same divided by the squared mean local contrast at those edges (the page-level p95-p5 spread is ~0 on text-sparse pages, so it cannot be used).
    Why after denoising: noise adds high-frequency energy that inflates every gradient-based sharpness score, so a
    blurry noisy page can look sharp. Calling this on a non-denoised image reintroduces that bias.
    Fails: glyph strokes are 1-3 px wide, so even a perfectly sharp page has a non-zero intrinsic edge_width (anti-aliasing,
    stroke width) - the score is a relative scale, not absolute focus; pages with almost no text (< `min_edges` edge
    pixels) return nan; heavy denoising itself blurs edges slightly (non-local means keeps them better than a Gaussian).
    """
    g = gray.astype(np.float32)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8.0, cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    mag = np.hypot(gx, gy)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    thr = max(10.0, 0.3 * float(np.percentile(mag, 99.5)))
    edge = mag > thr
    n_edge = int(edge.sum())
    if n_edge < min_edges:
        return BlurMetrics(float("nan"), 0.0, 0.0, 0.0, n_edge / edge.size, n_edge)
    region = cv2.dilate(edge.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    # local maxima of the gradient magnitude along the 3x3 neighbourhood = thin edge ridge
    ridge = edge & (mag >= cv2.dilate(mag, np.ones((3, 3), np.uint8)) - 1e-6)
    local_range = cv2.dilate(g, np.ones((9, 9), np.uint8)) - cv2.erode(g, np.ones((9, 9), np.uint8))
    width = local_range[ridge] / (np.sqrt(2 * np.pi) * mag[ridge])
    ten = float((mag[edge] ** 2).mean())
    contrast = max(1.0, float(local_range[edge].mean()))  # stroke-vs-paper contrast actually seen at the edges
    return BlurMetrics(edge_width=float(np.median(width)), lap_var_edge=float((lap[region] ** 2).mean()), tenengrad=ten,
                       tenengrad_norm=ten / contrast ** 2, edge_density=n_edge / edge.size, n_edge_pixels=n_edge)


# ----------------------------------------------------------------------------- exposure

@dataclass
class ExposureMetrics:
    mean: float
    contrast: float  # p95 - p5
    paper_level: float  # median of the estimated background
    clipped_high: float  # fraction of pixels >= 250
    clipped_low: float  # fraction of pixels <= 5
    unevenness: float  # p98 / p2 of the background map (1.0 = perfectly even)
    shadow_fraction: float  # fraction of area whose background < 0.7 x the median background
    glare_fraction: float  # area of large blown-out regions on a page whose paper is not itself saturated
    glare_edge_ratio: Optional[float]  # edge density inside glare / outside (~0 = text wiped out)


def background_map(gray: np.ndarray, work: int = 160) -> np.ndarray:
    """Paper brightness field: downsample, grey-close away the dark strokes, smooth. Returns a float map at `work` px."""
    h, w = gray.shape
    s = work / max(h, w)
    small = cv2.resize(gray, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
    k = max(3, int(0.06 * max(small.shape)) | 1)  # kernel larger than any glyph / table rule at this scale
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return cv2.GaussianBlur(bg, (0, 0), k / 3.0).astype(np.float32)


def exposure_metrics(gray: np.ndarray) -> ExposureMetrics:
    """Brightness, contrast, clipping, lighting unevenness and glare.

    What: mean / percentile contrast / clipped fractions describe global exposure. `unevenness` = ratio of the 98th to
    the 2nd percentile of a smooth paper-brightness map (so strokes are ignored). `shadow_fraction` is the share of
    area noticeably darker than typical paper. Glare = big connected blobs of >= 250 pixels (after a 25 px opening so
    isolated white specks and text counters do not count) on a page whose own paper level is below 240; `glare_edge_ratio`
    compares edge density inside the (hole-filled) blob with the ring around it, band by band (a blob that wiped out text has ~0; blown-out paper that still carries legible text is ~1; None when no band has surrounding text to compare with).
    Fails: a scan whose paper is legitimately pure white (paper_level >= 240) cannot be told from glare, so glare is
    reported as 0 there; large white figures/logos on grey paper are false glare (the edge ratio then stays high, which is why retake uses it); a bright-side saturation from a lighting gradient is blown-out paper, not glare, and is told apart only by its intact text edges; a very dark page makes the
    shadow_fraction meaningless (use paper_level).
    """
    p5, p95 = np.percentile(gray, (5, 95))
    bg = background_map(gray)
    paper = float(np.median(bg))
    unev = float(np.percentile(bg, 98) / max(1.0, np.percentile(bg, 2)))
    shadow = float((bg < 0.7 * paper).mean())
    glare, ratio = 0.0, None
    if paper < 240:
        blown = (gray >= 250).astype(np.uint8)
        blown = cv2.morphologyEx(blown, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
        glare = float(blown.mean())
        if glare > 0.002:
            edges = cv2.Canny(gray, 50, 150) > 0
            # Fill the holes that dark text itself punches in the blown-out mask; otherwise every text edge lies "outside" the region
            # and the ratio is ~0 even when the text is perfectly legible (found on a gradient-lit page whose bright side saturates).
            inside = cv2.morphologyEx(blown, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))).astype(bool)
            # Text-loss evidence, band by band (32 px high): where the ring just around the blob carries text edges, text is expected
            # inside the blob too; ratio = inside edge density / ring edge density, averaged over those bands (clipped to [0, 2]).
            # Bands whose surroundings are blank are ignored - a blown-out blank margin has no text to lose - and if no band qualifies the
            # ratio is None, which can never justify a retake.
            ring = cv2.dilate(blown, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81))).astype(bool) & ~inside
            per_band = []
            for r0 in range(0, gray.shape[0], 32):
                ib, rb, eb = inside[r0:r0 + 32], ring[r0:r0 + 32], edges[r0:r0 + 32]
                if ib.sum() < 500 or rb.sum() < 200:
                    continue
                ring_d = float(eb[rb].mean())
                if ring_d > 0.02:
                    per_band.append(min(2.0, float(eb[ib].mean()) / ring_d))
            ratio = float(np.mean(per_band)) if per_band else None
    return ExposureMetrics(mean=float(gray.mean()), contrast=float(p95 - p5), paper_level=paper, clipped_high=float((gray >= 250).mean()),
                           clipped_low=float((gray <= 5).mean()), unevenness=unev, shadow_fraction=shadow, glare_fraction=glare,
                           glare_edge_ratio=ratio)


# ----------------------------------------------------------------------------- resolution

@dataclass
class ResolutionMetrics:
    text_height_px: float  # median glyph-component height
    p25: float
    n_components: int


def text_height(gray: np.ndarray, min_area: int = 8) -> ResolutionMetrics:
    """Median height (px) of glyph-sized connected components.

    What/why: OCR recognition needs a minimum glyph size (roughly 8-10 px for capital height); the median connected-
    component height of the binarised page is a cheap proxy. Components that are too large to be glyphs (table rules,
    photos, merged blobs: taller than 5% of the page or wider than 30% of it) and specks (< `min_area` px) are excluded.
    Fails: touching characters merge into wide blobs (height unchanged, fine) but dotted letters (i, j) split in two
    (lowers the median slightly); severe blur merges lines vertically; heavy noise creates specks - call it on the
    denoised image; all-caps vs mixed-case pages differ by ~30% at the same font size, so thresholds should be
    conservative.
    """
    binv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binv, connectivity=8)
    h, w = gray.shape
    hs = [s[cv2.CC_STAT_HEIGHT] for s in stats[1:]
          if s[cv2.CC_STAT_AREA] >= min_area and s[cv2.CC_STAT_HEIGHT] <= 0.05 * h and s[cv2.CC_STAT_WIDTH] <= 0.3 * w]
    if not hs:
        return ResolutionMetrics(0.0, 0.0, 0)
    return ResolutionMetrics(float(np.median(hs)), float(np.percentile(hs, 25)), len(hs))


# ----------------------------------------------------------------------------- report

@dataclass
class QualityReport:
    noise: NoiseEstimate
    hints: NoiseHints
    blur: BlurMetrics
    exposure: ExposureMetrics
    resolution: ResolutionMetrics
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def assess(gray: np.ndarray, denoised: Optional[np.ndarray] = None) -> QualityReport:
    """Run every estimator. `denoised` (if given) is used for blur and resolution; otherwise `gray` is used as is."""
    gray = _gray_u8(gray)
    flat = flat_tiles(gray)
    base = _gray_u8(denoised) if denoised is not None else gray
    return QualityReport(noise=noise_sigma(gray, flat), hints=noise_hints(gray, flat), blur=blur_metrics(base),
                         exposure=exposure_metrics(gray), resolution=text_height(base))


# ----------------------------------------------------------------------------- thresholds and the accept / enhance / retake decision

THRESHOLDS_PATH = Path(__file__).with_name("quality_thresholds.json")


@dataclass
class Thresholds:
    """Gate thresholds. None means 'never reached in the calibration experiment' - that check is then inactive.

    All numeric values come from `docusight/quality_thresholds.json`, written by `scripts/quality_vs_cer.py` (which also stores the
    evidence each value was read from). Individual values can be overridden with environment variables
    DOCUSIGHT_Q_<FIELD_NAME_UPPERCASE> (see config.py), e.g. DOCUSIGHT_Q_NOISE_ENHANCE=3.0.
    """

    noise_enhance: Optional[float] = None  # denoise when estimated sigma is above this
    noise_retake: Optional[float] = None  # retake when sigma is above this (even the denoised page is unreadable)
    blur_enhance: Optional[float] = None  # deblur when edge width (px, after denoising) is above this
    blur_retake: Optional[float] = None
    text_height_min: Optional[float] = None  # retake when native glyph height (px) is below this
    glare_retake: Optional[float] = None  # retake needs this fraction of the page blown out ...
    glare_ratio_below: Optional[float] = None  # ... AND the text-edge ratio around/inside the blob below this (None ratio never retakes)
    unevenness_retake: Optional[float] = None  # retake when background max/min brightness ratio is above this
    impulse_enhance: Optional[float] = None  # despike when isolated-impulse fraction is above this
    dark_paper_min: Optional[float] = None  # retake when paper brightness (gray level) is below this
    clahe_contrast_below: Optional[float] = 150.0  # LEGACY gate (p95-p5 spread), not calibrated; see docs
    hint_impulse: Optional[float] = None
    hint_jpeg: Optional[float] = None
    hint_periodic: Optional[float] = None
    sigma_visible: float = 1.5
    source: str = "defaults"


def load_thresholds(path: Optional[Path] = None, overrides: Optional[dict] = None) -> Optional[Thresholds]:
    """Load calibrated thresholds; returns None if the file does not exist (the caller then falls back to the legacy gates)."""
    path = Path(path) if path else THRESHOLDS_PATH
    if not path.exists():
        return None
    j = json.loads(path.read_text())
    g = lambda *keys: _dig(j, keys)  # noqa: E731
    th = Thresholds(
        noise_enhance=g("noise_sigma", "enhance_above"), noise_retake=g("noise_sigma", "retake_above"),
        blur_enhance=g("blur_edge_width", "enhance_above"), blur_retake=g("blur_edge_width", "retake_above"),
        text_height_min=g("text_height_px", "retake_below"), glare_retake=g("glare_fraction", "retake_above"),
        unevenness_retake=g("illumination_unevenness", "retake_above"), glare_ratio_below=g("glare_text_loss", "retake_below"), impulse_enhance=g("impulse_fraction", "enhance_above"),
        dark_paper_min=g("darkness", "retake_below"),
        hint_impulse=g("hints", "impulse_fraction"), hint_jpeg=g("hints", "jpeg_blockiness"), hint_periodic=g("hints", "periodic_peak"),
        sigma_visible=g("hints", "sigma_visible") or 1.5, source=str(path.name))
    for k, v in (overrides or {}).items():
        if not hasattr(th, k):
            raise KeyError(f"unknown quality threshold override: {k}")
        setattr(th, k, None if v is None else float(v))
    return th


def _dig(d, keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def suspected_noise_types(sigma: float, hints: NoiseHints, th: Thresholds) -> list[str]:
    """Names of the noise/artifact types the hints indicate (several can apply). ['none'] if nothing is above threshold."""
    out = []
    if th.hint_periodic is not None and hints.periodic_peak is not None and hints.periodic_peak > th.hint_periodic:
        out.append("periodic")
    if th.hint_impulse is not None and hints.impulse_fraction > th.hint_impulse:
        out.append("salt_pepper")
    if th.hint_jpeg is not None and hints.jpeg_blockiness > th.hint_jpeg:
        out.append("jpeg_blocking")
    if sigma > th.sigma_visible:
        out.append("gaussian")
    return out or ["none"]


@dataclass
class QualityDecision:
    decision: str  # accept | enhance | retake
    enhance: list[str] = field(default_factory=list)  # fixes that were / would be applied
    reasons: list[str] = field(default_factory=list)  # human-readable, one per triggered rule


def decide(noise: NoiseEstimate, hints: NoiseHints, blur: BlurMetrics, exposure: ExposureMetrics, text_height_native: float,
           th: Thresholds, applied: Optional[list[str]] = None) -> QualityDecision:
    """Turn measurements into a decision.

    retake  : at least one defect is beyond what the best available enhancement can repair (the page's CER stayed above the
              calibration target even after enhancement), or information is simply missing (text too small, text wiped out by
              glare, page far too dark). Retake never blocks processing - the page is still OCR'd and flagged for review.
    enhance : at least one fixable defect above its 'enhancement helps' threshold (`applied` lists what the pipeline actually ran).
    accept  : none of the above; the page goes through with minimal processing.
    """
    retake, fix = [], []
    if th.text_height_min is not None and 0 < text_height_native < th.text_height_min:
        retake.append(f"text too small: median glyph height {text_height_native:.1f}px < {th.text_height_min:.1f}px")
    if th.glare_retake is not None and exposure.glare_fraction > th.glare_retake:
        wiped = th.glare_ratio_below is None or (exposure.glare_edge_ratio is not None and exposure.glare_edge_ratio < th.glare_ratio_below)
        if wiped:
            retake.append(f"glare covers {exposure.glare_fraction:.1%} of the page (limit {th.glare_retake:.1%}) and text edges inside it are "
                          f"{'lost' if exposure.glare_edge_ratio is None else f'{exposure.glare_edge_ratio:.2f}x those around it'}")
    if th.unevenness_retake is not None and exposure.unevenness > th.unevenness_retake:
        retake.append(f"uneven lighting: brightest/darkest background ratio {exposure.unevenness:.2f} > {th.unevenness_retake:.2f}")
    if th.dark_paper_min is not None and exposure.paper_level < th.dark_paper_min:
        retake.append(f"page too dark: paper level {exposure.paper_level:.0f} < {th.dark_paper_min:.0f}")
    if noise.measurable and th.noise_retake is not None and noise.sigma > th.noise_retake:
        retake.append(f"heavy noise: sigma {noise.sigma:.1f} > {th.noise_retake:.1f}")
    if not np.isnan(blur.edge_width) and th.blur_retake is not None and blur.edge_width > th.blur_retake:
        retake.append(f"too blurry: edge width {blur.edge_width:.2f}px > {th.blur_retake:.2f}px")
    if th.noise_enhance is not None and noise.measurable and noise.sigma > th.noise_enhance:
        fix.append(f"denoise: sigma {noise.sigma:.1f} > {th.noise_enhance:.1f}")
    if th.impulse_enhance is not None and hints.impulse_fraction > th.impulse_enhance:
        fix.append(f"despike: impulse fraction {hints.impulse_fraction:.4f} > {th.impulse_enhance:.4f}")
    if th.blur_enhance is not None and not np.isnan(blur.edge_width) and blur.edge_width > th.blur_enhance:
        fix.append(f"deblur: edge width {blur.edge_width:.2f}px > {th.blur_enhance:.2f}px")
    if retake:
        return QualityDecision("retake", applied or [], retake + fix)
    if fix:
        return QualityDecision("enhance", applied or [], fix)
    return QualityDecision("accept", applied or [], [])
