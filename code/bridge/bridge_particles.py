from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def lowfreq_gain(shape: tuple[int, int], rng: np.random.Generator, strength: float, sigma: float = 26.0) -> np.ndarray:
    if strength <= 0:
        return np.ones(shape, dtype=float)
    base = rng.normal(0.0, 1.0, shape)
    smooth = gaussian_filter(base, sigma=sigma, mode="reflect")
    smooth /= float(np.std(smooth) + 1e-8)
    gain = 1.0 + strength * 0.35 * smooth
    return np.clip(gain, 0.35, 1.80)


def illumination_field(shape: tuple[int, int], rng: np.random.Generator, strength: float) -> np.ndarray:
    if strength <= 0:
        return np.ones(shape, dtype=float)

    h, w = shape
    y, x = np.mgrid[0:h, 0:w].astype(float)
    x = (x - 0.5 * w) / max(w, 1)
    y = (y - 0.5 * h) / max(h, 1)
    ax = rng.uniform(-1.0, 1.0)
    ay = rng.uniform(-1.0, 1.0)
    grad = ax * x + ay * y

    spot_x = rng.uniform(-0.3, 0.3)
    spot_y = rng.uniform(-0.3, 0.3)
    r2 = (x - spot_x) ** 2 + (y - spot_y) ** 2
    spot = np.exp(-r2 / 0.12)

    field = 1.0 + strength * (0.45 * grad + 0.55 * (spot - np.mean(spot)))
    return np.clip(field, 0.45, 1.85)


def turnover_alpha(shape: tuple[int, int], rng: np.random.Generator, p_drop: float) -> np.ndarray:
    if p_drop <= 0:
        return np.zeros(shape, dtype=float)
    mask = (rng.random(shape) < p_drop).astype(float)
    alpha = gaussian_filter(mask, sigma=1.2, mode="reflect")
    alpha /= float(np.max(alpha) + 1e-8)
    alpha *= min(1.0, p_drop * 1.8)
    return np.clip(alpha, 0.0, 1.0)


def background_field(shape: tuple[int, int], rng: np.random.Generator, sigma_bg: float) -> np.ndarray:
    h, w = shape
    base = rng.normal(0.0, sigma_bg, shape)
    drift = gaussian_filter(rng.normal(0.0, sigma_bg * 0.7, shape), sigma=14.0, mode="reflect")
    offset = rng.uniform(0.0, sigma_bg * 1.5)
    return base + drift + offset

