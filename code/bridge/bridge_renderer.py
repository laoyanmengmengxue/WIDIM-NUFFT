from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from bridge_config import LevelSpec
from bridge_flow_truth import TruthCase
from bridge_particles import background_field, illumination_field, lowfreq_gain, turnover_alpha
from comparison import _warp_by_flow
from poc_common import DEFAULT_PARAMS, generate_speckle_field


FS = int(DEFAULT_PARAMS["field_size"])
PPP = float(DEFAULT_PARAMS["ppp"])
DTAU = float(DEFAULT_PARAMS["d_tau"])


@dataclass
class RenderedCase:
    i1: np.ndarray
    i2: np.ndarray
    i1_clean: np.ndarray
    i2_clean: np.ndarray
    i2_turnover: np.ndarray
    truth: TruthCase
    density_gain: np.ndarray
    illum_gain: np.ndarray
    turnover: np.ndarray


def render_case(truth: TruthCase, level: LevelSpec, seed: int) -> RenderedCase:
    rng = np.random.default_rng(seed + 70000)

    i1_clean = generate_speckle_field(FS, PPP, DTAU, 1.0, seed + 101)
    i2_clean = _warp_by_flow(i1_clean, truth.u, truth.v)
    i2_new_particles = generate_speckle_field(FS, PPP, DTAU, 1.0, seed + 202)

    density_gain = lowfreq_gain(i1_clean.shape, rng, level.rho_var)
    illum_gain = illumination_field(i1_clean.shape, rng, level.a_illum)
    turnover = turnover_alpha(i1_clean.shape, rng, level.p_drop)

    i1_mod = i1_clean * density_gain * illum_gain
    i2_warp = i2_clean * density_gain * illum_gain
    i2_new = i2_new_particles * density_gain * illum_gain
    i2_turnover = (1.0 - turnover) * i2_warp + turnover * i2_new

    i1 = i1_mod + background_field(i1_mod.shape, rng, level.sigma_bg)
    i2 = i2_turnover + background_field(i2_turnover.shape, rng, level.sigma_bg)

    if level.blur_sigma > 0:
        i1 = gaussian_filter(i1, sigma=level.blur_sigma, mode="reflect")
        i2 = gaussian_filter(i2, sigma=level.blur_sigma, mode="reflect")

    i1 = np.clip(i1, 0.0, 1.0)
    i2 = np.clip(i2, 0.0, 1.0)

    return RenderedCase(
        i1=i1,
        i2=i2,
        i1_clean=np.clip(i1_clean, 0.0, 1.0),
        i2_clean=np.clip(i2_clean, 0.0, 1.0),
        i2_turnover=np.clip(i2_turnover, 0.0, 1.0),
        truth=truth,
        density_gain=density_gain,
        illum_gain=illum_gain,
        turnover=turnover,
    )

