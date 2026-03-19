from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bridge_config import LevelSpec
from poc_common import DEFAULT_PARAMS


FS = int(DEFAULT_PARAMS["field_size"])


@dataclass(frozen=True)
class TruthCase:
    flow_name: str
    flow_label: str
    u: np.ndarray
    v: np.ndarray
    omega_deg: np.ndarray


def _coords() -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[0:FS, 0:FS].astype(float)
    return y, x


def make_rankine_truth(level: LevelSpec, seed: int) -> TruthCase:
    y, x = _coords()
    cx = FS / 2.0
    cy = FS / 2.0
    gamma = 1500.0
    r_core = 50.0

    rx = x - cx
    ry = y - cy
    r = np.hypot(rx, ry) + 1e-10
    omega_rad = gamma / (2.0 * np.pi * r_core**2)
    v_t = np.where(r <= r_core, omega_rad * r, gamma / (2.0 * np.pi * r))
    u = -v_t * ry / r
    v = v_t * rx / r
    omega_deg = np.where(r <= r_core, np.degrees(omega_rad), 0.0)

    return TruthCase("rankine", "Rankine vortex", u, v, omega_deg)


def make_lamb_oseen_truth(level: LevelSpec, seed: int) -> TruthCase:
    y, x = _coords()
    cx = FS / 2.0
    cy = FS / 2.0
    gamma = 1500.0
    sigma_core = 40.0

    rx = x - cx
    ry = y - cy
    r2 = rx**2 + ry**2 + 1e-10
    r = np.sqrt(r2)

    v_t = (gamma / (2.0 * np.pi * r)) * (1.0 - np.exp(-r2 / sigma_core**2))
    u = -v_t * ry / r
    v = v_t * rx / r
    omega_rad = (gamma / (np.pi * sigma_core**2)) * np.exp(-r2 / sigma_core**2)
    omega_deg = np.degrees(omega_rad)

    return TruthCase("lamb_oseen", "Lamb-Oseen vortex", u, v, omega_deg)


def make_mixed_vortex_truth(level: LevelSpec, seed: int) -> TruthCase:
    y, x = _coords()
    cx = FS / 2.0 + 8.0
    cy = FS / 2.0 - 10.0
    gamma = 1350.0
    sigma_core = 38.0

    rx = x - cx
    ry = y - cy
    r2 = rx**2 + ry**2 + 1e-10
    r = np.sqrt(r2)

    v_t = (gamma / (2.0 * np.pi * r)) * (1.0 - np.exp(-r2 / sigma_core**2))
    u_v = -v_t * ry / r
    v_v = v_t * rx / r
    omega_v = np.degrees((gamma / (np.pi * sigma_core**2)) * np.exp(-r2 / sigma_core**2))

    u0 = 2.2
    v0 = -0.8
    shear = 0.022
    u_s = shear * (y - FS / 2.0)
    v_s = np.zeros_like(u_s)
    omega_s = np.full_like(u_s, np.degrees(-0.5 * shear))

    perturb = level.flow_perturb
    phi = 2.0 * np.pi * ((seed % 11) / 11.0)
    psi = (
        perturb
        * np.sin(2.0 * np.pi * x / FS + phi)
        * np.sin(2.0 * np.pi * y / FS - 0.5 * phi)
    )
    dy = 1.0
    dx = 1.0
    u_p = np.gradient(psi, dy, axis=0)
    v_p = -np.gradient(psi, dx, axis=1)
    dvp_dx = np.gradient(v_p, dx, axis=1)
    dup_dy = np.gradient(u_p, dy, axis=0)
    omega_p = np.degrees(0.5 * (dvp_dx - dup_dy))

    u = u_v + u_s + u_p + u0
    v = v_v + v_s + v_p + v0
    omega_deg = omega_v + omega_s + omega_p

    return TruthCase("mixed_vortex", "Vortex + translation + shear", u, v, omega_deg)


def make_solid_rotation_truth(level: LevelSpec, seed: int) -> TruthCase:
    y, x = _coords()
    cx = FS / 2.0
    cy = FS / 2.0
    omega_deg_const = 3.6
    omega_rad = np.radians(omega_deg_const)

    rx = x - cx
    ry = y - cy
    u = -omega_rad * ry
    v = omega_rad * rx

    perturb = level.flow_perturb * 0.35
    phi = 2.0 * np.pi * ((seed % 13) / 13.0)
    psi = (
        perturb
        * np.cos(2.0 * np.pi * x / FS + phi)
        * np.sin(2.0 * np.pi * y / FS - 0.4 * phi)
    )
    u_p = np.gradient(psi, 1.0, axis=0)
    v_p = -np.gradient(psi, 1.0, axis=1)
    dvp_dx = np.gradient(v_p, 1.0, axis=1)
    dup_dy = np.gradient(u_p, 1.0, axis=0)
    omega_p = np.degrees(0.5 * (dvp_dx - dup_dy))

    u = u + u_p
    v = v + v_p
    omega_deg = np.full_like(u, omega_deg_const) + omega_p

    return TruthCase("solid_rotation", "Solid rotation", u, v, omega_deg)


def make_truth_case(flow_name: str, level: LevelSpec, seed: int) -> TruthCase:
    if flow_name == "rankine":
        return make_rankine_truth(level, seed)
    if flow_name == "lamb_oseen":
        return make_lamb_oseen_truth(level, seed)
    if flow_name == "solid_rotation":
        return make_solid_rotation_truth(level, seed)
    if flow_name == "mixed_vortex":
        return make_mixed_vortex_truth(level, seed)
    raise ValueError(f"Unknown flow_name: {flow_name}")
