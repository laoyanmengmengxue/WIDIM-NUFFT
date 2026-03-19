from __future__ import annotations

import contextlib
import io as _io
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates

from bridge_config import PRIMARY_GATE_PERCENTILE, PRIMARY_STEP
from comparison import _compute_gradients
from comparison import (
    _deform_window,
    _normalized_median_test,
    _replace_outliers,
    _residual_ncc,
    _search_ncc_window,
)
from poc_common import DEFAULT_PARAMS
from poc_point3 import OPT_POC, OPT_R_MIN, OPT_WEIGHT, estimate_rotation


WS = int(DEFAULT_PARAMS["window_size"])
SEARCH_RADIUS = 6
SEARCH_STEP = 1


@dataclass
class BridgeEval:
    centers: list[int]
    omega_truth_grid: np.ndarray
    omega_widim: np.ndarray
    alpha_raw: np.ndarray
    alpha_new: np.ndarray
    snr_map: np.ndarray
    gate: np.ndarray
    snr_thr: float
    rmse_true_raw: float
    rmse_true_widim: float
    rmse_true_new: float
    mae_true_raw: float
    mae_true_widim: float
    mae_true_new: float
    corr_true_raw: float
    corr_true_widim: float
    corr_true_new: float
    corr_raw_gate: float
    corr_new_gate: float
    rmse_raw_gate: float
    rmse_new_gate: float
    mae_raw_gate: float
    mae_new_gate: float
    delta_rmse_true: float
    delta_mae_true: float
    delta_corr_true: float
    delta_rmse_true_bridge: float
    delta_mae_true_bridge: float
    delta_corr_true_bridge: float
    delta_rmse_g: float
    delta_mae_g: float
    delta_corr_g: float
    n_eval: int
    pass_rate: float


def _run_widim(i1: np.ndarray, i2: np.ndarray, subset_size: int = WS, step: int = PRIMARY_STEP, n_iter: int = 3):
    h, w = i1.shape
    half = subset_size // 2
    centers = list(range(half, h - half + 1, step))
    n = len(centers)

    u_wdm = np.zeros((n, n), dtype=float)
    v_wdm = np.zeros((n, n), dtype=float)

    for ri, cr in enumerate(centers):
        for ci, cc in enumerate(centers):
            w1 = i1[cr - half : cr - half + subset_size, cc - half : cc - half + subset_size]
            raw = _search_ncc_window(w1, i2, cr, cc, half, h, w, SEARCH_RADIUS, SEARCH_STEP)
            u_wdm[ri, ci] = raw["dx"]
            v_wdm[ri, ci] = raw["dy"]

    for _ in range(n_iter):
        du_dx, du_dy, dv_dx, dv_dy = _compute_gradients(u_wdm, v_wdm, step)
        outliers = _normalized_median_test(u_wdm, v_wdm)
        if np.any(outliers):
            u_wdm, v_wdm = _replace_outliers(u_wdm, v_wdm, outliers)
            du_dx, du_dy, dv_dx, dv_dy = _compute_gradients(u_wdm, v_wdm, step)
        du_res = np.zeros((n, n), dtype=float)
        dv_res = np.zeros((n, n), dtype=float)
        for ri, cr in enumerate(centers):
            for ci, cc in enumerate(centers):
                w1 = i1[cr - half : cr - half + subset_size, cc - half : cc - half + subset_size]
                w2_def = _deform_window(
                    i2,
                    cr,
                    cc,
                    half,
                    h,
                    w,
                    u_wdm[ri, ci],
                    v_wdm[ri, ci],
                    du_dx[ri, ci],
                    du_dy[ri, ci],
                    dv_dx[ri, ci],
                    dv_dy[ri, ci],
                )
                dx_r, dy_r = _residual_ncc(w1, w2_def, 3)
                du_res[ri, ci] = dx_r
                dv_res[ri, ci] = dy_r
        u_wdm += du_res
        v_wdm += dv_res

    return centers, n, u_wdm, v_wdm


def _shift_window_translation(i2: np.ndarray, cr: int, cc: int, half: int, h: int, w: int, u: float, v: float) -> np.ndarray:
    yr = np.arange(-half, half)
    xr = np.arange(-half, half)
    yr, xr = np.meshgrid(yr, xr, indexing="ij")
    y2 = np.clip((cr + yr) + v, 0, h - 1)
    x2 = np.clip((cc + xr) + u, 0, w - 1)
    return map_coordinates(i2, [y2, x2], order=3, mode="nearest")


def _spatial_snr_gate(alpha_map: np.ndarray, snr_map: np.ndarray, snr_min: float = 1.8, z_thr: float = 2.8, grad_thr: float = 2.8, lap_thr: float = 2.8, edge_margin: int = 1, edge_delta: float = 0.6) -> np.ndarray:
    a = np.asarray(alpha_map, dtype=float)
    s = np.asarray(snr_map, dtype=float)
    n1, n2 = a.shape
    valid = np.isfinite(a) & np.isfinite(s)
    if not np.any(valid):
        return np.zeros_like(valid, dtype=bool)

    z_map = np.full_like(a, np.nan, dtype=float)
    grad_map = np.full_like(a, np.nan, dtype=float)
    grad_ratio = np.full_like(a, np.nan, dtype=float)
    lap_ratio = np.full_like(a, np.nan, dtype=float)

    for i in range(n1):
        for j in range(n2):
            if not valid[i, j]:
                continue
            r0 = max(0, i - 1)
            r1 = min(n1, i + 2)
            c0 = max(0, j - 1)
            c1 = min(n2, j + 2)
            nb = a[r0:r1, c0:c1].reshape(-1)
            center_idx = (i - r0) * (c1 - c0) + (j - c0)
            keep = np.ones(nb.shape[0], dtype=bool)
            keep[center_idx] = False
            nb = nb[keep]
            nb = nb[np.isfinite(nb)]
            if nb.size < 3:
                continue
            med = float(np.median(nb))
            mad = float(np.median(np.abs(nb - med)))
            sigma = 1.4826 * mad
            if sigma < 1e-6:
                sigma = float(np.std(nb)) + 1e-6
            z_map[i, j] = abs(a[i, j] - med) / sigma

    for i in range(n1):
        for j in range(n2):
            if not valid[i, j]:
                continue
            left = a[i, j - 1] if j - 1 >= 0 and np.isfinite(a[i, j - 1]) else np.nan
            right = a[i, j + 1] if j + 1 < n2 and np.isfinite(a[i, j + 1]) else np.nan
            up = a[i - 1, j] if i - 1 >= 0 and np.isfinite(a[i - 1, j]) else np.nan
            down = a[i + 1, j] if i + 1 < n1 and np.isfinite(a[i + 1, j]) else np.nan

            if np.isfinite(left) and np.isfinite(right):
                gx = 0.5 * (right - left)
            elif np.isfinite(right):
                gx = right - a[i, j]
            elif np.isfinite(left):
                gx = a[i, j] - left
            else:
                gx = np.nan

            if np.isfinite(up) and np.isfinite(down):
                gy = 0.5 * (down - up)
            elif np.isfinite(down):
                gy = down - a[i, j]
            elif np.isfinite(up):
                gy = a[i, j] - up
            else:
                gy = np.nan

            if np.isfinite(gx) and np.isfinite(gy):
                grad_map[i, j] = float(np.hypot(gx, gy))

    for i in range(n1):
        for j in range(n2):
            if not np.isfinite(grad_map[i, j]):
                continue
            r0 = max(0, i - 1)
            r1 = min(n1, i + 2)
            c0 = max(0, j - 1)
            c1 = min(n2, j + 2)
            gnb = grad_map[r0:r1, c0:c1].reshape(-1)
            gnb = gnb[np.isfinite(gnb)]
            if gnb.size < 3:
                continue
            grad_ratio[i, j] = grad_map[i, j] / (float(np.median(gnb)) + 1e-6)

    for i in range(n1):
        for j in range(n2):
            if not valid[i, j]:
                continue
            neigh = []
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ii = i + di
                jj = j + dj
                if 0 <= ii < n1 and 0 <= jj < n2 and np.isfinite(a[ii, jj]):
                    neigh.append(a[ii, jj])
            if len(neigh) < 3:
                continue
            lap = abs(len(neigh) * a[i, j] - float(np.sum(neigh)))

            local = []
            r0 = max(0, i - 1)
            r1 = min(n1, i + 2)
            c0 = max(0, j - 1)
            c1 = min(n2, j + 2)
            for ii in range(r0, r1):
                for jj in range(c0, c1):
                    if not valid[ii, jj]:
                        continue
                    neigh2 = []
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        iii = ii + di
                        jjj = jj + dj
                        if 0 <= iii < n1 and 0 <= jjj < n2 and np.isfinite(a[iii, jjj]):
                            neigh2.append(a[iii, jjj])
                    if len(neigh2) >= 3:
                        local.append(abs(len(neigh2) * a[ii, jj] - float(np.sum(neigh2))))
            if len(local) < 3:
                continue
            lap_ratio[i, j] = lap / (float(np.median(local)) + 1e-6)

    weak_snr_big_angle = (s < 2.2) & (np.abs(a) > 15.0)
    bad = valid & ((z_map > z_thr) | (grad_ratio > grad_thr) | (lap_ratio > lap_thr) | weak_snr_big_angle)

    if edge_margin > 0:
        edge = np.zeros_like(valid, dtype=bool)
        edge[:edge_margin, :] = True
        edge[-edge_margin:, :] = True
        edge[:, :edge_margin] = True
        edge[:, -edge_margin:] = True
        ez = z_thr - edge_delta
        eg = grad_thr - edge_delta
        el = lap_thr - edge_delta
        bad_edge = valid & edge & ((z_map > ez) | (grad_ratio > eg) | (lap_ratio > el))
        bad = bad | bad_edge

    return valid & (s >= snr_min) & (~bad)


def _refill_bad_by_trimmed_mean(alpha_map: np.ndarray, pass_mask: np.ndarray, n_iter: int = 3, radius: int = 1) -> np.ndarray:
    a = np.asarray(alpha_map, dtype=float)
    keep = np.asarray(pass_mask, dtype=bool)
    valid_orig = np.isfinite(a)
    work = np.where(keep & valid_orig, a, np.nan)
    target = valid_orig & (~keep)
    n1, n2 = a.shape

    for _ in range(max(1, int(n_iter))):
        new_work = work.copy()
        changed = 0
        for i in range(n1):
            for j in range(n2):
                if (not target[i, j]) or np.isfinite(work[i, j]):
                    continue
                r0 = max(0, i - radius)
                r1 = min(n1, i + radius + 1)
                c0 = max(0, j - radius)
                c1 = min(n2, j + radius + 1)
                nb = work[r0:r1, c0:c1].reshape(-1)
                nb = nb[np.isfinite(nb)]
                if nb.size < 3:
                    continue
                nb.sort()
                core = nb[1:-1] if nb.size >= 5 else nb
                if core.size == 0:
                    continue
                new_work[i, j] = float(np.mean(core))
                changed += 1
        work = new_work
        if changed == 0:
            break
    return work


def _second_pass_sign_bad(alpha_map: np.ndarray, z_thr: float = 2.5, sign_ratio_thr: float = 0.2) -> np.ndarray:
    a = np.asarray(alpha_map, dtype=float)
    n1, n2 = a.shape
    valid = np.isfinite(a)
    bad = np.zeros_like(valid, dtype=bool)

    for i in range(n1):
        for j in range(n2):
            if not valid[i, j]:
                continue
            r0 = max(0, i - 1)
            r1 = min(n1, i + 2)
            c0 = max(0, j - 1)
            c1 = min(n2, j + 2)
            nb = a[r0:r1, c0:c1].reshape(-1)
            center_idx = (i - r0) * (c1 - c0) + (j - c0)
            keep = np.ones(nb.shape[0], dtype=bool)
            keep[center_idx] = False
            nb = nb[keep]
            nb = nb[np.isfinite(nb)]
            if nb.size < 3:
                continue
            med = float(np.median(nb))
            mad = float(np.median(np.abs(nb - med)))
            sigma = 1.4826 * mad
            if sigma < 1e-6:
                sigma = float(np.std(nb)) + 1e-6
            z = abs(a[i, j] - med) / sigma
            center_sign = np.sign(a[i, j])
            if center_sign == 0:
                continue
            nz = np.abs(nb) > 1e-9
            if np.sum(nz) < 3:
                continue
            nb_sign = np.sign(nb[nz])
            same = np.sum(nb_sign == center_sign)
            ratio = same / float(np.sum(nz))
            if ratio < sign_ratio_thr and z > z_thr:
                bad[i, j] = True
    return bad


def _two_pass_gate_refill(alpha_map: np.ndarray, snr_map: np.ndarray) -> np.ndarray:
    gate1 = _spatial_snr_gate(alpha_map, snr_map)
    fill1 = _refill_bad_by_trimmed_mean(alpha_map, gate1, n_iter=3, radius=1)
    bad2 = _second_pass_sign_bad(fill1)
    if not np.any(bad2):
        return fill1
    pass2 = np.isfinite(fill1) & (~bad2)
    return _refill_bad_by_trimmed_mean(fill1, pass2, n_iter=2, radius=1)


def _abs_corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & np.asarray(mask, dtype=bool)
    if int(np.sum(valid)) < 3:
        return float("nan")
    c = np.corrcoef(x[valid], y[valid])[0, 1]
    return float(abs(c))


def _rmse(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & np.asarray(mask, dtype=bool)
    if int(np.sum(valid)) == 0:
        return float("nan")
    d = x[valid] - y[valid]
    return float(np.sqrt(np.mean(d * d)))


def _mae(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & np.asarray(mask, dtype=bool)
    if int(np.sum(valid)) == 0:
        return float("nan")
    return float(np.mean(np.abs(x[valid] - y[valid])))


def evaluate_bridge_case(i1: np.ndarray, i2: np.ndarray, omega_truth_field: np.ndarray) -> BridgeEval:
    h, w = i1.shape
    half = WS // 2

    centers, n, u_wdm, v_wdm = _run_widim(i1, i2, step=PRIMARY_STEP)
    du_dx, du_dy, dv_dx, dv_dy = _compute_gradients(u_wdm, v_wdm, PRIMARY_STEP)
    omega_widim = np.degrees(0.5 * (dv_dx - du_dy))

    alpha_raw = np.full((n, n), np.nan, dtype=float)
    snr_map = np.full((n, n), np.nan, dtype=float)
    omega_truth_grid = np.full((n, n), np.nan, dtype=float)

    for ri, cr in enumerate(centers):
        for ci, cc in enumerate(centers):
            if cr - half < 0 or cr + half > h or cc - half < 0 or cc + half > w:
                continue

            w1 = i1[cr - half : cr - half + WS, cc - half : cc - half + WS]
            w2 = _shift_window_translation(i2, cr, cc, half, h, w, u_wdm[ri, ci], v_wdm[ri, ci])

            with contextlib.redirect_stdout(_io.StringIO()):
                res = estimate_rotation(w1, w2, OPT_R_MIN, OPT_WEIGHT, OPT_POC, verbose=False)

            alpha_raw[ri, ci] = -float(res["angle_est"])
            snr_map[ri, ci] = float(res["snr"])
            omega_truth_grid[ri, ci] = float(omega_truth_field[cr, cc])

    alpha_new = _two_pass_gate_refill(alpha_raw, snr_map)
    finite_snr = np.isfinite(snr_map)
    snr_thr = float(np.percentile(snr_map[finite_snr], PRIMARY_GATE_PERCENTILE)) if np.any(finite_snr) else float("nan")
    gate = finite_snr & (snr_map >= snr_thr)

    eval_mask_new = gate & np.isfinite(alpha_new) & np.isfinite(omega_widim) & np.isfinite(omega_truth_grid)
    eval_mask_raw = gate & np.isfinite(alpha_raw) & np.isfinite(omega_truth_grid)
    corr_true_widim = _abs_corr(omega_widim, omega_truth_grid, eval_mask_new)
    corr_true_new = _abs_corr(alpha_new, omega_truth_grid, eval_mask_new)
    corr_true_raw = _abs_corr(alpha_raw, omega_truth_grid, eval_mask_raw)
    rmse_true_widim = _rmse(omega_widim, omega_truth_grid, eval_mask_new)
    rmse_true_new = _rmse(alpha_new, omega_truth_grid, eval_mask_new)
    rmse_true_raw = _rmse(alpha_raw, omega_truth_grid, eval_mask_raw)
    mae_true_widim = _mae(omega_widim, omega_truth_grid, eval_mask_new)
    mae_true_new = _mae(alpha_new, omega_truth_grid, eval_mask_new)
    mae_true_raw = _mae(alpha_raw, omega_truth_grid, eval_mask_raw)

    corr_raw_gate = _abs_corr(alpha_raw, omega_widim, gate)
    corr_new_gate = _abs_corr(alpha_new, omega_widim, gate)
    rmse_raw_gate = _rmse(alpha_raw, omega_widim, gate)
    rmse_new_gate = _rmse(alpha_new, omega_widim, gate)
    mae_raw_gate = _mae(alpha_raw, omega_widim, gate)
    mae_new_gate = _mae(alpha_new, omega_widim, gate)

    n_eval = int(np.sum(eval_mask_new))
    pass_rate = 100.0 * float(np.sum(gate)) / float(max(np.sum(np.isfinite(alpha_raw)), 1))

    return BridgeEval(
        centers=centers,
        omega_truth_grid=omega_truth_grid,
        omega_widim=omega_widim,
        alpha_raw=alpha_raw,
        alpha_new=alpha_new,
        snr_map=snr_map,
        gate=gate,
        snr_thr=snr_thr,
        rmse_true_raw=rmse_true_raw,
        rmse_true_widim=rmse_true_widim,
        rmse_true_new=rmse_true_new,
        mae_true_raw=mae_true_raw,
        mae_true_widim=mae_true_widim,
        mae_true_new=mae_true_new,
        corr_true_raw=corr_true_raw,
        corr_true_widim=corr_true_widim,
        corr_true_new=corr_true_new,
        corr_raw_gate=corr_raw_gate,
        corr_new_gate=corr_new_gate,
        rmse_raw_gate=rmse_raw_gate,
        rmse_new_gate=rmse_new_gate,
        mae_raw_gate=mae_raw_gate,
        mae_new_gate=mae_new_gate,
        delta_rmse_true=rmse_true_widim - rmse_true_new,
        delta_mae_true=mae_true_widim - mae_true_new,
        delta_corr_true=corr_true_new - corr_true_widim,
        delta_rmse_true_bridge=rmse_true_raw - rmse_true_new,
        delta_mae_true_bridge=mae_true_raw - mae_true_new,
        delta_corr_true_bridge=corr_true_new - corr_true_raw,
        delta_rmse_g=rmse_raw_gate - rmse_new_gate,
        delta_mae_g=mae_raw_gate - mae_new_gate,
        delta_corr_g=corr_new_gate - corr_raw_gate,
        n_eval=n_eval,
        pass_rate=pass_rate,
    )
