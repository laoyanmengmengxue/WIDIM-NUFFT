

"""
build_real_proxy_case_fig.py

Build publication-quality real-data figures:
  1. a compact main-text vortex-ring case figure, and
  2. an appendix-only source-wise proxy-gain distribution figure.

Default representative case: PIVbook cam2_5001, which is used in the paper
because it shows a clearer continuous vortex-ring structure under the unified
full-pair plotting protocol.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap

from comparison import _compute_gradients, widim_fullfield
from poc_point3 import OPT_POC, OPT_R_MIN, OPT_WEIGHT, estimate_rotation
from gt_comparison_vorticity_widim_track_nufft_spatialgate_signisland import (
    SEARCH_RADIUS,
    SEARCH_STEP,
    WS,
    _shift_window_translation,
    _two_pass_gate_refill,
)


REPO = Path(__file__).resolve().parent
IEEE_DIR = REPO / "IEEE-Transactions-LaTeX2e-templates-and-instructions"
OUT_DIR = REPO / "results_fix_figs"
OUT_DIR.mkdir(exist_ok=True)

METRICS_CSV = REPO / "真实数据结果汇总" / "metrics_all_pairs.csv"
DEFAULT_PAIR_LABEL = "cam2_5001"
GATE_PERCENTILE = 5.0
POOL_ORDER = ["PIV_C", "PIV_E", "PIV_book"]
POOL_LABELS = {
    "PIV_C": "PIV-C\nN=100",
    "PIV_E": "PIV-E\nN=198",
    "PIV_book": "PIV-book\nN=22",
}
POOL_COLORS = {
    "PIV_C": "#3C9BC9",
    "PIV_E": "#F2CF66",
    "PIV_book": "#62B38F",
}


def _safe_float(v: float) -> float:
    try:
        if np.isfinite(v):
            return float(v)
    except Exception:
        pass
    return float("nan")


def load_gray(path: Path, max_size: int = 512) -> np.ndarray:
    img = Image.open(path)
    if img.mode != "L":
        img = img.convert("L")
    arr = np.asarray(img, dtype=float) / 255.0
    h, w = arr.shape
    if h > max_size or w > max_size:
        rr = min(h, max_size)
        cc = min(w, max_size)
        r0 = (h - rr) // 2
        c0 = (w - cc) // 2
        arr = arr[r0:r0 + rr, c0:c0 + cc]
    return arr


def abs_corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    xv = x[valid]
    yv = y[valid]
    if xv.size <= 1:
        return float("nan")
    if np.nanstd(xv) <= 1e-12 or np.nanstd(yv) <= 1e-12:
        return float("nan")
    return float(abs(np.corrcoef(xv, yv)[0, 1]))


def rmse(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    d = x[valid] - y[valid]
    if d.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(d * d)))


def mae(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    d = np.abs(x[valid] - y[valid])
    if d.size == 0:
        return float("nan")
    return float(np.mean(d))


def p99_abs(a: np.ndarray, floor: float = 0.05) -> float:
    v = np.asarray(a, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return floor
    return max(float(np.percentile(np.abs(v), 99.0)), floor)


def load_metrics_df() -> pd.DataFrame:
    if not METRICS_CSV.exists():
        raise FileNotFoundError(METRICS_CSV)
    df = pd.read_csv(METRICS_CSV)
    for col in ["delta_corr_gate", "delta_rmse_gate", "delta_mae_gate"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def find_pair_paths(pair_label: str) -> tuple[Path, Path]:
    candidates = sorted(REPO.rglob(f"{pair_label}a.tif"))
    for p1 in candidates:
        p2 = p1.with_name(f"{pair_label}b.tif")
        if p2.exists():
            return p1, p2
    raise FileNotFoundError(f"Cannot find pair {pair_label}a.tif / {pair_label}b.tif under {REPO}")


def pair_metrics(df: pd.DataFrame, pair_label: str) -> dict:
    row = df[df["label"] == pair_label]
    if row.empty:
        return {}
    r = row.iloc[0]
    return {
        "source": str(r["source"]),
        "label": str(r["label"]),
        "delta_corr_gate": float(r["delta_corr_gate"]),
        "delta_rmse_gate": float(r["delta_rmse_gate"]),
        "delta_mae_gate": float(r["delta_mae_gate"]),
    }


def run_pair(i1: np.ndarray, i2: np.ndarray, gate_percentile: float) -> dict:
    wdm = widim_fullfield(
        i1,
        i2,
        subset_size=WS,
        step=10,
        search_radius=SEARCH_RADIUS,
        search_step=SEARCH_STEP,
        n_iter=3,
    )
    centers = wdm["centers"]
    n = int(wdm["n"])
    u = np.asarray(wdm["u_wdm"], dtype=float)
    v = np.asarray(wdm["v_wdm"], dtype=float)
    du_dx, du_dy, dv_dx, _ = _compute_gradients(u, v, 10)
    omega_w = np.degrees(0.5 * (dv_dx - du_dy))

    h, w_img = i1.shape
    half = WS // 2
    alpha_raw = np.full((n, n), np.nan, dtype=float)
    snr_map = np.full((n, n), np.nan, dtype=float)

    for ri, cr in enumerate(centers):
        print(f"  stage=nufft row {ri + 1}/{n}")
        for ci, cc in enumerate(centers):
            if cr - half < 0 or cr + half > h or cc - half < 0 or cc + half > w_img:
                continue
            w1 = i1[cr - half: cr - half + WS, cc - half: cc - half + WS]
            w2_t = _shift_window_translation(i2, cr, cc, half, h, w_img, u[ri, ci], v[ri, ci])
            res = estimate_rotation(w1, w2_t, OPT_R_MIN, OPT_WEIGHT, OPT_POC, verbose=False)
            alpha_raw[ri, ci] = -float(res["angle_est"])
            snr_map[ri, ci] = float(res["snr"])

    alpha_new = _two_pass_gate_refill(alpha_raw, snr_map)
    finite_snr = np.isfinite(snr_map)
    snr_thr = float(np.percentile(snr_map[finite_snr], gate_percentile)) if np.any(finite_snr) else float("nan")
    gate = finite_snr & (snr_map >= snr_thr)

    corr_raw_gate = abs_corr(alpha_raw, omega_w, gate)
    corr_new_gate = abs_corr(alpha_new, omega_w, gate)
    rmse_raw_gate = rmse(alpha_raw, omega_w, gate)
    rmse_new_gate = rmse(alpha_new, omega_w, gate)
    mae_raw_gate = mae(alpha_raw, omega_w, gate)
    mae_new_gate = mae(alpha_new, omega_w, gate)
    valid_windows = np.isfinite(alpha_raw) & np.isfinite(alpha_new) & np.isfinite(omega_w)
    n_windows = int(np.sum(valid_windows))
    n_pass = int(np.sum(valid_windows & gate))

    return {
        "centers": centers,
        "omega_w": omega_w,
        "alpha_raw": alpha_raw,
        "alpha_new": alpha_new,
        "snr": snr_map,
        "gate": gate,
        "snr_thr": snr_thr,
        "corr_raw_gate": corr_raw_gate,
        "corr_new_gate": corr_new_gate,
        "delta_corr_gate": corr_new_gate - corr_raw_gate,
        "rmse_raw_gate": rmse_raw_gate,
        "rmse_new_gate": rmse_new_gate,
        "delta_rmse_gate": rmse_raw_gate - rmse_new_gate,
        "mae_raw_gate": mae_raw_gate,
        "mae_new_gate": mae_new_gate,
        "delta_mae_gate": mae_raw_gate - mae_new_gate,
        "n_windows": n_windows,
        "n_pass": n_pass,
        "pass_rate": 100.0 * n_pass / max(n_windows, 1),
    }


def _crop_window(arr: np.ndarray, center_rc: tuple[int, int], half_size: int) -> np.ndarray:
    r0 = max(int(center_rc[0]) - half_size, 0)
    c0 = max(int(center_rc[1]) - half_size, 0)
    r1 = min(int(center_rc[0]) + half_size, arr.shape[0])
    c1 = min(int(center_rc[1]) + half_size, arr.shape[1])
    return arr[r0:r1, c0:c1]


def _grid_crop(arr: np.ndarray, center_rc: tuple[int, int], half_size: int) -> np.ndarray:
    r0 = max(int(center_rc[0]) - half_size, 0)
    c0 = max(int(center_rc[1]) - half_size, 0)
    r1 = min(int(center_rc[0]) + half_size + 1, arr.shape[0])
    c1 = min(int(center_rc[1]) + half_size + 1, arr.shape[1])
    return arr[r0:r1, c0:c1]


def _top_fraction_mask(abs_field: np.ndarray, frac: float = 0.08) -> tuple[np.ndarray, float]:
    vals = np.asarray(abs_field, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.zeros_like(abs_field, dtype=bool), float("nan")
    thr = float(np.quantile(vals, 1.0 - frac))
    mask = np.isfinite(abs_field) & (abs_field >= thr)
    return mask, thr


def _largest_component(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage as ndi
    structure = np.ones((3, 3), dtype=np.uint8)
    labeled, n_comp = ndi.label(mask.astype(np.uint8), structure=structure)
    if n_comp == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = ndi.sum(mask.astype(np.uint8), labeled, index=np.arange(1, n_comp + 1))
    largest_label = int(np.argmax(sizes)) + 1
    return labeled == largest_label


def _fit_circle_from_component(comp: np.ndarray) -> tuple[float, float, float]:
    ys, xs = np.nonzero(comp)
    if ys.size < 3:
        return float(comp.shape[0] / 2), float(comp.shape[1] / 2), 4.0
    A = np.column_stack([xs, ys, np.ones_like(xs)])
    b = -(xs ** 2 + ys ** 2)
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, beta, c = coef
    cx = float(-a / 2.0)
    cy = float(-beta / 2.0)
    rr = (a * a + beta * beta) / 4.0 - c
    radius = float(np.sqrt(max(rr, 1e-8)))
    return cy, cx, radius


def _crop_with_center(arr: np.ndarray, center_rc: tuple[float, float], half_size: int) -> tuple[np.ndarray, tuple[float, float]]:
    cy, cx = center_rc
    r0 = max(int(round(cy)) - half_size, 0)
    c0 = max(int(round(cx)) - half_size, 0)
    r1 = min(int(round(cy)) + half_size + 1, arr.shape[0])
    c1 = min(int(round(cx)) + half_size + 1, arr.shape[1])
    crop = arr[r0:r1, c0:c1]
    return crop, (cy - r0, cx - c0)


def add_distribution_panel(ax: plt.Axes, df: pd.DataFrame, metric: str, title: str, ylabel: str, letter: str) -> None:
    data = [df[df["source"] == src][metric].dropna().to_numpy(dtype=float) for src in POOL_ORDER]
    box = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops=dict(color="#1F1F1F", linewidth=1.5),
        whiskerprops=dict(color="#6E6E6E", linewidth=1.1),
        capprops=dict(color="#6E6E6E", linewidth=1.1),
    )
    rng = np.random.default_rng(7)
    for patch, src, vals, x0 in zip(box["boxes"], POOL_ORDER, data, range(1, len(POOL_ORDER) + 1)):
        color = POOL_COLORS[src]
        patch.set_facecolor(color)
        patch.set_alpha(0.28)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.4)
        x = rng.normal(x0, 0.045, size=len(vals))
        ax.scatter(x, vals, s=11, alpha=0.23, color=color, edgecolors="none", zorder=2)
    ax.axhline(0.0, color="#7F8C8D", lw=1.0, ls="--", alpha=0.9)
    ax.set_xticks(range(1, len(POOL_ORDER) + 1))
    ax.set_xticklabels([POOL_LABELS[src] for src in POOL_ORDER], fontsize=8.9)
    ax.set_ylabel(ylabel, fontsize=9.2)
    ax.set_title(f"({letter}) {title}", fontsize=10.6, pad=6)
    ax.grid(axis="y", alpha=0.18)
    ax.tick_params(axis="y", labelsize=8.7)
    ax.text(0.03, 0.96, "positive = better", transform=ax.transAxes,
            ha="left", va="top", fontsize=8.4,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.85, edgecolor="#D6D6D6"))


def _build_distribution_figure(df: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(8.6, 2.55))
    gs = fig.add_gridspec(1, 3, wspace=0.28)
    add_distribution_panel(fig.add_subplot(gs[0, 0]), df, "delta_rmse_gate",
                           r"Source-wise $\Delta$RMSE$_g$", r"$\Delta$RMSE$_g$ (deg/frame)", "a")
    add_distribution_panel(fig.add_subplot(gs[0, 1]), df, "delta_mae_gate",
                           r"Source-wise $\Delta$MAE$_g$", r"$\Delta$MAE$_g$ (deg/frame)", "b")
    add_distribution_panel(fig.add_subplot(gs[0, 2]), df, "delta_corr_gate",
                           r"Source-wise $\Delta$Corr$_g$", r"$\Delta$Corr$_g$", "c")
    return fig


def build_figure(pair_label: str = DEFAULT_PAIR_LABEL, copy_final: bool = False) -> Path:
    df = load_metrics_df()
    pair_stats = pair_metrics(df, pair_label)
    p1, p2 = find_pair_paths(pair_label)
    print(f"[real-case] loading {p1.name} / {p2.name}")
    i1 = load_gray(p1, max_size=512)
    i2 = load_gray(p2, max_size=512)

    print(f"[real-case] running final pipeline on {pair_label}")
    r = run_pair(i1, i2, gate_percentile=GATE_PERCENTILE)

    centers = np.asarray(r["centers"], dtype=int)
    omega_w = np.asarray(r["omega_w"], dtype=float)
    alpha_new = np.asarray(r["alpha_new"], dtype=float)
    snr_map = np.asarray(r["snr"], dtype=float)
    gate = np.asarray(r["gate"], dtype=bool)

    i1_crop = i1
    i2_crop = i2
    snr_crop = snr_map
    gate_crop = gate
    omega_g = np.where(gate, omega_w, np.nan)
    new_g = np.where(gate, alpha_new, np.nan)
    diff_g = np.where(gate, alpha_new - omega_w, np.nan)


    ring_mask, _ = _top_fraction_mask(np.abs(new_g), frac=0.08)
    ring_comp = _largest_component(ring_mask)
    ring_cy, ring_cx, ring_r = _fit_circle_from_component(ring_comp)
    roi_half = 12
    w_ring_crop, w_center_local = _crop_with_center(omega_g, (ring_cy, ring_cx), roi_half)
    n_ring_crop, n_center_local = _crop_with_center(new_g, (ring_cy, ring_cx), roi_half)

    vmax_field = max(p99_abs(np.concatenate([omega_g.ravel(), new_g.ravel()]), floor=2.0), 2.0)
    vmax_diff = max(p99_abs(diff_g, floor=0.5), 0.5)

    cmap_field = LinearSegmentedColormap.from_list(
        "bridge_vort_cmap",
        ["#FC757B", "#F97F5F", "#FAA26F", "#FDCD94",
         "#FEE199", "#B0D6A9", "#65BDBA", "#3C9BC9"]
    )
    cmap_field.set_bad("#ECE8E1")
    cmap_diff = plt.get_cmap("PuOr").copy()
    cmap_diff.set_bad("#ECE8E1")
    cmap_snr = plt.get_cmap("viridis").copy()

    fig = plt.figure(figsize=(10.8, 4.95))
    gs = fig.add_gridspec(
        2,
        4,
        width_ratios=[1.0, 1.0, 1.0, 0.055],
        height_ratios=[1.0, 1.0],
        hspace=0.26,
        wspace=0.16,
    )

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(i1_crop, cmap="gray", vmin=0, vmax=1)
    ax.set_title("(a) Frame $t$", fontsize=9.9)
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(i2_crop, cmap="gray", vmin=0, vmax=1)
    ax.set_title("(b) Frame $t+1$", fontsize=9.9)
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 2])
    im_ring_w = ax.imshow(w_ring_crop, cmap=cmap_field, vmin=-vmax_field, vmax=vmax_field, interpolation="nearest")
    circ_w = plt.Circle((w_center_local[1], w_center_local[0]), ring_r, fill=False, ec="black", lw=1.1, ls="--")
    ax.add_patch(circ_w)
    ax.set_title(r"(c) Local ring ROI in $\Omega_{\mathrm{WIDIM}}$", fontsize=8.4, pad=2)
    ax.set_ylabel("local-y", fontsize=7.5)
    ax.tick_params(labelsize=7.0, labelbottom=False)

    ax = fig.add_subplot(gs[1, 0])
    im_w = ax.imshow(omega_g, cmap=cmap_field, vmin=-vmax_field, vmax=vmax_field, interpolation="nearest")
    ax.set_title(r"(d) Gated reference $\Omega_{\mathrm{WIDIM}}$", fontsize=9.3)
    ax.set_xlabel("grid-x", fontsize=7.5)
    ax.set_ylabel("grid-y", fontsize=7.5)
    ax.tick_params(labelsize=7.0)

    ax2 = fig.add_subplot(gs[1, 1])
    im_n = ax2.imshow(new_g, cmap=cmap_field, vmin=-vmax_field, vmax=vmax_field, interpolation="nearest")
    ax2.set_title(r"(e) Gated proposed $\hat{\Omega}$", fontsize=9.3)
    ax2.set_xlabel("grid-x", fontsize=7.5)
    ax2.set_ylabel("grid-y", fontsize=7.5)
    ax2.tick_params(labelsize=7.0)

    ax3 = fig.add_subplot(gs[1, 2])
    im_d = ax3.imshow(n_ring_crop, cmap=cmap_field, vmin=-vmax_field, vmax=vmax_field, interpolation="nearest")
    circ_n = plt.Circle((n_center_local[1], n_center_local[0]), ring_r, fill=False, ec="black", lw=1.1, ls="--")
    ax3.add_patch(circ_n)
    ax3.set_title(r"(f) Local ring ROI in gated $\hat{\Omega}$", fontsize=8.3, pad=2)
    ax3.set_xlabel("local-x", fontsize=7.5)
    ax3.set_ylabel("local-y", fontsize=7.5)
    ax3.tick_params(labelsize=7.0)

    cax = fig.add_subplot(gs[:, 3])
    cbar_field = fig.colorbar(im_n, cax=cax)
    cbar_field.set_label(r"$\Omega$ (deg/frame)", fontsize=7.9)
    cbar_field.ax.tick_params(labelsize=7.0)

    out = OUT_DIR / f"fig16_real_proxy_case_{pair_label}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    dist_fig = _build_distribution_figure(df)
    dist_out = OUT_DIR / "figA16_real_proxy_distributions.png"
    dist_fig.savefig(dist_out, dpi=170, bbox_inches="tight", pad_inches=0.03)
    plt.close(dist_fig)

    print(f"[real-case] saved {out}")
    print(f"[real-case] saved {dist_out}")
    if pair_stats:
        print(
            f"[real-case] metrics {pair_label}: "
            f"dCorr={pair_stats['delta_corr_gate']:+.6f}, "
            f"dRMSE={pair_stats['delta_rmse_gate']:+.6f}, "
            f"dMAE={pair_stats['delta_mae_gate']:+.6f}"
        )

    if copy_final:
        for name in ("fig16_real_proxy_case.png", "fig10_real_proxy_case.png"):
            ieee_out = IEEE_DIR / name
            shutil.copy2(out, ieee_out)
            print(f"[real-case] copied to {ieee_out}")
        dist_ieee_out = IEEE_DIR / "figA16_real_proxy_distributions.png"
        shutil.copy2(dist_out, dist_ieee_out)
        print(f"[real-case] copied to {dist_ieee_out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build publication-quality real-data proxy figure")
    ap.add_argument("--pair-label", default=DEFAULT_PAIR_LABEL, help="PIVbook pair label, e.g. cam2_5001")
    ap.add_argument("--copy-final", action="store_true", help="Copy the rendered figure to the IEEE directory as fig16_real_proxy_case.png")
    args = ap.parse_args()
    build_figure(pair_label=args.pair_label, copy_final=args.copy_final)
