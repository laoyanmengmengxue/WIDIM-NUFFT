from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

from build_real_proxy_case_fig import (
    DEFAULT_PAIR_LABEL,
    GATE_PERCENTILE,
    METRICS_CSV,
    find_pair_paths,
    load_gray,
    pair_metrics,
    load_metrics_df,
    run_pair,
)


REPO = Path(__file__).resolve().parent
OUT_DIR = REPO / "results_real_structure_validation"
OUT_DIR.mkdir(exist_ok=True)

PAIR_LABEL = "cam2_5001"
TOP_FRACTIONS = [0.10, 0.08, 0.06]
N_ANGLE_BINS = 72
ROI_HALF_SIZE = 13


@dataclass
class StructureStats:
    top_fraction: float
    threshold: float
    n_active: int
    n_components: int
    largest_component_pixels: int
    largest_component_fraction: float
    angular_coverage: float
    radial_cv: float
    ring_coherence: float
    centroid_y: float
    centroid_x: float


@dataclass
class AnnulusStats:
    rel_threshold: float
    annulus_mean: float
    background_mean: float
    annulus_contrast: float
    angular_coverage: float
    longest_arc_fraction: float
    active_pixels: int


def top_fraction_mask(abs_field: np.ndarray, frac: float) -> tuple[np.ndarray, float]:
    vals = np.asarray(abs_field, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.zeros_like(abs_field, dtype=bool), float("nan")
    threshold = float(np.quantile(vals, 1.0 - frac))
    mask = np.isfinite(abs_field) & (abs_field >= threshold)
    return mask, threshold


def _largest_component(mask: np.ndarray) -> tuple[np.ndarray, int]:
    structure = np.ones((3, 3), dtype=np.uint8)
    labeled, n_comp = ndi.label(mask.astype(np.uint8), structure=structure)
    if n_comp == 0:
        return np.zeros_like(mask, dtype=bool), 0
    sizes = ndi.sum(mask.astype(np.uint8), labeled, index=np.arange(1, n_comp + 1))
    largest_label = int(np.argmax(sizes)) + 1
    comp = labeled == largest_label
    return comp, int(n_comp)


def component_stats(abs_field: np.ndarray, frac: float) -> tuple[StructureStats, np.ndarray]:
    mask, thr = top_fraction_mask(abs_field, frac)
    comp, n_comp = _largest_component(mask)
    ys, xs = np.nonzero(comp)

    if ys.size == 0:
        stats = StructureStats(
            top_fraction=frac,
            threshold=thr,
            n_active=int(mask.sum()),
            n_components=0,
            largest_component_pixels=0,
            largest_component_fraction=0.0,
            angular_coverage=0.0,
            radial_cv=float("nan"),
            ring_coherence=0.0,
            centroid_y=float("nan"),
            centroid_x=float("nan"),
        )
        return stats, comp

    cy = float(np.mean(ys))
    cx = float(np.mean(xs))
    radii = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    r_mean = float(np.mean(radii)) if radii.size else float("nan")
    r_std = float(np.std(radii)) if radii.size else float("nan")
    radial_cv = float(r_std / r_mean) if r_mean > 1e-8 else float("nan")

    theta = np.mod(np.arctan2(ys - cy, xs - cx), 2.0 * np.pi)
    angle_bins = np.floor(theta / (2.0 * np.pi) * N_ANGLE_BINS).astype(int)
    angle_bins = np.clip(angle_bins, 0, N_ANGLE_BINS - 1)
    angular_coverage = float(np.unique(angle_bins).size / N_ANGLE_BINS)

    n_active = int(mask.sum())
    largest_pixels = int(comp.sum())
    largest_frac = float(largest_pixels / max(n_active, 1))

    if not np.isfinite(radial_cv):
        ring_coherence = 0.0
    else:
        ring_coherence = float(largest_frac * angular_coverage / max(radial_cv, 1e-6))

    stats = StructureStats(
        top_fraction=frac,
        threshold=thr,
        n_active=n_active,
        n_components=int(n_comp),
        largest_component_pixels=largest_pixels,
        largest_component_fraction=largest_frac,
        angular_coverage=angular_coverage,
        radial_cv=radial_cv,
        ring_coherence=ring_coherence,
        centroid_y=cy,
        centroid_x=cx,
    )
    return stats, comp


def summarize_structure(abs_field: np.ndarray) -> tuple[list[StructureStats], dict[str, float], StructureStats, np.ndarray]:
    stats_list: list[StructureStats] = []
    dominant_stats: StructureStats | None = None
    dominant_comp: np.ndarray | None = None

    for frac in TOP_FRACTIONS:
        stats, comp = component_stats(abs_field, frac)
        stats_list.append(stats)
        if dominant_stats is None and np.isfinite(stats.centroid_x):
            dominant_stats = stats
            dominant_comp = comp

    arr_frac = np.array([s.largest_component_fraction for s in stats_list], dtype=float)
    arr_cov = np.array([s.angular_coverage for s in stats_list], dtype=float)
    arr_cv = np.array([s.radial_cv for s in stats_list], dtype=float)
    arr_score = np.array([s.ring_coherence for s in stats_list], dtype=float)
    arr_n = np.array([s.n_components for s in stats_list], dtype=float)

    summary = {
        "largest_component_fraction_mean": float(np.nanmean(arr_frac)),
        "angular_coverage_mean": float(np.nanmean(arr_cov)),
        "radial_cv_mean": float(np.nanmean(arr_cv)),
        "ring_coherence_mean": float(np.nanmean(arr_score)),
        "n_components_mean": float(np.nanmean(arr_n)),
    }

    if dominant_stats is None:
        dominant_stats = stats_list[0]
        dominant_comp = np.zeros_like(abs_field, dtype=bool)

    return stats_list, summary, dominant_stats, dominant_comp


def fit_circle_from_component(comp: np.ndarray) -> tuple[float, float, float]:
    ys, xs = np.nonzero(comp)
    if ys.size < 3:
        return float("nan"), float("nan"), float("nan")
    A = np.column_stack([xs, ys, np.ones_like(xs)])
    b = -(xs ** 2 + ys ** 2)
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, beta, c = coef
    cx = float(-a / 2.0)
    cy = float(-beta / 2.0)
    rr = (a * a + beta * beta) / 4.0 - c
    radius = float(np.sqrt(max(rr, 1e-8)))
    return cy, cx, radius


def longest_circular_run(occ: np.ndarray) -> int:
    occ2 = np.concatenate([occ.astype(int), occ.astype(int)])
    best = 0
    cur = 0
    for v in occ2:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(min(best, occ.size))


def annulus_stats(
    abs_field: np.ndarray,
    center_rc: tuple[float, float],
    radius: float,
    rel_thresholds: list[float] | tuple[float, ...] = (0.30, 0.40, 0.50),
    annulus_halfwidth: float = 1.5,
    background_gap: float = 3.0,
    background_width: float = 3.0,
) -> tuple[list[AnnulusStats], dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    y, x = np.indices(abs_field.shape)
    cy, cx = center_rc
    rr = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    theta = np.mod(np.arctan2(y - cy, x - cx), 2.0 * np.pi)

    annulus = np.isfinite(abs_field) & (rr >= radius - annulus_halfwidth) & (rr <= radius + annulus_halfwidth)
    bg_outer = np.isfinite(abs_field) & (rr >= radius + background_gap) & (rr <= radius + background_gap + background_width)
    inner_lo = max(radius - background_gap - background_width, 0.0)
    inner_hi = max(radius - background_gap, 0.0)
    bg_inner = np.isfinite(abs_field) & (rr >= inner_lo) & (rr <= inner_hi)
    background = bg_outer | bg_inner

    vals_ann = abs_field[annulus]
    ann_mean = float(np.nanmean(vals_ann)) if vals_ann.size else float("nan")
    bg_mean = float(np.nanmean(abs_field[background])) if np.any(background) else float("nan")
    contrast = float(ann_mean / max(bg_mean, 1e-6)) if np.isfinite(ann_mean) and np.isfinite(bg_mean) else float("nan")

    stats_list: list[AnnulusStats] = []
    cov_arr = []
    long_arr = []
    for rel in rel_thresholds:
        thr = float(rel * np.nanmax(vals_ann)) if vals_ann.size else float("nan")
        active = annulus & np.isfinite(abs_field) & (abs_field >= thr)
        angle_bins = np.floor(theta[active] / (2.0 * np.pi) * N_ANGLE_BINS).astype(int)
        angle_bins = np.clip(angle_bins, 0, N_ANGLE_BINS - 1)
        occ = np.zeros(N_ANGLE_BINS, dtype=int)
        if angle_bins.size:
            occ[np.unique(angle_bins)] = 1
        cov = float(occ.sum() / N_ANGLE_BINS)
        longest = float(longest_circular_run(occ) / N_ANGLE_BINS)
        cov_arr.append(cov)
        long_arr.append(longest)
        stats_list.append(
            AnnulusStats(
                rel_threshold=rel,
                annulus_mean=ann_mean,
                background_mean=bg_mean,
                annulus_contrast=contrast,
                angular_coverage=cov,
                longest_arc_fraction=longest,
                active_pixels=int(active.sum()),
            )
        )

    summary = {
        "annulus_mean": ann_mean,
        "background_mean": bg_mean,
        "annulus_contrast": contrast,
        "angular_coverage_mean": float(np.nanmean(np.asarray(cov_arr, dtype=float))),
        "longest_arc_fraction_mean": float(np.nanmean(np.asarray(long_arr, dtype=float))),
    }
    return stats_list, summary, annulus, background, rr


def crop_rc(arr: np.ndarray, center_rc: tuple[float, float], half_size: int) -> np.ndarray:
    cy, cx = center_rc
    r0 = max(int(round(cy)) - half_size, 0)
    c0 = max(int(round(cx)) - half_size, 0)
    r1 = min(int(round(cy)) + half_size + 1, arr.shape[0])
    c1 = min(int(round(cx)) + half_size + 1, arr.shape[1])
    return arr[r0:r1, c0:c1]


def radial_profile(abs_field: np.ndarray, center_rc: tuple[float, float], r_max: int = 16) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.indices(abs_field.shape)
    cy, cx = center_rc
    rr = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    radii = np.arange(0, r_max + 1, dtype=int)
    prof = np.full_like(radii, np.nan, dtype=float)
    for i, r in enumerate(radii):
        sel = (rr >= r) & (rr < r + 1) & np.isfinite(abs_field)
        if np.any(sel):
            prof[i] = float(np.mean(abs_field[sel]))
    return radii.astype(float), prof


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def build_figure(
    img1: np.ndarray,
    img2: np.ndarray,
    omega_w: np.ndarray,
    omega_p: np.ndarray,
    comp_w: np.ndarray,
    comp_p: np.ndarray,
    center_rc: tuple[float, float],
    radius: float,
    summary_w: dict[str, float],
    summary_p: dict[str, float],
    ann_summary_w: dict[str, float],
    ann_summary_p: dict[str, float],
    pair_info: dict,
    out_path: Path,
) -> None:
    avg_img = 0.5 * (img1 + img2)
    abs_w = np.abs(omega_w)
    abs_p = np.abs(omega_p)

    img_crop = crop_rc(avg_img, center_rc, ROI_HALF_SIZE * 6)
    w_crop = crop_rc(abs_w, center_rc, ROI_HALF_SIZE + 2)
    p_crop = crop_rc(abs_p, center_rc, ROI_HALF_SIZE + 2)
    mw_crop = crop_rc(comp_w.astype(float), center_rc, ROI_HALF_SIZE + 2)
    mp_crop = crop_rc(comp_p.astype(float), center_rc, ROI_HALF_SIZE + 2)

    radii, prof_w = radial_profile(abs_w, center_rc, r_max=16)
    _, prof_p = radial_profile(abs_p, center_rc, r_max=16)

    vmax = max(np.nanpercentile(abs_w, 99.0), np.nanpercentile(abs_p, 99.0), 1e-6)

    fig = plt.figure(figsize=(11.8, 6.5))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.0, 1.05], width_ratios=[1.20, 1.0, 1.0, 1.0, 1.05])

    def _overlay_circle(ax: plt.Axes, arr: np.ndarray) -> None:
        cy, cx = center_rc
        r0 = max(int(round(cy)) - (ROI_HALF_SIZE + 2), 0)
        c0 = max(int(round(cx)) - (ROI_HALF_SIZE + 2), 0)
        cy_loc = cy - r0
        cx_loc = cx - c0
        circ = plt.Circle((cx_loc, cy_loc), radius, fill=False, ec="white", lw=1.2, ls="--")
        ax.add_patch(circ)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img_crop, cmap="gray")
    ax.set_title("Average raw image", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(w_crop, cmap="magma", vmin=0.0, vmax=vmax)
    ax.set_title(r"WIDIM $|\Omega|$", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    _overlay_circle(ax, w_crop)

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(p_crop, cmap="magma", vmin=0.0, vmax=vmax)
    ax.set_title(r"Final $|\hat{\Omega}|$", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    _overlay_circle(ax, p_crop)

    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(mw_crop, cmap="gray_r", vmin=0, vmax=1)
    ax.set_title("WIDIM dominant mask", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 4])
    ax.imshow(mp_crop, cmap="gray_r", vmin=0, vmax=1)
    ax.set_title("Final dominant mask", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    cbar = fig.colorbar(im, ax=[fig.axes[1], fig.axes[2]], fraction=0.030, pad=0.03)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(r"$|\Omega|$ (deg/frame)", fontsize=9)

    ax = fig.add_subplot(gs[1, 0:2])
    ax.plot(radii, prof_w, lw=2.0, color="#6A8FBF", label="WIDIM")
    ax.plot(radii, prof_p, lw=2.0, color="#D9584A", label="Final")
    ax.set_xlabel("radius (grid cells)", fontsize=9)
    ax.set_ylabel(r"mean $|\Omega|$", fontsize=9)
    ax.set_title("Radial profile around fitted ring center", fontsize=10)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=8)
    ax.legend(frameon=False, fontsize=8)

    metrics = [
        ("annulus contrast", ann_summary_w["annulus_contrast"], ann_summary_p["annulus_contrast"]),
        ("ann. coverage", ann_summary_w["angular_coverage_mean"], ann_summary_p["angular_coverage_mean"]),
        ("longest arc", ann_summary_w["longest_arc_fraction_mean"], ann_summary_p["longest_arc_fraction_mean"]),
        ("largest comp.", summary_w["largest_component_fraction_mean"], summary_p["largest_component_fraction_mean"]),
        ("n components", summary_w["n_components_mean"], summary_p["n_components_mean"]),
    ]
    ax = fig.add_subplot(gs[1, 2:4])
    y = np.arange(len(metrics))
    widim_vals = [m[1] for m in metrics]
    final_vals = [m[2] for m in metrics]
    h = 0.34
    ax.barh(y + h / 2, widim_vals, height=h, color="#6A8FBF", label="WIDIM")
    ax.barh(y - h / 2, final_vals, height=h, color="#D9584A", label="Final")
    ax.set_yticks(y)
    ax.set_yticklabels([m[0] for m in metrics], fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title("Ring-centric and global structure diagnostics", fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.invert_yaxis()

    ax = fig.add_subplot(gs[1, 4])
    ax.axis("off")
    txt = (
        f"Representative pair: {pair_info['label']}\n"
        f"Source: {pair_info['source']}\n\n"
        f"Gated metrics from full-pair run:\n"
        f"ΔCorr_gate = {pair_info['delta_corr_gate']:.4f}\n"
        f"ΔRMSE_gate = {pair_info['delta_rmse_gate']:.3f}\n"
        f"ΔMAE_gate = {pair_info['delta_mae_gate']:.3f}\n\n"
        f"Ring-centric diagnostics:\n"
        f"Annulus contrast: {ann_summary_w['annulus_contrast']:.3f}"
        f" → {ann_summary_p['annulus_contrast']:.3f}\n"
        f"Angular coverage: {fmt_pct(ann_summary_w['angular_coverage_mean'])}"
        f" → {fmt_pct(ann_summary_p['angular_coverage_mean'])}\n"
        f"Longest arc: {fmt_pct(ann_summary_w['longest_arc_fraction_mean'])}"
        f" → {fmt_pct(ann_summary_p['longest_arc_fraction_mean'])}\n\n"
        f"Global top-f diagnostic:\n"
        f"Largest comp.: {fmt_pct(summary_w['largest_component_fraction_mean'])}"
        f" → {fmt_pct(summary_p['largest_component_fraction_mean'])}\n"
        f"n components: {summary_w['n_components_mean']:.1f}"
        f" → {summary_p['n_components_mean']:.1f}"
    )
    ax.text(
        0.0,
        1.0,
        txt,
        va="top",
        ha="left",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FAFAFA", edgecolor="#CCCCCC"),
    )

    fig.suptitle(
        "Independent structure validation on real vortex-ring case cam2_5001",
        fontsize=12,
        y=0.98,
    )
    fig.subplots_adjust(left=0.04, right=0.985, top=0.90, bottom=0.08, wspace=0.32, hspace=0.36)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def append_report(
    txt_path: Path,
    pair_info: dict,
    summary_w: dict[str, float],
    summary_p: dict[str, float],
    ann_summary_w: dict[str, float],
    ann_summary_p: dict[str, float],
    ann_stats_w: list[AnnulusStats],
    ann_stats_p: list[AnnulusStats],
    w_stats: list[StructureStats],
    p_stats: list[StructureStats],
    fig_path: Path,
    json_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("")
    lines.append("")
    lines.append("独立结构验证：cam2_5001（不以 WIDIM 为评价参考场）")
    lines.append("=" * 90)
    lines.append("目标：验证最终协议在真实涡环场景中是否恢复出更连续、更少碎裂、更接近环结构的主旋转组织。")
    lines.append(f"pair={pair_info['label']} | source={pair_info['source']} | gate_p={GATE_PERCENTILE}%")
    lines.append(
        "full-pair gated gains: "
        f"ΔCorr_gate={pair_info['delta_corr_gate']:.6f}, "
        f"ΔRMSE_gate={pair_info['delta_rmse_gate']:.6f}, "
        f"ΔMAE_gate={pair_info['delta_mae_gate']:.6f}"
    )
    lines.append("")
    lines.append("[全场 top-f% 连通域诊断（多阈值均值）]")
    lines.append(
        "WIDIM: "
        f"largest_frac={summary_w['largest_component_fraction_mean']:.4f}, "
        f"coverage={summary_w['angular_coverage_mean']:.4f}, "
        f"radial_cv={summary_w['radial_cv_mean']:.4f}, "
        f"ring_score={summary_w['ring_coherence_mean']:.4f}, "
        f"n_comp={summary_w['n_components_mean']:.2f}"
    )
    lines.append(
        "Final: "
        f"largest_frac={summary_p['largest_component_fraction_mean']:.4f}, "
        f"coverage={summary_p['angular_coverage_mean']:.4f}, "
        f"radial_cv={summary_p['radial_cv_mean']:.4f}, "
        f"ring_score={summary_p['ring_coherence_mean']:.4f}, "
        f"n_comp={summary_p['n_components_mean']:.2f}"
    )
    lines.append("")
    lines.append("[拟合环带诊断（以 final 主弧拟合候选圆，不以 WIDIM 作为评价参考场）]")
    lines.append(
        "WIDIM: "
        f"annulus_mean={ann_summary_w['annulus_mean']:.4f}, "
        f"background_mean={ann_summary_w['background_mean']:.4f}, "
        f"contrast={ann_summary_w['annulus_contrast']:.4f}, "
        f"coverage={ann_summary_w['angular_coverage_mean']:.4f}, "
        f"longest_arc={ann_summary_w['longest_arc_fraction_mean']:.4f}"
    )
    lines.append(
        "Final: "
        f"annulus_mean={ann_summary_p['annulus_mean']:.4f}, "
        f"background_mean={ann_summary_p['background_mean']:.4f}, "
        f"contrast={ann_summary_p['annulus_contrast']:.4f}, "
        f"coverage={ann_summary_p['angular_coverage_mean']:.4f}, "
        f"longest_arc={ann_summary_p['longest_arc_fraction_mean']:.4f}"
    )
    lines.append("")
    lines.append("[逐阈值明细：WIDIM]")
    for s in w_stats:
        lines.append(
            f"  top={int(s.top_fraction*100):02d}% | thr={s.threshold:.4f} | active={s.n_active} | "
            f"n_comp={s.n_components} | largest={s.largest_component_fraction:.4f} | "
            f"coverage={s.angular_coverage:.4f} | radial_cv={s.radial_cv:.4f} | ring_score={s.ring_coherence:.4f}"
        )
    lines.append("[逐阈值明细：Final]")
    for s in p_stats:
        lines.append(
            f"  top={int(s.top_fraction*100):02d}% | thr={s.threshold:.4f} | active={s.n_active} | "
            f"n_comp={s.n_components} | largest={s.largest_component_fraction:.4f} | "
            f"coverage={s.angular_coverage:.4f} | radial_cv={s.radial_cv:.4f} | ring_score={s.ring_coherence:.4f}"
        )
    lines.append("[逐阈值明细：环带诊断 WIDIM]")
    for s in ann_stats_w:
        lines.append(
            f"  rel={s.rel_threshold:.2f} | ann={s.annulus_mean:.4f} | bg={s.background_mean:.4f} | "
            f"contrast={s.annulus_contrast:.4f} | coverage={s.angular_coverage:.4f} | "
            f"longest_arc={s.longest_arc_fraction:.4f} | active={s.active_pixels}"
        )
    lines.append("[逐阈值明细：环带诊断 Final]")
    for s in ann_stats_p:
        lines.append(
            f"  rel={s.rel_threshold:.2f} | ann={s.annulus_mean:.4f} | bg={s.background_mean:.4f} | "
            f"contrast={s.annulus_contrast:.4f} | coverage={s.angular_coverage:.4f} | "
            f"longest_arc={s.longest_arc_fraction:.4f} | active={s.active_pixels}"
        )
    lines.append("")
    lines.append("[结论判断]")
    ring_msgs = []
    if ann_summary_p["annulus_contrast"] > ann_summary_w["annulus_contrast"]:
        ring_msgs.append("候选环带上的能量-背景对比更高")
    if ann_summary_p["angular_coverage_mean"] > ann_summary_w["angular_coverage_mean"]:
        ring_msgs.append("候选环带上的角向覆盖略高")
    if ann_summary_p["longest_arc_fraction_mean"] > ann_summary_w["longest_arc_fraction_mean"]:
        ring_msgs.append("候选环带上的最长连续弧更长")

    global_msgs = []
    if summary_p["largest_component_fraction_mean"] > summary_w["largest_component_fraction_mean"]:
        global_msgs.append("全场主连通域占比更大")
    if summary_p["n_components_mean"] < summary_w["n_components_mean"]:
        global_msgs.append("全场碎裂度更低")

    if ring_msgs and not global_msgs:
        lines.append("structure_status=PARTIAL_SUPPORT")
        lines.append("判定理由：全场 top-f% 连通域诊断并不支持 final，但围绕拟合候选环带的局部结构诊断支持 final。")
        lines.append("局部支持点：" + "；".join(ring_msgs) + "。")
    elif ring_msgs and global_msgs:
        lines.append("structure_status=SUCCESS")
        lines.append("判定理由：全场与环带两类结构诊断均支持 final。")
    else:
        lines.append("structure_status=INCONCLUSIVE")
        lines.append("判定理由：局部环带诊断与全场连通域诊断均未形成稳定同向优势。")
    lines.append(f"figure={fig_path}")
    lines.append(f"json={json_path}")
    marker = "独立结构验证：cam2_5001（不以 WIDIM 为评价参考场）"
    old = txt_path.read_text(encoding="utf-8-sig") if txt_path.exists() else ""
    if marker in old:
        old = old.split(marker)[0].rstrip()
    new_text = old + ("\n\n" if old else "") + "\n".join(lines)
    txt_path.write_text(new_text, encoding="utf-8-sig")


def main() -> None:
    metrics_df = load_metrics_df()
    pair_info = pair_metrics(metrics_df, PAIR_LABEL)
    p1, p2 = find_pair_paths(PAIR_LABEL)
    img1 = load_gray(p1)
    img2 = load_gray(p2)

    print(f"Running independent structure validation on {PAIR_LABEL}")
    result = run_pair(img1, img2, GATE_PERCENTILE)

    omega_w = np.asarray(result["omega_w"], dtype=float)
    omega_p = np.asarray(result["alpha_new"], dtype=float)

    w_stats, w_summary, _, comp_w = summarize_structure(np.abs(omega_w))
    p_stats, p_summary, p_dominant, comp_p = summarize_structure(np.abs(omega_p))
    center_rc = (p_dominant.centroid_y, p_dominant.centroid_x)
    fit_center_y, fit_center_x, fit_radius = fit_circle_from_component(comp_p)
    fit_center = (fit_center_y, fit_center_x)
    ann_stats_w, ann_summary_w, _, _, _ = annulus_stats(np.abs(omega_w), fit_center, fit_radius)
    ann_stats_p, ann_summary_p, _, _, _ = annulus_stats(np.abs(omega_p), fit_center, fit_radius)

    fig_path = OUT_DIR / "structure_validation_cam2_5001.png"
    json_path = OUT_DIR / "structure_validation_cam2_5001.json"
    build_figure(
        img1,
        img2,
        omega_w,
        omega_p,
        comp_w,
        comp_p,
        fit_center,
        fit_radius,
        w_summary,
        p_summary,
        ann_summary_w,
        ann_summary_p,
        pair_info,
        fig_path,
    )

    payload = {
        "pair_info": pair_info,
        "widim_threshold_stats": [asdict(s) for s in w_stats],
        "final_threshold_stats": [asdict(s) for s in p_stats],
        "ring_fit": {
            "center_y": fit_center_y,
            "center_x": fit_center_x,
            "radius": fit_radius,
        },
        "widim_annulus_stats": [asdict(s) for s in ann_stats_w],
        "final_annulus_stats": [asdict(s) for s in ann_stats_p],
        "widim_summary": w_summary,
        "final_summary": p_summary,
        "widim_annulus_summary": ann_summary_w,
        "final_annulus_summary": ann_summary_p,
        "figure": str(fig_path),
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    append_report(
        REPO / "桥接实验.txt",
        pair_info,
        w_summary,
        p_summary,
        ann_summary_w,
        ann_summary_p,
        ann_stats_w,
        ann_stats_p,
        w_stats,
        p_stats,
        fig_path,
        json_path,
    )
    print(f"Saved figure to {fig_path}")
    print(f"Saved metrics to {json_path}")


if __name__ == "__main__":
    main()
