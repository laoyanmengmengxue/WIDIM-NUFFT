from __future__ import annotations

import shutil
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
IEEE_DIR = ROOT / "IEEE-Transactions-LaTeX2e-templates-and-instructions"
OUT_DIR = IEEE_DIR


FLOW_COLORS = {
    "Rankine": "#1f77b4",
    "LambOseen": "#2ca02c",
    "SolidRotation": "#ff7f0e",
    "LinearShear": "#d62728",
}

FLOW_TITLES = {
    "Rankine": "Rankine",
    "LambOseen": "Lamb-Oseen",
    "SolidRotation": "Solid rotation",
    "LinearShear": "Linear shear",
}

SOURCE_COLORS = {
    "PIV_C": "#2c7fb8",
    "PIV_E": "#fdae61",
    "PIV_book": "#4dac26",
}

VERSION_LABELS = ["V0", "V1", "V2", "V3"]
VERSION_TITLES = [
    "V0 fixed-window",
    "V1 affine prior",
    "V2 track prior",
    "V3 final protocol",
]


def _find(name: str) -> Path:
    return next(ROOT.rglob(name))


def _save(fig: plt.Figure, name: str) -> None:
    out = OUT_DIR / name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _sign_agreement(x: pd.Series, y: pd.Series) -> float:
    return float((((x > 0) == (y > 0)).mean()) * 100.0)


def _parse_gt_results(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    current: str | None = None
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if line.startswith("Flow: "):
            current = (
                line.replace("Flow: ", "")
                .replace(" Vortex", "")
                .replace(" ", "")
                .replace("-", "")
            )
            out[current] = {}
        elif current and line.startswith("NUFFT RMSE:"):
            match = re.search(r"([-+]?\d+(?:\.\d+)?)", line.split(":", 1)[1])
            if match:
                out[current]["variant_rmse"] = float(match.group(1))
        elif current and line.startswith("WIDIM RMSE:"):
            match = re.search(r"([-+]?\d+(?:\.\d+)?)", line.split(":", 1)[1])
            if match:
                out[current]["widim_rmse"] = float(match.group(1))
        elif current and line.startswith("Improvement:"):
            match = re.search(r"([-+]?\d+(?:\.\d+)?)", line.split(":", 1)[1])
            if match:
                out[current]["gain"] = float(match.group(1))
    return out


def build_fig_a1() -> None:
    src = IEEE_DIR / "fig08_disp_crlb.png"
    dst = IEEE_DIR / "figA1_disp_crlb_reference.png"
    shutil.copyfile(src, dst)


def build_fig_a2() -> None:
    df = pd.read_csv(_find("synthetic_proxy_calibration_2026-03-16.csv"))

    panels = [
        ("delta_rmse_proxy", r"$\Delta \mathrm{RMSE}_g$ ($^\circ$/frame)"),
        ("delta_mae_proxy", r"$\Delta \mathrm{MAE}_g$ ($^\circ$/frame)"),
        ("delta_corr_proxy", r"$\Delta \mathrm{Corr}_g$"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.2))
    y = df["delta_rmse_true"].to_numpy()

    for ax, (col, xlabel) in zip(axes, panels):
        x = df[col].to_numpy()
        for flow, sub in df.groupby("flow"):
            ax.scatter(
                sub[col],
                sub["delta_rmse_true"],
                s=26,
                alpha=0.82,
                color=FLOW_COLORS.get(flow, "#333333"),
                edgecolor="white",
                linewidth=0.35,
                label=flow,
            )

        coef = np.polyfit(x, y, 1)
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 200)
        ax.plot(xs, coef[0] * xs + coef[1], color="#222222", lw=1.3, alpha=0.9)
        ax.axhline(0.0, color="#999999", lw=0.8, ls="--")
        if col != "delta_corr_proxy":
            ax.axvline(0.0, color="#999999", lw=0.8, ls="--")

        corr = pd.Series(x).corr(pd.Series(y))
        sign = _sign_agreement(pd.Series(x), pd.Series(y))
        ax.text(
            0.03,
            0.97,
            f"Pearson = {corr:.3f}\nSign = {sign:.1f}%",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#bbbbbb", alpha=0.95),
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\Delta \mathrm{RMSE}_{true}$ ($^\circ$/frame)")
        ax.grid(alpha=0.22, lw=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        frameon=False,
    )
    fig.subplots_adjust(top=0.74, wspace=0.28)
    _save(fig, "figA2_proxy_calibration.png")


def build_fig_a12() -> None:
    df = pd.read_csv(_find("synthetic_proxy_calibration_2026-03-16.csv"))
    order = ["Rankine", "LambOseen", "SolidRotation", "LinearShear"]
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.2))

    for ax, flow in zip(axes.ravel(), order):
        sub = df[df["flow"] == flow].copy()
        x = sub["delta_rmse_proxy"]
        y = sub["delta_rmse_true"]
        color = FLOW_COLORS[flow]

        ax.scatter(
            x,
            y,
            s=40,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            alpha=0.88,
        )

        xmin = float(min(x.min(), y.min()))
        xmax = float(max(x.max(), y.max()))
        pad = 0.08 * (xmax - xmin if xmax > xmin else 1.0)
        lo, hi = xmin - pad, xmax + pad
        xs = np.linspace(lo, hi, 200)
        ax.plot(xs, xs, color="#555555", lw=1.0, ls="--", alpha=0.75)

        coef = np.polyfit(x, y, 1)
        ax.plot(xs, coef[0] * xs + coef[1], color="#111111", lw=1.3, alpha=0.9)
        ax.axhline(0.0, color="#999999", lw=0.8, ls=":")
        ax.axvline(0.0, color="#999999", lw=0.8, ls=":")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(FLOW_TITLES[flow], fontsize=11)
        ax.set_xlabel(r"$\Delta \mathrm{RMSE}_g$ ($^\circ$/frame)")
        ax.set_ylabel(r"$\Delta \mathrm{RMSE}_{\mathrm{true}}$ ($^\circ$/frame)")
        ax.grid(alpha=0.22, lw=0.5)

        pear = x.corr(y, method="pearson")
        spear = x.corr(y, method="spearman")
        sign = _sign_agreement(x, y)
        ax.text(
            0.04,
            0.96,
            f"Pearson = {pear:.3f}\nSpearman = {spear:.3f}\nSign = {sign:.1f}%",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#bbbbbb", alpha=0.95),
        )

    fig.suptitle(
        r"Flow-wise calibration of the primary proxy $\Delta \mathrm{RMSE}_g$",
        fontsize=12.2,
        y=0.99,
    )
    fig.subplots_adjust(top=0.91, wspace=0.26, hspace=0.28)
    _save(fig, "figA12_proxy_by_flow.png")


def build_fig_a13() -> None:
    df = pd.read_csv(_find("synthetic_proxy_calibration_2026-03-16.csv"))
    sub = df[df["flow"] == "Rankine"].copy()

    panels = [
        ("delta_rmse_proxy", r"$\Delta \mathrm{RMSE}_g$ ($^\circ$/frame)", "#1f77b4"),
        ("delta_mae_proxy", r"$\Delta \mathrm{MAE}_g$ ($^\circ$/frame)", "#ff7f0e"),
        ("delta_corr_proxy", r"$\Delta \mathrm{Corr}_g$", "#2ca02c"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0))
    y = sub["delta_rmse_true"]

    for ax, (col, xlabel, color) in zip(axes, panels):
        x = sub[col]
        ax.scatter(
            x,
            y,
            s=42,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            alpha=0.9,
        )
        coef = np.polyfit(x, y, 1)
        xs = np.linspace(float(x.min()), float(x.max()), 200)
        ax.plot(xs, coef[0] * xs + coef[1], color="#111111", lw=1.3)
        ax.axhline(0.0, color="#999999", lw=0.8, ls=":")
        if col != "delta_corr_proxy":
            ax.axvline(0.0, color="#999999", lw=0.8, ls=":")
        pear = x.corr(y, method="pearson")
        spear = x.corr(y, method="spearman")
        sign = _sign_agreement(x, y)
        ax.text(
            0.04,
            0.96,
            f"Pearson = {pear:.3f}\nSpearman = {spear:.3f}\nSign = {sign:.1f}%",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.8,
            bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="#bbbbbb", alpha=0.95),
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\Delta \mathrm{RMSE}_{\mathrm{true}}$ ($^\circ$/frame)")
        ax.grid(alpha=0.22, lw=0.5)

    fig.suptitle(
        "Rankine-only proxy-to-truth calibration (same scenario as the primary main-text experiment)",
        fontsize=12.0,
        y=1.02,
    )
    fig.subplots_adjust(top=0.79, wspace=0.28)
    _save(fig, "figA13_rankine_proxy_triplet.png")


def build_fig_a3() -> None:
    x = np.linspace(-1.0, 1.0, 9)
    y = np.linspace(-1.0, 1.0, 9)
    X, Y = np.meshgrid(x, y)

    tx, ty = 0.85, 0.35
    omega = 0.9
    U = tx - omega * Y
    V = ty + omega * X

    U_shift = U - tx
    V_shift = V - ty

    U_affine = np.zeros_like(U)
    V_affine = np.zeros_like(V)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    titles = [
        "Observed local motion",
        "After translation-only alignment",
        "After affine de-rotation",
    ]
    fields = [(U, V), (U_shift, V_shift), (U_affine, V_affine)]
    subtitles = [
        "translation + rotation",
        "rotation signal preserved",
        "rotation signal attenuated",
    ]

    for ax, title, subtitle, (uu, vv) in zip(axes, titles, subtitles, fields):
        mag = np.sqrt(uu**2 + vv**2)
        ax.quiver(
            X,
            Y,
            uu,
            vv,
            mag,
            cmap="YlGnBu",
            angles="xy",
            scale_units="xy",
            scale=4.5,
            width=0.012,
        )
        ax.set_title(title, fontsize=11, pad=10)
        ax.text(0.5, -0.12, subtitle, transform=ax.transAxes, ha="center", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#777777")
            spine.set_linewidth(0.8)

    fig.subplots_adjust(wspace=0.18)
    _save(fig, "figA3_translation_vs_affine.png")


def build_fig_a4() -> None:
    df = pd.read_csv(_find("metrics_all_pairs.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.4))

    order = ["PIV_C", "PIV_E", "PIV_book"]
    data = [df.loc[df["source"] == src, "snr_thr_p"].to_numpy() for src in order]
    box = axes[0].boxplot(
        data,
        patch_artist=True,
        tick_labels=order,
        widths=0.55,
        medianprops=dict(color="#222222", linewidth=1.2),
        boxprops=dict(linewidth=1.0),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
    )
    for patch, src in zip(box["boxes"], order):
        patch.set_facecolor(SOURCE_COLORS[src])
        patch.set_alpha(0.75)
        patch.set_edgecolor("#555555")

    axes[0].axhline(2.2, color="#cc2f27", linestyle="--", linewidth=1.2, label="example fixed threshold = 2.2")
    axes[0].set_ylabel(r"pair-wise p5 SNR threshold")
    axes[0].set_title("Per-pair adaptive thresholds vary by data source")
    axes[0].grid(axis="y", alpha=0.22, lw=0.5)
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")

    bins = np.linspace(df["snr_thr_p"].min() - 0.02, df["snr_thr_p"].max() + 0.02, 18)
    axes[1].hist(df["snr_thr_p"], bins=bins, color="#9ecae1", edgecolor="white", alpha=0.95)
    for src in order:
        val = df.loc[df["source"] == src, "snr_thr_p"].mean()
        axes[1].axvline(val, color=SOURCE_COLORS[src], linewidth=2.0, label=f"{src} mean")
    axes[1].axvline(2.2, color="#cc2f27", linestyle="--", linewidth=1.2, label="example fixed threshold")
    axes[1].set_xlabel(r"pair-wise p5 SNR threshold")
    axes[1].set_ylabel("number of pairs")
    axes[1].set_title("A single fixed threshold would not match all pools")
    axes[1].grid(axis="y", alpha=0.22, lw=0.5)
    axes[1].legend(frameon=False, fontsize=9, loc="upper left")

    fig.subplots_adjust(wspace=0.28)
    _save(fig, "figA4_gate_percentile.png")


def build_fig_a5() -> None:
    shutil.copyfile(ROOT / "results_crlb" / "a3_snr_profile.png", IEEE_DIR / "figA5_snr_profile.png")


def build_fig_a6() -> None:
    shutil.copyfile(ROOT / "results_crlb" / "b3_multiseed_stability.png", IEEE_DIR / "figA6_multiseed_stability.png")


def build_fig_a7() -> None:
    shutil.copyfile(ROOT / "results_crlb" / "b4_noise_model_validation.png", IEEE_DIR / "figA7_noise_model_validation.png")


def build_fig_a8() -> None:
    df = pd.read_csv(_find("metrics_by_source.csv"))
    order = ["PIV_C", "PIV_E", "PIV_book"]
    df["source"] = pd.Categorical(df["source"], categories=order, ordered=True)
    df = df.sort_values("source")

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.5))
    x = np.arange(len(df))
    width = 0.34

    axes[0].bar(x - width / 2, df["mean_delta_rmse_gate"], width, label=r"$\Delta \mathrm{RMSE}_g$", color="#2c7fb8")
    axes[0].bar(x + width / 2, df["mean_delta_mae_gate"], width, label=r"$\Delta \mathrm{MAE}_g$", color="#fdae61")
    axes[0].set_xticks(x, df["source"])
    axes[0].set_ylabel(r"mean proxy gain ($^\circ$/frame)")
    axes[0].set_title("Error-proxy gains by source")
    axes[0].grid(axis="y", alpha=0.22, lw=0.5)
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")

    bars = axes[1].bar(x, df["mean_delta_corr_gate"], color=[SOURCE_COLORS[s] for s in df["source"]], width=0.56)
    axes[1].set_xticks(x, df["source"])
    axes[1].set_ylabel(r"mean $\Delta \mathrm{Corr}_g$")
    axes[1].set_title("Structural agreement gain")
    axes[1].grid(axis="y", alpha=0.22, lw=0.5)
    for xi, bar, pos, total in zip(x, bars, df["pos_delta_corr"], df["n_pairs"]):
        axes[1].text(
            xi,
            bar.get_height() + 0.0012,
            f"{int(pos)}/{int(total)} positive",
            ha="center",
            va="bottom",
            fontsize=8.8,
        )

    ax3 = axes[2]
    ax3.bar(x, df["mean_snr_thr"], color="#9ecae1", width=0.56, label="mean p5 threshold")
    ax3.set_xticks(x, df["source"])
    ax3.set_ylabel("mean p5 threshold")
    ax3.set_title("Gate statistics in the executed real-data batch")
    ax3.grid(axis="y", alpha=0.22, lw=0.5)
    ax3b = ax3.twinx()
    ax3b.plot(x, df["mean_pass_rate"], color="#cc2f27", marker="o", lw=1.6, label="mean pass rate")
    ax3b.set_ylabel("mean pass rate (%)")
    ax3b.set_ylim(90, 100)
    for xi, thr, pr in zip(x, df["mean_snr_thr"], df["mean_pass_rate"]):
        ax3.text(xi, thr + 0.035, f"{thr:.2f}", ha="center", va="bottom", fontsize=8.5)
        ax3b.text(xi, pr + 0.18, f"{pr:.1f}%", ha="center", va="bottom", fontsize=8.5, color="#aa2a20")

    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=8.8, loc="lower left")

    fig.subplots_adjust(wspace=0.34)
    _save(fig, "figA8_real_source_summary.png")


def build_fig_a9() -> None:
    version_files = [
        ROOT / "results_gt" / "gt_vorticity_results.txt",
        ROOT / "results_gt_widim_nufft_nogate" / "gt_vorticity_results.txt",
        ROOT / "results_gt_widim_track_nufft_nogate" / "gt_vorticity_results.txt",
        ROOT / "results_gt_widim_track_nufft_spatialgate_signpass2" / "gt_vorticity_results.txt",
    ]
    results = [_parse_gt_results(path) for path in version_files]

    flows = ["Rankine", "LambOseen", "SolidRotation", "LinearShear"]
    titles = ["Rankine", "Lamb-Oseen", "Solid rotation", "Linear shear"]

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.0), sharex=True)
    xs = np.arange(len(VERSION_LABELS))
    for ax, flow, title in zip(axes.ravel(), flows, titles):
        variant = [res[flow]["variant_rmse"] for res in results]
        widim = [res[flow]["widim_rmse"] for res in results]
        gain = [res[flow]["gain"] for res in results]

        ax.plot(xs, variant, marker="o", lw=2.0, color=FLOW_COLORS[flow], label="Proposed variant")
        ax.plot(xs, widim, marker="s", lw=1.7, ls="--", color="#444444", label="Matched WIDIM baseline")
        ax.set_title(title)
        ax.set_ylabel(r"RMSE ($^\circ$/frame)")
        ax.grid(alpha=0.22, lw=0.5)
        ax.set_xticks(xs, VERSION_LABELS)
        best_idx = int(np.argmin(variant))
        ax.scatter([best_idx], [variant[best_idx]], s=70, color="#111111", zorder=5)
        ax.text(
            0.03,
            0.97,
            "\n".join(f"{lab}: {g:.2f}x" for lab, g in zip(VERSION_LABELS, gain)),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.4,
            bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="#bbbbbb", alpha=0.95),
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.01), frameon=False)
    fig.subplots_adjust(top=0.91, wspace=0.24, hspace=0.30)
    _save(fig, "figA9_protocol_evolution_absolute.png")


def build_fig_a10() -> None:
    shutil.copyfile(ROOT / "results_crlb" / "b5_complexity_table.png", IEEE_DIR / "figA10_runtime_complexity.png")


def _resolve_real_pair(data_root: Path, label: str) -> tuple[Path, Path]:
    if label.startswith("cam"):
        p1 = next(data_root.rglob(f"{label}a.tif"))
        p2 = p1.with_name(f"{label}b.tif")
        return p1, p2
    if label.startswith("c"):
        p1 = next(data_root.rglob(f"{label}a.bmp"))
        p2 = p1.with_name(f"{label}b.bmp")
        return p1, p2
    if label.startswith("E_camera_"):
        prefix, f0, f1 = label.rsplit("_", 2)
        p1 = next(data_root.rglob(f"{prefix}_{f0}.tif"))
        p2 = next(data_root.rglob(f"{prefix}_{f1}.tif"))
        return p1, p2
    raise FileNotFoundError(label)


def _build_real_case_grid(selected: list[tuple[str, str]], outname: str, title: str) -> None:
    from build_real_proxy_case_fig import GATE_PERCENTILE, load_gray, p99_abs, run_pair

    data_root = _find("source_map.txt").parent

    cmap_field = plt.get_cmap("coolwarm").copy()
    cmap_field.set_bad("#ECE8E1")
    cmap_snr = plt.get_cmap("viridis").copy()

    n_rows = len(selected)
    fig_h = 3.25 * n_rows + 0.5
    fig, axes = plt.subplots(n_rows, 4, figsize=(13.4, fig_h))
    axes = np.atleast_2d(axes)
    field_im = None
    snr_im = None

    for row, (src_title, label) in enumerate(selected):
        p1, p2 = _resolve_real_pair(data_root, label)
        i1 = load_gray(p1, max_size=512)
        i2 = load_gray(p2, max_size=512)
        result = run_pair(i1, i2, gate_percentile=GATE_PERCENTILE)

        omega_w = np.where(result["gate"], result["omega_w"], np.nan)
        omega_n = np.where(result["gate"], result["alpha_new"], np.nan)
        snr = np.asarray(result["snr"], dtype=float)
        gate = np.asarray(result["gate"], dtype=bool)

        vmax_field = max(
            p99_abs(np.concatenate([omega_w.ravel(), omega_n.ravel()]), floor=2.0),
            2.0,
        )

        ax = axes[row, 0]
        ax.imshow(i1, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"({chr(97 + row * 4)}) {src_title} frame $t$", fontsize=10.3)
        ax.text(
            0.02,
            0.04,
            label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.9,
            color="white",
            bbox=dict(boxstyle="round,pad=0.18", fc="black", ec="none", alpha=0.72),
        )
        ax.axis("off")

        ax = axes[row, 1]
        snr_im = ax.imshow(snr, cmap=cmap_snr, interpolation="nearest")
        bad_y, bad_x = np.where(~gate)
        if bad_x.size:
            ax.plot(bad_x, bad_y, "x", color="#F94144", ms=3.4, mew=0.95, alpha=0.82)
        ax.set_title(f"({chr(98 + row * 4)}) Local SNR + gate", fontsize=10.3)
        ax.set_xlabel("grid-x", fontsize=8.4)
        ax.set_ylabel("grid-y", fontsize=8.4)
        ax.tick_params(labelsize=7.8)

        ax = axes[row, 2]
        field_im = ax.imshow(
            omega_w,
            cmap=cmap_field,
            vmin=-vmax_field,
            vmax=vmax_field,
            interpolation="nearest",
        )
        ax.set_title(f"({chr(99 + row * 4)}) Gated $\\Omega_{{\\mathrm{{WIDIM}}}}$", fontsize=10.3)
        ax.set_xlabel("grid-x", fontsize=8.4)
        ax.set_ylabel("grid-y", fontsize=8.4)
        ax.tick_params(labelsize=7.8)

        ax = axes[row, 3]
        ax.imshow(
            omega_n,
            cmap=cmap_field,
            vmin=-vmax_field,
            vmax=vmax_field,
            interpolation="nearest",
        )
        ax.set_title(f"({chr(100 + row * 4)}) Gated $\\hat{{\\Omega}}$", fontsize=10.3)
        ax.set_xlabel("grid-x", fontsize=8.4)
        ax.set_ylabel("grid-y", fontsize=8.4)
        ax.tick_params(labelsize=7.8)
        ax.text(
            0.02,
            0.98,
            (
                rf"$\Delta$RMSE$_g$={result['delta_rmse_gate']:.2f}" "\n"
                rf"$\Delta$MAE$_g$={result['delta_mae_gate']:.2f}" "\n"
                rf"$\Delta$Corr$_g$={result['delta_corr_gate']:.3f}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.0,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#bbbbbb", alpha=0.93),
        )

    if field_im is not None:
        cbar1 = fig.colorbar(field_im, ax=axes[:, 2:4], fraction=0.02, pad=0.02)
        cbar1.ax.tick_params(labelsize=8.0)
        cbar1.set_label(r"rotation rate ($^\circ$/frame)", fontsize=8.8)
    if snr_im is not None:
        cbar2 = fig.colorbar(snr_im, ax=axes[:, 1], fraction=0.028, pad=0.02)
        cbar2.ax.tick_params(labelsize=8.0)
        cbar2.set_label("SNR", fontsize=8.8)

    fig.suptitle(title, fontsize=12.0, y=0.992)
    fig.subplots_adjust(top=0.92, wspace=0.30, hspace=0.34)
    _save(fig, outname)


def build_fig_a11() -> None:
    selected = [
        ("PIV-C", "c099"),
        ("PIV-E", "E_camera_3_frame_00050_00051"),
        ("PIV-book", "cam2_5001"),
    ]
    _build_real_case_grid(
        selected,
        "figA11_real_source_cases.png",
        "Representative additional real-data examples from the three executed source pools",
    )


def build_fig_a14() -> None:
    selected = [
        ("PIV-E boundary case", "E_camera_1_frame_00061_00062"),
    ]
    _build_real_case_grid(
        selected,
        "figA14_real_boundary_case.png",
        "Boundary case where the gain remains positive but visually more modest",
    )


def build_fig_a15() -> None:
    root = ROOT / "results_vortex_all"
    report_p5 = root / "PIVlook_prior_newflow_vorticity_p5" / "prior_report_2026-03-15.txt"
    report_ng = root / "PIVbook_cam1_verify2_nogate" / "report_nogate_2026-03-15.txt"

    pat_p5 = re.compile(
        r"^\s+(?P<pair>[^|]+)\s+\|\s+CorrGateRaw=(?P<craw>[-+0-9.]+)\s+"
        r"CorrGateNew=(?P<cnew>[-+0-9.]+)\s+DeltaCorr=(?P<dc>[-+0-9.]+)\s+\|\s+"
        r"RMSEGateRaw=(?P<rraw>[-+0-9.]+)\s+RMSEGateNew=(?P<rnew>[-+0-9.]+)\s+DeltaRMSE=(?P<dr>[-+0-9.]+)\s+\|\s+"
        r"MAEGateRaw=(?P<mraw>[-+0-9.]+)\s+MAEGateNew=(?P<mnew>[-+0-9.]+)\s+DeltaMAE=(?P<dm>[-+0-9.]+)"
    )
    pat_ng = re.compile(
        r"^\s+(?P<pair>[^|]+)\s+\|\s+Corr=(?P<c>[-+0-9.]+)\s+MAE=(?P<m>[-+0-9.]+)\s+RMSE=(?P<r>[-+0-9.]+)"
    )

    p5: dict[str, dict[str, float]] = {}
    for line in report_p5.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pat_p5.match(line)
        if match:
            pair = match.group("pair").strip()
            p5[pair] = {
                "corr": float(match.group("cnew")),
                "rmse": float(match.group("rnew")),
                "mae": float(match.group("mnew")),
            }

    ng: dict[str, dict[str, float]] = {}
    for line in report_ng.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pat_ng.match(line)
        if match:
            pair = match.group("pair").strip()
            ng[pair] = {
                "corr": float(match.group("c")),
                "rmse": float(match.group("r")),
                "mae": float(match.group("m")),
            }

    common = sorted(set(p5) & set(ng))
    panels = [
        ("corr", r"proxy Corr (higher is better)", "#2c7fb8", "high"),
        ("rmse", r"proxy RMSE ($^\circ$/frame)", "#d95f0e", "low"),
        ("mae", r"proxy MAE ($^\circ$/frame)", "#4dac26", "low"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.3))

    for ax, (metric, ylabel, color, better) in zip(axes, panels):
        ng_vals = np.array([ng[k][metric] for k in common], dtype=float)
        p5_vals = np.array([p5[k][metric] for k in common], dtype=float)

        for a, b in zip(ng_vals, p5_vals):
            ax.plot([0, 1], [a, b], color="#b8b8b8", lw=1.0, alpha=0.9, zorder=1)

        ax.scatter(np.zeros_like(ng_vals), ng_vals, color="#6f6f6f", s=36, zorder=3, label="No gate")
        ax.scatter(np.ones_like(p5_vals), p5_vals, color=color, s=40, zorder=3, label="Adaptive p5")
        ax.set_xticks([0, 1], ["No gate", "Adaptive p5"])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.22, lw=0.5)

        improved = int(np.sum(p5_vals > ng_vals)) if better == "high" else int(np.sum(p5_vals < ng_vals))
        ax.text(
            0.03,
            0.97,
            f"mean: {ng_vals.mean():.3f} -> {p5_vals.mean():.3f}\n"
            f"improved pairs: {improved}/{len(common)}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.8,
            bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="#bbbbbb", alpha=0.95),
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:2], labels[:2], ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(
        "Held-out real-data subset audit: no gate versus adaptive p5 on 11 PIV-book cam1 pairs",
        fontsize=12.0,
        y=1.06,
    )
    fig.subplots_adjust(top=0.76, wspace=0.34)
    _save(fig, "figA15_gate_subset.png")


def main() -> None:
    build_fig_a1()
    build_fig_a2()
    build_fig_a12()
    build_fig_a13()
    build_fig_a3()
    build_fig_a4()
    build_fig_a5()
    build_fig_a6()
    build_fig_a7()
    build_fig_a8()
    build_fig_a9()
    build_fig_a10()
    build_fig_a11()
    build_fig_a14()
    build_fig_a15()
    print("Appendix assets built in", OUT_DIR)


if __name__ == "__main__":
    main()
