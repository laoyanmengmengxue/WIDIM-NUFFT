from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

from bridge_config import FIGS_DIR, RESULTS_ROOT


def build_truth_vs_realism(df: pd.DataFrame, out_path: Path, layout: str = "2x2") -> None:
    level_order = ["L0", "L1", "L2", "L3"]
    flow_order = ["rankine", "lamb_oseen", "solid_rotation", "mixed_vortex"]
    flow_labels = {
        "rankine": "Rankine",
        "lamb_oseen": "Lamb-Oseen",
        "solid_rotation": "Solid rotation",
        "mixed_vortex": "Mixed vortex",
    }
    colors = {
        "rankine": "#2F6690",
        "lamb_oseen": "#3A7D44",
        "solid_rotation": "#8E5572",
        "mixed_vortex": "#C96E12",
    }

    if layout == "1x4":
        nrows, ncols = 1, 4
        figsize = (8.6, 2.35)
    else:
        nrows, ncols = 2, 2
        figsize = (6.0, 4.6)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.asarray(axes).reshape(-1)
    xs = np.arange(len(level_order))
    for idx, (ax, flow) in enumerate(zip(axes, flow_order)):
        sub = (
            df[df["flow_name"] == flow]
            .groupby("level_name", as_index=False)[["rmse_true_widim", "rmse_true_new"]]
            .mean()
            .set_index("level_name")
            .reindex(level_order)
        )
        y_w = sub["rmse_true_widim"].to_numpy(dtype=float)
        y_p = sub["rmse_true_new"].to_numpy(dtype=float)
        color = colors[flow]
        ax.fill_between(xs, y_p, y_w, where=(y_w >= y_p), color=color, alpha=0.16, zorder=1)
        ax.plot(xs, y_w, "--o", color="#7F7F7F", lw=1.45, ms=3.6, alpha=0.95, zorder=3)
        ax.plot(xs, y_p, "-o", color=color, lw=1.85, ms=3.8, zorder=4)
        ymin = float(min(np.min(y_w), np.min(y_p)))
        ymax = float(max(np.max(y_w), np.max(y_p)))
        pad = max(0.06 * (ymax - ymin), 0.06)
        ax.set_ylim(ymin - pad, ymax + pad)
        rel = 100.0 * (1.0 - np.mean(y_p) / np.mean(y_w))
        ax.text(
            0.04,
            0.94,
            rf"$\downarrow {rel:.1f}\%$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.6,
            color=color,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#C7C7C7", alpha=0.92),
        )
        ax.set_title(flow_labels[flow], fontsize=6.0, pad=2.0)
        ax.set_xticks(xs, level_order)
        ax.grid(alpha=0.24, lw=0.45)
        ax.tick_params(labelsize=5.8, length=2.0, pad=1.3)
        ax.set_box_aspect(3.0 / 4.0)
        if idx not in (0, 2):
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("Truth RMSE", fontsize=6.1)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    if layout == "1x4":
        fig.subplots_adjust(top=0.80, left=0.06, right=0.995, bottom=0.10, wspace=0.34, hspace=0.20)
    else:
        fig.subplots_adjust(top=0.95, left=0.06, right=0.995, bottom=0.05, wspace=0.04, hspace=0.10)
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def build_bridge_visual_grid(df: pd.DataFrame, cases_dir: Path, out_path: Path) -> None:
    flow_order = ["rankine", "lamb_oseen", "solid_rotation", "mixed_vortex"]
    title_map = {
        "rankine": "Rankine",
        "lamb_oseen": "Lamb-Oseen",
        "solid_rotation": "Solid rotation",
        "mixed_vortex": "Mixed vortex",
    }
    arrays = []
    vmax = 1.0
    for flow in flow_order:
        path = cases_dir / f"{flow}_L3_seed0_sample.npz"
        d = np.load(path, allow_pickle=True)
        truth = d["omega_truth_grid"]
        widim = d["omega_widim"]
        prop = d["alpha_new"]
        vmax = max(
            vmax,
            float(np.nanpercentile(np.abs(truth), 99)),
            float(np.nanpercentile(np.abs(widim), 99)),
            float(np.nanpercentile(np.abs(prop), 99)),
        )
        arrays.append((flow, truth, widim, prop))

    fig = plt.figure(figsize=(8.85, 5.45))
    gs = fig.add_gridspec(3, 5, width_ratios=[1, 1, 1, 1, 0.075], wspace=0.12, hspace=0.10)
    axes = np.empty((3, 4), dtype=object)
    for row in range(3):
        for col in range(4):
            axes[row, col] = fig.add_subplot(gs[row, col])
    cax = fig.add_subplot(gs[:, 4])
    cmap = LinearSegmentedColormap.from_list(
        "bridge_vort_cmap",
        ["#FC757B", "#F97F5F", "#FAA26F", "#FDCD94",
         "#FEE199", "#B0D6A9", "#65BDBA", "#3C9BC9"]
    )
    ims = []
    for col, (flow, truth, widim, prop) in enumerate(arrays):
        for row, (img, row_label) in enumerate(
            ((truth, "Truth"), (widim, "WIDIM"), (prop, "Proposed"))
        ):
            ax = axes[row, col]
            im = ax.imshow(img, cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="nearest")
            ims.append(im)
            if row == 0:
                ax.set_title(title_map[flow], fontsize=10.5)
            if col == 0:
                ax.set_ylabel(row_label, fontsize=10.0)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)
                spine.set_color("#555555")

    cbar = fig.colorbar(ims[-1], cax=cax)
    cbar.ax.tick_params(labelsize=7.4)
    cbar.set_label("Rotation rate (deg/frame)", fontsize=8.0)
    fig.subplots_adjust(left=0.06, right=0.95, top=0.92, bottom=0.07)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def build_proxy_bridge(df: pd.DataFrame, out_path: Path) -> None:
    flow_colors = {
        "rankine": "#2F6690",
        "lamb_oseen": "#3A7D44",
        "solid_rotation": "#8E5572",
        "mixed_vortex": "#C96E12",
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))

    ax = axes[0]
    for flow, color in flow_colors.items():
        sub = df[df["flow_name"] == flow]
        ax.scatter(sub["delta_rmse_g"], sub["delta_rmse_true"], s=16, alpha=0.82, c=color, label=flow.replace("_", " "))
    x = df["delta_rmse_g"].to_numpy(dtype=float)
    y = df["delta_rmse_true"].to_numpy(dtype=float)
    coeff = np.polyfit(x, y, deg=1) if len(x) >= 2 else [1.0, 0.0]
    xr = np.linspace(np.nanmin(x), np.nanmax(x), 100)
    ax.plot(xr, coeff[0] * xr + coeff[1], color="#222222", lw=1.1)
    ax.axhline(0.0, color="#888888", ls="--", lw=0.8)
    ax.axvline(0.0, color="#888888", ls="--", lw=0.8)
    ax.set_xlabel("Delta RMSE_g")
    ax.set_ylabel("Delta RMSE_true (WIDIM->final)")
    ax.set_title("Primary proxy")
    ax.grid(alpha=0.22)

    ax = axes[1]
    for flow, color in flow_colors.items():
        sub = df[df["flow_name"] == flow]
        ax.scatter(sub["delta_corr_g"], sub["delta_rmse_true"], s=16, alpha=0.82, c=color, label=flow.replace("_", " "))
    x = df["delta_corr_g"].to_numpy(dtype=float)
    y = df["delta_rmse_true"].to_numpy(dtype=float)
    coeff = np.polyfit(x, y, deg=1) if len(x) >= 2 else [1.0, 0.0]
    xr = np.linspace(np.nanmin(x), np.nanmax(x), 100)
    ax.plot(xr, coeff[0] * xr + coeff[1], color="#222222", lw=1.1)
    ax.axhline(0.0, color="#888888", ls="--", lw=0.8)
    ax.axvline(0.0, color="#888888", ls="--", lw=0.8)
    ax.set_xlabel("Delta Corr_g")
    ax.set_ylabel("Delta RMSE_true (WIDIM->final)")
    ax.set_title("Supporting proxy")
    ax.grid(alpha=0.22)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", frameon=False, fontsize=7.8)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def build_realism_ladder(sample_npz_paths: list[Path], out_path: Path) -> None:
    sample_npz_paths = sorted(sample_npz_paths)
    rows = 4
    cols = len(sample_npz_paths)
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 8.6))
    if cols == 1:
        axes = np.asarray(axes).reshape(rows, 1)

    for col, path in enumerate(sample_npz_paths):
        d = np.load(path, allow_pickle=True)
        title = f"{str(d['flow_label'])} | {str(d['level_name'])}"
        show_list = [
            (d["i1"], "I1"),
            (d["omega_truth_grid"], "Truth"),
            (d["omega_widim"], "WIDIM"),
            (d["alpha_new"], "Proposed"),
        ]
        vmax = max(
            np.nanpercentile(np.abs(d["omega_truth_grid"]), 99),
            np.nanpercentile(np.abs(d["alpha_new"]), 99),
            np.nanpercentile(np.abs(d["omega_widim"]), 99),
            1.0,
        )
        for row, (img, name) in enumerate(show_list):
            ax = axes[row, col]
            if row == 0:
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(img, cmap="coolwarm", vmin=-vmax, vmax=vmax)
            if row == 0:
                ax.set_title(title, fontsize=9.0)
            ax.set_ylabel(name, fontsize=8.5)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    metrics_path = RESULTS_ROOT / "metrics" / "all_cases.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(metrics_path)

    build_truth_vs_realism(df, FIGS_DIR / "bridge_truth_vs_realism.png")
    build_bridge_visual_grid(df, RESULTS_ROOT / "cases", FIGS_DIR / "bridge_visual_grid.png")
    build_proxy_bridge(df, FIGS_DIR / "bridge_proxy_bridge.png")

    sample_paths = list((RESULTS_ROOT / "cases").glob("*sample*.npz"))
    if sample_paths:
        build_realism_ladder(sample_paths, FIGS_DIR / "bridge_realism_ladder.png")


if __name__ == "__main__":
    main()
