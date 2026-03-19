from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
IEEE_DIR = ROOT / "IEEE-Transactions-LaTeX2e-templates-and-instructions"

FLOW_COLORS = {
    "Rankine": "#1f77b4",
    "LambOseen": "#2ca02c",
    "SolidRotation": "#ff7f0e",
    "LinearShear": "#d62728",
}

FLOW_LABELS = {
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

SOURCE_LABELS = {
    "PIV_C": "PIV-C",
    "PIV_E": "PIV-E",
    "PIV_book": "PIV-book",
}


def _find(name: str) -> Path:
    return next(ROOT.rglob(name))


def _save(fig: plt.Figure, name: str) -> None:
    out = IEEE_DIR / name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_proxy_bridge() -> None:
    df = pd.read_csv(_find("synthetic_proxy_calibration_2026-03-16.csv"))

    panels = [
        ("delta_rmse_proxy", r"$\Delta \mathrm{RMSE}_g$", "Primary proxy"),
        ("delta_corr_proxy", r"$\Delta \mathrm{Corr}_g$", "Supporting proxy"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.10, 2.18), sharey=True)
    y = df["delta_rmse_true"].to_numpy()

    for ax, (col, xlabel, title) in zip(axes, panels):
        x = df[col].to_numpy()
        for flow, sub in df.groupby("flow"):
            ax.scatter(
                sub[col],
                sub["delta_rmse_true"],
                s=13,
                alpha=0.85,
                color=FLOW_COLORS[flow],
                edgecolor="white",
                linewidth=0.35,
                label=FLOW_LABELS[flow],
            )

        coef = np.polyfit(x, y, 1)
        xs = np.linspace(float(np.min(x)), float(np.max(x)), 200)
        ax.plot(xs, coef[0] * xs + coef[1], color="#222222", lw=1.15, alpha=0.9)
        ax.axhline(0.0, color="#999999", lw=0.8, ls="--")
        if col != "delta_corr_proxy":
            ax.axvline(0.0, color="#999999", lw=0.8, ls="--")

        pear = pd.Series(x).corr(pd.Series(y), method="pearson")
        sign = (((x > 0) == (y > 0)).mean()) * 100.0
        ax.text(
            0.03,
            0.97,
            f"{title}\nPearson={pear:.3f}\nSign={sign:.1f}%",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=6.5,
            bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="#bbbbbb", alpha=0.95),
        )
        ax.set_xlabel(xlabel, fontsize=6.8)
        ax.set_ylabel(r"$\Delta \mathrm{RMSE}_{\mathrm{true}}$", fontsize=6.8)
        ax.grid(alpha=0.22, lw=0.45)
        ax.tick_params(labelsize=6.6)
        ax.set_title(title, fontsize=6.8, pad=2.5)

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        frameon=False,
        fontsize=6.6,
        columnspacing=0.7,
        handletextpad=0.4,
    )
    fig.subplots_adjust(top=0.80, wspace=0.14, bottom=0.24, left=0.07, right=0.995)
    _save(fig, "fig08_proxy_bridge.png")


def build_real_gain_modes() -> None:
    df = pd.read_csv(_find("metrics_all_pairs.csv"))

    fig, axes = plt.subplots(2, 1, figsize=(3.28, 4.25))


    ax = axes[0]
    for source, sub in df.groupby("source"):
        ax.scatter(
            sub["delta_rmse_gate"],
            sub["delta_corr_gate"],
            s=12,
            alpha=0.58,
            color=SOURCE_COLORS[source],
            edgecolor="none",
            label=SOURCE_LABELS[source],
        )
    ax.axhline(0.0, color="#999999", lw=0.85, ls="--")
    ax.axvline(0.0, color="#999999", lw=0.85, ls="--")
    ax.set_xlabel(r"$\Delta \mathrm{RMSE}_g$", fontsize=7.1)
    ax.set_ylabel(r"$\Delta \mathrm{Corr}_g$", fontsize=7.1)
    ax.grid(alpha=0.22, lw=0.45)
    ax.tick_params(labelsize=6.9)
    ax.text(10.9, -0.010, "tail suppression", fontsize=6.5, color="#555555")
    ax.text(0.82, 0.090, "structure recovery", fontsize=6.5, color="#555555")
    ax.text(0.97, 0.97, "PIV-C", transform=ax.transAxes, ha="right", va="top", fontsize=6.4, color=SOURCE_COLORS["PIV_C"])
    ax.text(0.97, 0.89, "PIV-E", transform=ax.transAxes, ha="right", va="top", fontsize=6.4, color=SOURCE_COLORS["PIV_E"])
    ax.text(0.97, 0.81, "PIV-book", transform=ax.transAxes, ha="right", va="top", fontsize=6.4, color=SOURCE_COLORS["PIV_book"])

    for label, marker, dx, dy in [
        ("cam2_5001", "*", 0.20, 0.022),
        ("cam1_0001", "D", 0.16, -0.012),
    ]:
        sub = df.loc[df["label"] == label]
        if not sub.empty:
            x = float(sub["delta_rmse_gate"].iloc[0])
            y = float(sub["delta_corr_gate"].iloc[0])
            ax.scatter([x], [y], s=70 if marker == "*" else 28, color="#111111", marker=marker, zorder=5)
            ax.text(x + dx, y + dy, label, fontsize=6.7, color="#111111", va="center")


    ax = axes[1]
    for source, sub in df.groupby("source"):
        x_med = float(sub["delta_rmse_gate"].median())
        y_med = float(sub["delta_corr_gate"].median())
        x_lo = x_med - float(sub["delta_rmse_gate"].quantile(0.25))
        x_hi = float(sub["delta_rmse_gate"].quantile(0.75)) - x_med
        y_lo = y_med - float(sub["delta_corr_gate"].quantile(0.25))
        y_hi = float(sub["delta_corr_gate"].quantile(0.75)) - y_med
        ax.errorbar(
            x_med,
            y_med,
            xerr=np.array([[x_lo], [x_hi]]),
            yerr=np.array([[y_lo], [y_hi]]),
            fmt="o",
            ms=5.8,
            color=SOURCE_COLORS[source],
            ecolor=SOURCE_COLORS[source],
            elinewidth=1.0,
            capsize=2,
            label=SOURCE_LABELS[source],
        )
        ax.text(x_med + 0.14, y_med + 0.0018, SOURCE_LABELS[source], fontsize=6.7, color=SOURCE_COLORS[source])

    ax.axhline(0.0, color="#999999", lw=0.85, ls="--")
    ax.axvline(0.0, color="#999999", lw=0.85, ls="--")
    ax.set_xlabel(r"median $\Delta \mathrm{RMSE}_g$", fontsize=7.1)
    ax.set_ylabel(r"median $\Delta \mathrm{Corr}_g$", fontsize=7.1)
    ax.grid(alpha=0.22, lw=0.45)
    ax.tick_params(labelsize=6.9)

    fig.subplots_adjust(top=0.97, hspace=0.34, bottom=0.10, left=0.16, right=0.985)
    _save(fig, "fig09_real_gain_modes.png")


def main() -> None:
    build_proxy_bridge()
    build_real_gain_modes()
    print("Built main extra figures in", IEEE_DIR)


if __name__ == "__main__":
    main()
