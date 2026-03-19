from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


plt.rcParams["font.family"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


C = {
    "bg": "#FFFDF8",
    "paper": "#FFFFFF",
    "ink": "#1F2933",
    "line": "#35556B",
    "muted": "#607180",
    "c1": "#3C9BC9",
    "c2": "#65BDBA",
    "c3": "#B0D6A9",
    "c4": "#FEE199",
    "c5": "#FDCD94",
    "c6": "#F97F5F",
    "c7": "#FC757B",
}


def add_card(fig, img_path: Path, x: float, y: float, w: float, h: float,
             step_no: int, title: str, subtitle: str, accent: str) -> tuple[float, float, float, float]:
    bg_ax = fig.add_axes([0, 0, 1, 1], zorder=0)
    bg_ax.axis("off")
    card = FancyBboxPatch(
        (x, y),
        w,
        h,
        transform=fig.transFigure,
        boxstyle="round,pad=0.008,rounding_size=0.018",
        linewidth=1.4,
        edgecolor=accent,
        facecolor=C["paper"],
        zorder=0.5,
    )
    bg_ax.add_patch(card)

    fig.text(
        x + 0.018,
        y + h - 0.035,
        str(step_no),
        ha="center",
        va="center",
        fontsize=10.5,
        color="white",
        fontweight="bold",
        bbox=dict(boxstyle="circle,pad=0.24", fc=accent, ec=accent),
        zorder=4,
    )
    fig.text(
        x + 0.042,
        y + h - 0.036,
        title,
        ha="left",
        va="center",
        fontsize=10.6,
        color=C["ink"],
        fontweight="bold",
        zorder=4,
    )
    fig.text(
        x + 0.042,
        y + h - 0.062,
        subtitle,
        ha="left",
        va="center",
        fontsize=8.5,
        color=C["muted"],
        zorder=4,
    )

    ax = fig.add_axes([x + 0.012, y + 0.015, w - 0.024, h - 0.088], zorder=2)
    ax.imshow(mpimg.imread(img_path))
    ax.axis("off")
    return x, y, w, h


def add_arrow(fig, start: tuple[float, float], end: tuple[float, float], rad: float = 0.0) -> None:
    ax = fig.add_axes([0, 0, 1, 1], zorder=3)
    ax.axis("off")
    arrow = FancyArrowPatch(
        start,
        end,
        transform=fig.transFigure,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.8,
        color=C["line"],
        alpha=0.96,
    )
    ax.add_patch(arrow)


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    flow_dir = repo_root / "流程图"
    ieee_dir = repo_root / "IEEE-Transactions-LaTeX2e-templates-and-instructions"

    panels = {
        1: flow_dir / "panel01_input_pair.png",
        2: flow_dir / "panel02_widim_tracking.png",
        3: flow_dir / "panel03_translation_prior.png",
        4: flow_dir / "panel04_nufft_rotation.png",
        5: flow_dir / "panel05_raw_rotation_map.png",
        6: flow_dir / "panel06_spatial_postprocess.png",
        7: flow_dir / "panel07_adaptive_p5_gate.png",
        8: flow_dir / "panel08_dual_outputs.png",
    }
    for p in panels.values():
        if not p.exists():
            raise FileNotFoundError(f"Missing panel image: {p}")

    fig = plt.figure(figsize=(16.2, 9.0), dpi=300)
    fig.patch.set_facecolor(C["bg"])

    fig.text(
        0.5,
        0.955,
        "Practical Deployment Workflow on a Representative Real PIV Pair",
        ha="center",
        va="center",
        fontsize=15.2,
        fontweight="bold",
        color=C["ink"],
    )
    fig.text(
        0.5,
        0.928,
        "All local windows are taken from the same representative spike region corrected by the final protocol.",
        ha="center",
        va="center",
        fontsize=10.0,
        color=C["muted"],
    )

    xs = [0.028, 0.270, 0.512, 0.754]
    top_y = 0.55
    bot_y = 0.11
    w = 0.218
    h = 0.33

    card1 = add_card(fig, panels[1], xs[0], top_y, w, h, 1, "Input Pair", "same query location on I1 and I2", C["c1"])
    card2 = add_card(fig, panels[2], xs[1], top_y, w, h, 2, "WIDIM Tracking", "window correspondence and displacement prior", C["c1"])
    card3 = add_card(fig, panels[3], xs[2], top_y, w, h, 3, "Translation-only Prior", "keep rotation signal; no affine de-rotation", C["c2"])
    card4 = add_card(fig, panels[4], xs[3], top_y, w, h, 4, "NUFFT Rotation Estimate", "polar spectra, S(theta), and peak SNR", C["c3"])

    card8 = add_card(fig, panels[8], xs[0], bot_y, w, h, 8, "Dual Outputs", "WIDIM reference and proposed rotation-rate map", C["c7"])
    card7 = add_card(fig, panels[7], xs[1], bot_y, w, h, 7, "Adaptive p5 Gate", "pair-wise reliability threshold on SNR", C["c6"])
    card6 = add_card(fig, panels[6], xs[2], bot_y, w, h, 6, "Spatial Post-process", "pass-1 reject, pass-2 sign-island, local refill", C["c5"])
    card5 = add_card(fig, panels[5], xs[3], bot_y, w, h, 5, "Raw Rotation Map", "the corrected spike is marked in red", C["c4"])


    add_arrow(fig, (card1[0] + w, top_y + h * 0.52), (card2[0], top_y + h * 0.52))
    add_arrow(fig, (card2[0] + w, top_y + h * 0.52), (card3[0], top_y + h * 0.52))
    add_arrow(fig, (card3[0] + w, top_y + h * 0.52), (card4[0], top_y + h * 0.52))


    add_arrow(fig, (card4[0] + w * 0.90, top_y), (card5[0] + w * 0.90, bot_y + h), rad=-0.06)


    add_arrow(fig, (card5[0], bot_y + h * 0.52), (card6[0] + w, bot_y + h * 0.52))
    add_arrow(fig, (card6[0], bot_y + h * 0.52), (card7[0] + w, bot_y + h * 0.52))
    add_arrow(fig, (card7[0], bot_y + h * 0.52), (card8[0] + w, bot_y + h * 0.52))

    fig.text(
        0.5,
        0.035,
        "Key design: WIDIM provides only the displacement prior; NUFFT remains the rotation estimator.",
        ha="center",
        va="center",
        fontsize=10.0,
        color=C["line"],
    )

    out_flow = flow_dir / "fig04_system_arch_detailed.png"
    out_ieee = ieee_dir / "fig04_system_arch.png"
    fig.savefig(out_flow, dpi=300, bbox_inches="tight", facecolor=C["bg"], edgecolor="none")
    fig.savefig(out_ieee, dpi=300, bbox_inches="tight", facecolor=C["bg"], edgecolor="none")
    plt.close(fig)
    print(f"Generated: {out_flow}")
    print(f"Generated: {out_ieee}")


if __name__ == "__main__":
    main()
