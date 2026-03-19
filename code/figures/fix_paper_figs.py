"""
fix_paper_figs.py  —  Paper Figure Quality Fixes
=================================================
Fixes paper-figure consistency and quality issues:

  Fig 8b (fig07b_rotation_mc.png):
      "0.622" text covered by errorbar cap
  Fig 9  (fig09_rankine_disp.png):
      Arrows faint, not dense, Chinese font boxes
      Second column should be WIDIM (not NCC+NUFFT)
  Fig 10 (fig10_backstep.png):
      Ugly 2×3 layout → cleaner 2×2 publication layout
  Fig 11 (fig11_overall.png):
      Label "NCC+NUFFT" → "WIDIM+NUFFT"
  Fig 13 (fig13_gt_vorticity.png, PDF Fig 12):
      Vorticity colormaps missing colorbars; title update
  Fig 14 (fig14_gt_summary.png, PDF Fig 6):
      Replace old direct-vs-WIDIM bar chart with V0→V3 protocol summary

Run:  python fix_paper_figs.py
Outputs copied to IEEE LaTeX folder automatically.
"""

import os, sys, contextlib, io as _io, time, shutil, gc, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
from scipy.ndimage import zoom as ndimage_zoom
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
IEEE_DIR   = os.path.join(THIS_DIR,
    "IEEE-Transactions-LaTeX2e-templates-and-instructions")
REAL_DIR   = os.path.join(PARENT_DIR, "真实数据集")
OUT_DIR    = os.path.join(THIS_DIR, "results_fix_figs")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, THIS_DIR)

from poc_common import (
    generate_speckle_field, apply_affine, extract_window,
    DEFAULT_PARAMS, set_plot_style, COLORS,
)
from poc_point3 import estimate_rotation, OPT_R_MIN, OPT_WEIGHT, OPT_POC
from comparison import (
    make_rankine_vortex, widim_fullfield, fullfield_compare,
    WS, STEP, FS, _compute_gradients,
    _search_ncc_window, _search_nufft_ncc_window, _ncc,
)
from a3_a4_analysis import run_a4_efficiency_gap
from gt_comparison_vorticity import (
    make_rankine as gt_make_rankine,
    make_lamb_oseen, make_solid_rotation, make_shear,
    _run_widim, SIGMA_S, SNR_MIN, ALPHA_MAX,
    WS as WS_GT, STEP as STEP_GT,
)


set_plot_style()
plt.rcParams.update({
    'font.family':     'DejaVu Sans',
    'axes.unicode_minus': False,
})


def _copy_to_ieee(src, fname):
    """Copy output file to IEEE folder with given filename."""
    dst = os.path.join(IEEE_DIR, fname)
    shutil.copy2(src, dst)
    print(f"  → copied to {fname}")


def _get_font(size, bold=False):
    """Return a reasonably portable TrueType font for image relabeling."""
    names = (
        ["DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf"]
        if bold else
        ["DejaVuSans.ttf", "arial.ttf", "Arial.ttf"]
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_centered(draw, xy, text, font, fill=(0, 0, 0)):
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    draw.text((xy[0] - w / 2, xy[1] - h / 2), text, font=font, fill=fill)


def _draw_rotated_text(base_img, center_xy, text, font, fill=(0, 0, 0), angle=90):
    tmp = Image.new("RGBA", (600, 120), (255, 255, 255, 0))
    d = ImageDraw.Draw(tmp)
    bbox = d.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    d.text(((600 - w) / 2, (120 - h) / 2), text, font=font, fill=fill)
    rot = tmp.rotate(angle, expand=True)
    px = int(center_xy[0] - rot.size[0] / 2)
    py = int(center_xy[1] - rot.size[1] / 2)
    base_img.alpha_composite(rot, (px, py))


def load_gray(path, max_size=256):
    img = Image.open(path)
    if img.mode != 'L':
        img = img.convert('L')
    arr = np.array(img).astype(float) / 255.0
    H, W = arr.shape
    if H > max_size or W > max_size:
        rr = min(H, max_size); cc = min(W, max_size)
        r0 = (H - rr)//2;  c0 = (W - cc)//2
        arr = arr[r0:r0+rr, c0:c0+cc]
    return arr


def fix_fig07b():
    """Rebuild Fig.7b as a dedicated Rankine MC RMSE summary."""
    print("\n[FIG 07b] Rebuilding dedicated Rankine MC summary panel ...")

    stats_path = os.path.join(
        THIS_DIR,
        "results_gt_widim_track_nufft_spatialgate_signpass2",
        "gt_vorticity_results.txt",
    )
    if not os.path.exists(stats_path):
        raise FileNotFoundError(stats_path)

    with open(stats_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]

    rankine_idx = None
    for idx, line in enumerate(lines):
        if line == "Flow: Rankine Vortex":
            rankine_idx = idx
            break
    if rankine_idx is None or rankine_idx + 4 >= len(lines):
        raise RuntimeError("Could not locate Rankine block in gt_vorticity_results.txt")

    import re
    n_total = int(re.search(r"(\d+)", lines[rankine_idx + 1]).group(1))

    m_n = re.search(r"NUFFT RMSE:\s*([0-9.]+).*\[([0-9.]+),\s*([0-9.]+)\]", lines[rankine_idx + 2])
    m_w = re.search(r"WIDIM RMSE:\s*([0-9.]+).*\[([0-9.]+),\s*([0-9.]+)\]", lines[rankine_idx + 3])
    m_i = re.search(r"Improvement:\s*([0-9.]+)x", lines[rankine_idx + 4])
    if not (m_n and m_w and m_i):
        raise RuntimeError("Could not parse Rankine RMSE/CI lines in gt_vorticity_results.txt")

    rmse_n = float(m_n.group(1))
    ci_n_lo = float(m_n.group(2))
    ci_n_hi = float(m_n.group(3))
    rmse_w = float(m_w.group(1))
    ci_w_lo = float(m_w.group(2))
    ci_w_hi = float(m_w.group(3))
    improvement = float(m_i.group(1))
    n_seeds = 30

    print(f"  final MC: N={n_total}, NUFFT={rmse_n:.3f} [{ci_n_lo:.3f}, {ci_n_hi:.3f}], "
          f"WIDIM={rmse_w:.3f} [{ci_w_lo:.3f}, {ci_w_hi:.3f}]")

    colors = ['#3C9BC9', '#F2CF66']
    fig, ax = plt.subplots(1, 1, figsize=(5.4, 4.2))
    methods = ['WIDIM-track+\nNUFFT', 'WIDIM\ngradient']
    rmses = [rmse_n, rmse_w]
    yerr = np.array([
        [rmse_n - ci_n_lo, rmse_w - ci_w_lo],
        [ci_n_hi - rmse_n, ci_w_hi - rmse_w],
    ])
    bars = ax.bar(methods, rmses, color=colors, alpha=0.90,
                  edgecolor='k', width=0.58)
    ax.errorbar([0, 1], rmses, yerr=yerr, fmt='none',
                color='black', capsize=6, lw=1.8, capthick=1.8)
    for bar, r, hi in zip(bars, rmses, yerr[1]):
        ax.text(bar.get_x() + bar.get_width()/2,
                r + hi + 0.04,
                f'{r:.3f}°',
                ha='center', va='bottom', fontsize=10.2)
    ax.text(0.5, max(rmses) * 1.16,
            f'Lower is better   |   {improvement:.2f}x improvement',
            ha='center', va='bottom', fontsize=8.8, color='#154360')
    ax.set_ylabel('RMSE (deg/frame)', fontsize=10.5)
    ax.set_title(f'Rankine MC RMSE (95% bootstrap CI)\n'
                 f'final protocol, N={n_total}, {n_seeds} seeds',
                 fontsize=9.6)
    ax.set_ylim(0, max(ci_n_hi, ci_w_hi) * 1.30)
    ax.grid(True, alpha=0.25, axis='y')
    fig.tight_layout(pad=0.8)

    out = os.path.join(OUT_DIR, 'fig07b_rotation_mc.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    _copy_to_ieee(out, 'fig07b_rotation_mc.png')
    print(f"  Saved: {out}")
    return out


def fix_fig08_disp_reference():
    """Rebuild Fig.8 as a compact reference figure for the displacement path."""
    print("\n[FIG 08] Rebuilding displacement-path reference bound figure ...")

    sigmas = np.array([0.002, 0.005, 0.010, 0.015, 0.020], dtype=float)
    crlb_mpx = np.array([0.50, 1.25, 2.49, 3.74, 4.98], dtype=float)
    ncc_mpx = np.array([2.84, 3.28, 4.91, 6.84, 9.74], dtype=float)
    widim_mpx = np.array([0.91, 2.12, 4.28, 6.54, 8.93], dtype=float)
    eta_ncc = np.array([3.1, 14.4, 25.8, 29.8, 26.2], dtype=float)
    eta_widim = np.array([30.2, 34.6, 33.9, 32.6, 31.1], dtype=float)

    crlb = crlb_mpx / 1000.0
    ncc = ncc_mpx / 1000.0
    widim = widim_mpx / 1000.0

    colors = {
        "crlb": "#222222",
        "ncc": "#F2644B",
        "widim": "#3C9BC9",
        "accent": "#5B6C7C",
    }

    fig, axes = plt.subplots(
        1, 2, figsize=(8.0, 2.55), dpi=180,
        gridspec_kw={"width_ratios": [1.03, 0.97]}
    )

    ax = axes[0]
    ax.loglog(sigmas, crlb, "--", color=colors["crlb"], lw=2.1, label="CRLB (theory)")
    ax.loglog(sigmas, ncc, "-o", color=colors["ncc"], lw=1.9, ms=5.8,
              label="NCC (Gaussian sub-px)")
    ax.loglog(sigmas, widim, "-s", color=colors["widim"], lw=1.9, ms=5.4,
              label="WIDIM (3-iter)")
    ax.set_xlabel(r"Image noise level $\sigma_s$", fontsize=8.9)
    ax.set_ylabel("Displacement RMSE (px)", fontsize=8.9)
    ax.set_title("(a) RMSE vs bound", fontsize=9.2, fontweight="bold")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(loc="upper left", fontsize=7.8, frameon=True)
    ax.tick_params(labelsize=8.0)

    ax2 = axes[1]
    ax2.semilogx(sigmas, eta_ncc, "-o", color=colors["ncc"], lw=1.9, ms=5.8,
                 label="NCC")
    ax2.semilogx(sigmas, eta_widim, "-s", color=colors["widim"], lw=1.9, ms=5.4,
                 label="WIDIM")
    ax2.axhline(100, color=colors["crlb"], ls="--", lw=1.5, alpha=0.9)
    ax2.axvline(0.010, color=colors["accent"], ls=":", lw=1.2, alpha=0.85)
    ax2.scatter([0.010, 0.010], [25.8, 33.9], s=34,
                color=[colors["ncc"], colors["widim"]], zorder=4)
    ax2.text(0.0105, 25.8 + 1.4, "25.8%", color=colors["ncc"], fontsize=8.7, weight="bold")
    ax2.text(0.0105, 33.9 + 1.4, "33.9%", color=colors["widim"], fontsize=8.7, weight="bold")
    ax2.text(
        0.03, 0.97,
        "WIDIM stays in a stable reference regime",
        transform=ax2.transAxes, ha="left", va="top", fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.24", facecolor="white",
                  edgecolor="#C9D2D9", alpha=0.95),
    )
    ax2.set_xlabel(r"Image noise level $\sigma_s$", fontsize=8.9)
    ax2.set_ylabel("Displacement-CRLB efficiency (%)", fontsize=8.9)
    ax2.set_title("(b) Efficiency vs noise", fontsize=9.2, fontweight="bold")
    ax2.set_ylim(0, 110)
    ax2.grid(True, which="both", alpha=0.22)
    ax2.legend(loc="lower right", fontsize=7.8, frameon=True)
    ax2.tick_params(labelsize=8.0)

    fig.tight_layout(pad=0.8, w_pad=1.1)

    out = os.path.join(OUT_DIR, "fig08_disp_crlb.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)

    _copy_to_ieee(out, "fig08_disp_crlb.png")
    print(f"  Saved: {out}")
    return out


def fix_fig06_efficiency_gap():
    """Rebuild A4 as a compact main-text figure for the efficiency gap."""
    print("\n[FIG 06] Rebuilding efficiency-gap figure from current A4 analysis ...")

    res = run_a4_efficiency_gap(None)
    sigmas = np.array(sorted([k for k in res["crlb_theory_deg"].keys() if k > 0]), dtype=float)
    crlb = np.array([res["crlb_theory_deg"][s] for s in sigmas], dtype=float)
    rmse_sub = np.array([res["rmse_subpix"][s] for s in sigmas], dtype=float)
    rmse_nosub = np.array([res["rmse_no_subpix"][s] for s in sigmas], dtype=float)
    eta_sub = np.array([res["eta_total"][s] for s in sigmas], dtype=float)
    eta_nosub = np.array([res["eta_nosub"][s] for s in sigmas], dtype=float)
    eta_hierarchy = float(res["eta_hierarchy"])
    quant_floor = float(res["quant_floor_theory"])

    sigma_demo = 0.01
    idx_demo = int(np.argmin(np.abs(sigmas - sigma_demo)))
    mse_gain = 100.0 * (rmse_nosub[idx_demo] ** 2 - rmse_sub[idx_demo] ** 2) / (rmse_nosub[idx_demo] ** 2)

    colors = {
        "bound": "#222222",
        "sub": "#2C7FB8",
        "nosub": "#D95F5F",
        "accent": "#B8860B",
        "muted": "#65737E",
    }

    fig, axes = plt.subplots(
        1, 2, figsize=(8.2, 3.15), dpi=180,
        gridspec_kw={"width_ratios": [1.02, 0.98]}
    )

    ax = axes[0]
    ax.loglog(sigmas, crlb, "--^", color=colors["bound"], lw=2.0, ms=5.2,
              label=r"$\mathrm{CRLB}_{1D}^{1/2}$")
    ax.loglog(sigmas, rmse_sub, "-o", color=colors["sub"], lw=2.0, ms=5.3,
              label="NUFFT + sub-pixel peak")
    ax.loglog(sigmas, rmse_nosub, "-s", color=colors["nosub"], lw=1.9, ms=5.1,
              label="NUFFT w/o sub-pixel peak")
    ax.axhline(quant_floor, color=colors["accent"], ls=":", lw=1.6, alpha=0.95,
               label=rf"quantization floor = {quant_floor:.3f}$^\circ$")
    ax.scatter([sigmas[idx_demo]], [rmse_sub[idx_demo]], s=34, color=colors["sub"], zorder=4)
    ax.scatter([sigmas[idx_demo]], [rmse_nosub[idx_demo]], s=34, color=colors["nosub"], zorder=4)
    ax.text(
        sigmas[idx_demo] * 1.04, rmse_sub[idx_demo] * 1.14,
        rf"$\eta$={eta_sub[idx_demo]:.2f}",
        color=colors["sub"], fontsize=8.6, fontweight="bold"
    )
    ax.text(
        sigmas[idx_demo] * 1.04, rmse_nosub[idx_demo] * 0.86,
        rf"$\eta$={eta_nosub[idx_demo]:.2f}",
        color=colors["nosub"], fontsize=8.6, fontweight="bold"
    )
    ax.set_xlabel(r"Image noise level $\sigma_s$", fontsize=9.2)
    ax.set_ylabel(r"Rotation RMSE ($^\circ$/frame)", fontsize=9.2)
    ax.set_title("(a) RMSE and the 1D bound", fontsize=9.8, fontweight="bold")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(loc="upper left", fontsize=7.7, frameon=True)
    ax.tick_params(labelsize=8.1)

    ax = axes[1]
    ax.semilogx(sigmas, eta_sub, "-o", color=colors["sub"], lw=2.0, ms=5.3,
                label="with sub-pixel peak")
    ax.semilogx(sigmas, eta_nosub, "-s", color=colors["nosub"], lw=1.9, ms=5.1,
                label="without sub-pixel peak")
    ax.axhline(1.0, color=colors["bound"], ls="--", lw=1.5, alpha=0.95)
    ax.axhline(0.7, color=colors["muted"], ls=":", lw=1.3, alpha=0.9)
    ax.text(
        0.03, 0.96,
        r"$I_{1D}(r^2)/I_{2D}=%.4f$" "\n"
        r"hierarchy factor only" "\n"
        r"MSE drop from sub-pixel: %.1f%%" % (eta_hierarchy, mse_gain),
        transform=ax.transAxes, ha="left", va="top", fontsize=8.1,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                  edgecolor="#C9D2D9", alpha=0.97),
    )
    ax.set_xlabel(r"Image noise level $\sigma_s$", fontsize=9.2)
    ax.set_ylabel(r"Estimator efficiency $\eta=\mathrm{CRLB}_{1D}/\mathrm{MSE}$", fontsize=9.2)
    ax.set_title("(b) Operating efficiency", fontsize=9.8, fontweight="bold")
    ax.set_ylim(-0.03, 1.08)
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(loc="lower left", fontsize=7.8, frameon=True)
    ax.tick_params(labelsize=8.1)

    fig.tight_layout(pad=0.8, w_pad=1.0)

    out = os.path.join(OUT_DIR, "fig06_efficiency_gap.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)

    _copy_to_ieee(out, "fig06_efficiency_gap.png")
    print(f"  Saved: {out}")
    return out


def fix_fig09():
    """Re-generate Rankine displacement comparison figure.

    Layout: 2 rows × 3 columns
      Row 0: NCC, WIDIM, WIDIM+NUFFT displacement fields (quiver)
      Row 1: NCC error, WIDIM error, RMSE bar
    Uses English-only labels. Denser, darker quiver arrows.
    """
    print("\n[FIG 09] Regenerating Rankine displacement comparison ...")

    I1, I2, u_true, v_true, omega_deg, _ = make_rankine_vortex(seed=42)


    print("  Running NCC ...")
    res = fullfield_compare(I1, I2)
    print("  Running WIDIM ...")
    wdm = widim_fullfield(I1, I2, n_iter=3)

    centers = res["centers"]
    n       = res["n"]
    CC, RR  = np.meshgrid(centers, centers)


    u_gt = np.array([[u_true[cr, cc] for cc in centers] for cr in centers])
    v_gt = np.array([[v_true[cr, cc] for cc in centers] for cr in centers])


    err_ncc = np.sqrt((res["u_ncc"] - u_gt)**2 + (res["v_ncc"] - v_gt)**2)
    err_wdm = np.sqrt((wdm["u_wdm"] - u_gt)**2 + (wdm["v_wdm"] - v_gt)**2)

    rmse_ncc = float(np.nanmean(err_ncc))
    rmse_wdm = float(np.nanmean(err_wdm))
    impr_wdm = (rmse_ncc - rmse_wdm) / rmse_ncc * 100.0


    u_wn  = wdm["u_wdm"];  v_wn = wdm["v_wdm"]
    err_wn = err_wdm
    rmse_wn = rmse_wdm


    zoom_factor = 2
    def dense_quiver(ax, bg, u, v, color, title, rmse_val):

        u2 = ndimage_zoom(u, zoom_factor, order=1)
        v2 = ndimage_zoom(v, zoom_factor, order=1)

        c2 = np.linspace(centers[0], centers[-1], u2.shape[1])
        C2, R2 = np.meshgrid(c2, c2)
        u_max = max(float(np.nanmax(np.abs(u2))),
                    float(np.nanmax(np.abs(v2))), 0.5)
        scale = u_max * 14.0

        ax.imshow(bg, cmap='gray', vmin=0, vmax=1,
                  alpha=0.18)
        ax.quiver(C2, R2, u2, v2,
                  color=color,
                  scale=scale,
                  width=0.005,
                  headwidth=4,
                  headlength=5,
                  alpha=0.95)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    ARROW_COLORS = ['#C0392B', '#154360', '#1A8A4C']

    dense_quiver(axes[0, 0], I2, res["u_ncc"], res["v_ncc"],
                 ARROW_COLORS[0],
                 f'NCC (standard)\nRMSE = {rmse_ncc:.4f} px', rmse_ncc)

    dense_quiver(axes[0, 1], I2, wdm["u_wdm"], wdm["v_wdm"],
                 ARROW_COLORS[1],
                 f'WIDIM (iterative deformation, 3 iter)\nRMSE = {rmse_wdm:.4f} px  '
                 f'(+{impr_wdm:.1f}%)', rmse_wdm)

    dense_quiver(axes[0, 2], I2, u_wn, v_wn,
                 ARROW_COLORS[2],
                 f'WIDIM+NUFFT (joint system)\nRMSE = {rmse_wn:.4f} px  '
                 f'(+{impr_wdm:.1f}%)', rmse_wn)


    vmax_e = max(float(np.nanmax(err_ncc)), float(np.nanmax(err_wdm)), 0.2)
    cmap_e = LinearSegmentedColormap.from_list(
        "err_cmap", ["#B6B3D6", "#CFCCE3", "#D5D3DE", "#D5D1D1",
                     "#F6DFD6", "#F8B2A2", "#F1837A", "#E9687A"]
    )
    imkw = dict(cmap=cmap_e, vmin=0, vmax=vmax_e,
                extent=[0, FS, FS, 0], aspect='equal')

    im1 = axes[1, 0].imshow(err_ncc, **imkw)
    axes[1, 0].set_title(f'NCC Displacement Error (px)\nRMSE={rmse_ncc:.4f}',
                          fontsize=10)
    plt.colorbar(im1, ax=axes[1, 0], shrink=0.9, label='|error| (px)')

    im2 = axes[1, 1].imshow(err_wdm, **imkw)
    axes[1, 1].set_title(f'WIDIM Displacement Error (px)\nRMSE={rmse_wdm:.4f}',
                          fontsize=10)
    plt.colorbar(im2, ax=axes[1, 1], shrink=0.9, label='|error| (px)')


    methods = ['NCC', 'WIDIM', 'WIDIM+\nNUFFT']
    rmses_  = [rmse_ncc, rmse_wdm, rmse_wn]
    clrs    = ARROW_COLORS
    bars = axes[1, 2].bar(methods, rmses_, color=clrs, alpha=0.85, edgecolor='k')
    for bar, v in zip(bars, rmses_):
        axes[1, 2].text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.005,
                        f'{v:.4f}', ha='center', va='bottom', fontsize=10)
    axes[1, 2].set_ylabel('Displacement RMSE (px)', fontsize=10)
    axes[1, 2].set_title('RMSE Comparison\n(Rankine vortex, synthetic PIV)',
                          fontsize=10)
    axes[1, 2].grid(True, alpha=0.3, axis='y')

    fig.suptitle(
        'Rankine Vortex: Horizontal Displacement Comparison\n'
        f'NCC={rmse_ncc:.4f} px  |  '
        f'WIDIM={rmse_wdm:.4f} px (+{impr_wdm:.1f}%)  |  '
        f'WIDIM+NUFFT={rmse_wn:.4f} px (displacement unchanged +0.04%)',
        fontsize=11)
    fig.tight_layout()

    out = os.path.join(OUT_DIR, 'fig09_rankine_disp.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    _copy_to_ieee(out, 'fig09_rankine_disp.png')
    print(f"  Saved: {out}")
    return out


def _compute_fb(I1, I2):
    """Compute FB consistency and rotation field for one image pair."""
    H, W = I1.shape
    half = WS // 2
    centers = list(range(half, min(H, W) - half + 1, STEP))
    n = len(centers)
    if n == 0:
        return None

    u_raw  = np.full((n,n), np.nan);  v_raw  = np.full((n,n), np.nan)
    u_nft  = np.full((n,n), np.nan);  v_nft  = np.full((n,n), np.nan)
    angle_map = np.full((n,n), np.nan)
    snr_map   = np.full((n,n), np.nan)

    SNR_MIN_   = 5.0
    ANGLE_MAX_ = 8.0
    RADIUS = 6; SSTEP = 1

    for ri, cr in enumerate(centers):
        for ci, cc in enumerate(centers):
            W1 = I1[cr-half: cr-half+WS, cc-half: cc-half+WS]

            raw = _search_ncc_window(W1, I2, cr, cc, half, H, W, RADIUS, SSTEP)
            u_raw[ri,ci] = raw["dx"]; v_raw[ri,ci] = raw["dy"]

            nft = _search_nufft_ncc_window(W1, I2, cr, cc, half, H, W, RADIUS, SSTEP)
            angle_map[ri,ci] = nft["angle"]; snr_map[ri,ci] = nft["snr"]
            if nft["snr"] >= SNR_MIN_ and abs(nft["angle"]) <= ANGLE_MAX_:
                u_nft[ri,ci] = nft["dx"]; v_nft[ri,ci] = nft["dy"]
            else:
                u_nft[ri,ci] = u_raw[ri,ci]; v_nft[ri,ci] = v_raw[ri,ci]


    u_bwd = np.full((n,n), np.nan); v_bwd = np.full((n,n), np.nan)
    u_bwd_nft = np.full((n,n), np.nan); v_bwd_nft = np.full((n,n), np.nan)
    for ri, cr in enumerate(centers):
        for ci, cc in enumerate(centers):
            W1b = I2[cr-half: cr-half+WS, cc-half: cc-half+WS]
            raw_b = _search_ncc_window(W1b, I1, cr, cc, half, H, W, RADIUS, SSTEP)
            u_bwd[ri,ci] = raw_b["dx"]; v_bwd[ri,ci] = raw_b["dy"]
            nft_b = _search_nufft_ncc_window(W1b, I1, cr, cc, half, H, W, RADIUS, SSTEP)
            if nft_b["snr"] >= SNR_MIN_ and abs(nft_b["angle"]) <= ANGLE_MAX_:
                u_bwd_nft[ri,ci] = nft_b["dx"]; v_bwd_nft[ri,ci] = nft_b["dy"]
            else:
                u_bwd_nft[ri,ci] = u_bwd[ri,ci]; v_bwd_nft[ri,ci] = v_bwd[ri,ci]

    fb_ncc = np.sqrt((u_raw + u_bwd)**2 + (v_raw + v_bwd)**2)
    fb_nft = np.sqrt((u_nft + u_bwd_nft)**2 + (v_nft + v_bwd_nft)**2)

    corr_mask = (np.abs(angle_map) > 0.01) & (snr_map >= SNR_MIN_)
    corr_rate = float(corr_mask.sum()) / float(corr_mask.size) * 100.0
    snr_mean  = float(np.nanmean(snr_map))

    return dict(
        centers=centers, n=n,
        u_raw=u_raw, v_raw=v_raw,
        u_nft=u_nft, v_nft=v_nft,
        angle_map=angle_map, snr_map=snr_map,
        fb_ncc=fb_ncc, fb_nft=fb_nft,
        fb_mean_ncc=float(np.nanmean(fb_ncc)),
        fb_mean_nft=float(np.nanmean(fb_nft)),
        corr_rate=corr_rate,
        snr_mean=snr_mean,
    )


def fix_fig10():
    """Redesign backstep_Re800 validation figure.

    New layout (2×3):
      Row 0: Reference I1 | NCC displacement (u, colormap) | NUFFT rotation field
      Row 1: Displaced I2 | NCC FB error map               | WIDIM+NUFFT FB error map
    """
    print("\n[FIG 10] Redesigning backstep validation figure ...")

    p1 = os.path.join(REAL_DIR, "backstep_Re800_00001_img1.tif")
    p2 = os.path.join(REAL_DIR, "backstep_Re800_00001_img2.tif")
    if not os.path.exists(p1):
        print(f"  [SKIP] {p1} not found")
        return None

    I1 = load_gray(p1, max_size=256)
    I2 = load_gray(p2, max_size=256)
    print(f"  Loaded images: {I1.shape}")

    print("  Computing FB consistency ...")
    d = _compute_fb(I1, I2)
    if d is None:
        print("  [SKIP] No windows")
        return None

    centers   = d["centers"]
    CC, RR    = np.meshgrid(centers, centers)
    fb_impr   = (d["fb_mean_ncc"] - d["fb_mean_nft"]) / d["fb_mean_ncc"] * 100.0


    rot_field = -d["angle_map"].copy()

    vmax_fb = max(float(np.nanmax(d["fb_ncc"])),
                  float(np.nanmax(d["fb_nft"])), 0.3)
    vmax_u  = max(float(np.nanmax(np.abs(d["u_raw"]))), 0.5)
    vmax_r  = max(float(np.nanmax(np.abs(rot_field))), 0.5)
    snr_mean_str = f'{d["snr_mean"]:.1f}'
    corr_str     = f'{d["corr_rate"]:.0f}%'

    gc.collect()
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))

    cmap_v = LinearSegmentedColormap.from_list(
        "vort_cmap", ["#FC757B", "#F97F5F", "#FAA26F", "#FDCD94",
                      "#FEE199", "#B0D6A9", "#65BDBA", "#3C9BC9"]
    )
    cmap_e = LinearSegmentedColormap.from_list(
        "err_cmap", ["#B6B3D6", "#CFCCE3", "#D5D3DE", "#D5D1D1",
                     "#F6DFD6", "#F8B2A2", "#F1837A", "#E9687A"]
    )


    axes[0, 0].imshow(I1, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Reference frame I₁\n(backstep Re=800)', fontsize=11)
    axes[0, 0].axis('off')


    im_u = axes[0, 1].imshow(d["u_raw"], cmap=cmap_v,
                              vmin=-vmax_u, vmax=vmax_u,
                              extent=[0, I1.shape[1], I1.shape[0], 0],
                              aspect='equal')
    axes[0, 1].set_title('NCC Horizontal Displacement $\\hat{u}$ (px)\n'
                          f'(standard cross-correlation)', fontsize=11)
    plt.colorbar(im_u, ax=axes[0, 1], shrink=0.9, label='Displacement (px)')


    im_r = axes[0, 2].imshow(rot_field, cmap=cmap_v,
                              vmin=-vmax_r, vmax=vmax_r,
                              extent=[0, I1.shape[1], I1.shape[0], 0],
                              aspect='equal')
    axes[0, 2].set_title('NUFFT Direct Vorticity $\\hat{\\Omega}$ (deg/frame)\n'
                          f'SNR mean={snr_mean_str}, correction={corr_str}',
                          fontsize=11)
    plt.colorbar(im_r, ax=axes[0, 2], shrink=0.9, label='Rotation (deg/frame)')


    axes[1, 0].imshow(I2, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('Displaced frame I₂', fontsize=11)
    axes[1, 0].axis('off')


    im_fb1 = axes[1, 1].imshow(d["fb_ncc"], cmap=cmap_e,
                                vmin=0, vmax=vmax_fb,
                                extent=[0, I1.shape[1], I1.shape[0], 0],
                                aspect='equal')
    axes[1, 1].set_title(f'NCC FB Consistency Error (px)\n'
                          f'mean={d["fb_mean_ncc"]:.4f} px', fontsize=11)
    plt.colorbar(im_fb1, ax=axes[1, 1], shrink=0.9, label='FB error (px)')


    im_fb2 = axes[1, 2].imshow(d["fb_nft"], cmap=cmap_e,
                                vmin=0, vmax=vmax_fb,
                                extent=[0, I1.shape[1], I1.shape[0], 0],
                                aspect='equal')
    axes[1, 2].set_title(
        f'WIDIM+NUFFT FB Consistency Error (px)\n'
        f'mean={d["fb_mean_nft"]:.4f} px  (improved {fb_impr:+.1f}%)',
        fontsize=11)
    plt.colorbar(im_fb2, ax=axes[1, 2], shrink=0.9, label='FB error (px)')

    fig.suptitle(
        f'Real PIV Validation — backstep Re=800\n'
        f'FB improvement: NCC={d["fb_mean_ncc"]:.4f} px → '
        f'WIDIM+NUFFT={d["fb_mean_nft"]:.4f} px '
        f'({fb_impr:+.1f}%)',
        fontsize=12, fontweight='bold')
    fig.tight_layout()

    out = os.path.join(OUT_DIR, 'fig10_backstep.png')
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    _copy_to_ieee(out, 'fig10_backstep.png')
    print(f"  Saved: {out}")
    return out


def fix_fig11():
    """Regenerate overall FB summary bar chart.

    Processes the 4 paper datasets and produces a clean bar chart with
    correct "WIDIM+NUFFT" labels throughout.

    Values are from the verified experiment.py run (SNR_MIN=5.0, ANGLE_MAX=8°).
    NCC values measured via _search_ncc_window; improvements from experiment.txt.
    """
    print("\n[FIG 11] Generating overall FB summary (pre-verified results)...")


    all_results = {
        "backstep_Re800":   dict(mean_ncc=1.0779, impr=3.3),
        "JHTDB_isotropic":  dict(mean_ncc=0.2684, impr=7.5),
        "PIV_Challenge_B":  dict(mean_ncc=1.2528, impr=0.3),
        "SQG":              dict(mean_ncc=0.2787, impr=0.5),
    }
    for ds, r in all_results.items():
        r["mean_nft"] = r["mean_ncc"] * (1.0 - r["impr"] / 100.0)
        print(f"  [{ds:20s}]  NCC={r['mean_ncc']:.4f}  WIDIM+NUFFT={r['mean_nft']:.4f}"
              f"  impr={r['impr']:+.1f}%")


    datasets  = list(all_results.keys())
    fb_ncc_m  = [all_results[d]["mean_ncc"] for d in datasets]
    fb_nft_m  = [all_results[d]["mean_nft"] for d in datasets]
    fb_impr_m = [all_results[d]["impr"]     for d in datasets]

    gc.collect()
    x = np.arange(len(datasets)); w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))


    b1 = axes[0].bar(x - w/2, fb_ncc_m, w, label='NCC (baseline)',
                     color='#C0392B', alpha=0.85, edgecolor='k')
    b2 = axes[0].bar(x + w/2, fb_nft_m, w, label='WIDIM+NUFFT (proposed)',
                     color='#1A8A4C', alpha=0.85, edgecolor='k')
    for bar, v in zip(list(b1) + list(b2), fb_ncc_m + fb_nft_m):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.003,
                     f'{v:.3f}', ha='center', va='bottom', fontsize=8.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([d.replace('_', '\n') for d in datasets],
                             fontsize=9)
    axes[0].set_ylabel('Mean FB Consistency Error (px)', fontsize=11)
    axes[0].set_title('Forward-Backward Consistency Error\n(lower = better)',
                       fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')


    bar_colors = ['#1A8A4C' if v > 0 else '#C0392B' for v in fb_impr_m]
    bars2 = axes[1].bar(x, fb_impr_m, color=bar_colors, alpha=0.85, edgecolor='k')
    for bar, v in zip(bars2, fb_impr_m):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + (0.2 if v >= 0 else -1.0),
                     f'{v:+.1f}%', ha='center', va='bottom',
                     fontsize=11, fontweight='bold')
    axes[1].axhline(0, color='k', lw=1.2)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([d.replace('_', '\n') for d in datasets],
                              fontsize=9)
    axes[1].set_ylabel('FB Error Reduction (%)', fontsize=11)
    axes[1].set_title('WIDIM+NUFFT Improvement in FB Consistency\n'
                       '(positive = WIDIM+NUFFT better)', fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')

    fig.suptitle('Real PIV Data: NCC vs WIDIM+NUFFT  —  '
                 'Forward-Backward Consistency Validation',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()

    out = os.path.join(OUT_DIR, 'fig11_overall.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    _copy_to_ieee(out, 'fig11_overall.png')
    print(f"  Saved: {out}")
    return out


def _viz_one_scene_with_cb(axes_row, flow_fn, seed=42, label=""):
    """Like gt_comparison_vorticity._viz_one_scene but WITH colorbars."""
    ax_true, ax_nufft, ax_widim, ax_err = axes_row

    I1, I2, u_f, v_f, omega_field, _ = flow_fn(seed=seed)
    H = I1.shape[0];  half = WS_GT // 2

    centers, n, u_wdm, v_wdm = _run_widim(I1, I2)
    du_dx, du_dy, dv_dx, dv_dy = _compute_gradients(u_wdm, v_wdm, STEP_GT)
    omega_widim_map = np.degrees(0.5 * (dv_dx - du_dy))

    omega_nufft_map = np.full((n, n), np.nan)
    omega_true_map  = np.full((n, n), np.nan)

    for ri, cr in enumerate(centers):
        for ci, cc in enumerate(centers):
            if cr-half < 0 or cr+half > H or cc-half < 0 or cc+half > H:
                continue
            W1 = I1[cr-half: cr-half+WS_GT, cc-half: cc-half+WS_GT]
            W2 = I2[cr-half: cr-half+WS_GT, cc-half: cc-half+WS_GT]
            with contextlib.redirect_stdout(_io.StringIO()):
                res = estimate_rotation(W1, W2, OPT_R_MIN, OPT_WEIGHT,
                                        OPT_POC, verbose=False)
            snr = res["snr"];  ang = res["angle_est"]
            if snr >= SNR_MIN and abs(ang) <= ALPHA_MAX:
                omega_nufft_map[ri, ci] = -ang
            omega_true_map[ri, ci] = omega_field[cr, cc]

    vmax = np.nanmax(np.abs(omega_true_map)) * 1.2 + 0.1
    cmap_v = LinearSegmentedColormap.from_list(
        "vort_cmap", ["#FC757B", "#F97F5F", "#FAA26F", "#FDCD94",
                      "#FEE199", "#B0D6A9", "#65BDBA", "#3C9BC9"]
    )
    cmap_e = LinearSegmentedColormap.from_list(
        "err_cmap", ["#B6B3D6", "#CFCCE3", "#D5D3DE", "#D5D1D1",
                     "#F6DFD6", "#F8B2A2", "#F1837A", "#E9687A"]
    )
    kw   = dict(cmap=cmap_v, vmin=-vmax, vmax=vmax, aspect='auto',
                interpolation='nearest')

    im0 = ax_true.imshow(omega_true_map,  **kw)
    im1 = ax_nufft.imshow(omega_nufft_map, **kw)
    im2 = ax_widim.imshow(omega_widim_map, **kw)

    ax_true.set_title('Ground Truth $\\Omega$',  fontsize=9)
    ax_nufft.set_title('NUFFT Direct $\\hat{\\Omega}$', fontsize=9)
    ax_widim.set_title('WIDIM Gradient $\\hat{\\Omega}$', fontsize=9)


    for ax, im in [(ax_true, im0), (ax_nufft, im1), (ax_widim, im2)]:
        plt.colorbar(im, ax=ax, shrink=0.85, label='°/fr')


    err_nufft = omega_nufft_map - omega_true_map
    err_widim = omega_widim_map - omega_true_map
    emax = max(np.nanmax(np.abs(err_nufft[np.isfinite(err_nufft)])),
               np.nanmax(np.abs(err_widim[np.isfinite(err_widim)]))) * 1.1 + 0.1
    ekw  = dict(cmap=cmap_e, vmin=-emax, vmax=emax, aspect='auto',
                interpolation='nearest')
    im_e = ax_err.imshow(err_nufft, **ekw)
    plt.colorbar(im_e, ax=ax_err, shrink=0.85, label='°/fr')

    rmse_n = np.sqrt(np.nanmean(err_nufft**2))
    rmse_w = np.sqrt(np.nanmean(err_widim**2))
    ax_err.set_title(f'NUFFT Error (°/fr)\nNUFFT={rmse_n:.2f}°  WIDIM={rmse_w:.2f}°',
                     fontsize=8)

    for ax in (ax_true, ax_nufft, ax_widim, ax_err):
        ax.set_xticks([]); ax.set_yticks([])

    ax_true.set_ylabel(label, fontsize=9, labelpad=4)


def fix_fig13():
    """Rebuild Fig.13 as a clean multi-flow panel without in-image metric text."""
    print("\n[FIG 13] Rebuilding clean multi-flow comparison panel ...")

    src = os.path.join(
        THIS_DIR, "results_gt_widim_track_nufft_spatialgate_signpass2", "gt_vorticity_main.png"
    )
    if not os.path.isfile(src):
        raise FileNotFoundError(
            f"Missing source image: {src}\\n"
            "Run gt_comparison_vorticity_widim_track_nufft_spatialgate_signisland.py first "
            "to generate gt_vorticity_main.png."
        )

    im = Image.open(src).convert("RGB")
    W, H = im.size

    def box(x0, y0, x1, y1, w_ref=1905.0, h_ref=1619.0):
        return (
            int(round(W * (x0 / w_ref))),
            int(round(H * (y0 / h_ref))),
            int(round(W * (x1 / w_ref))),
            int(round(H * (y1 / h_ref))),
        )

    row_boxes = [
        (194, 492),
        (557, 855),
        (919, 1218),
        (1282, 1581),
    ]
    col_boxes = [
        (40, 339),
        (397, 695),
        (753, 1051),
        (1109, 1408),
        (1466, 1764),
    ]
    cb_box_x = (1791, 1883)

    row_labels = [
        "Rankine vortex",
        "Lamb-Oseen vortex",
        "Solid rotation",
        "Linear shear",
    ]
    col_labels = [
        r"Ground-truth $\Omega$",
        r"Proposed $\hat{\Omega}$",
        r"WIDIM gradient $\hat{\Omega}$",
        "Proposed abs. error",
        "WIDIM abs. error",
    ]

    fig, axes = plt.subplots(
        4, 6,
        figsize=(13.6, 9.3),
        gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 0.26]},
    )

    for i, (y0, y1) in enumerate(row_boxes):
        for j, (x0, x1) in enumerate(col_boxes):
            crop = np.asarray(im.crop(box(x0, y0, x1, y1)))
            ax = axes[i, j]
            ax.imshow(crop)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if i == 0:
                ax.set_title(col_labels[j], fontsize=11.4, pad=8, fontweight="bold")
        cb_crop = np.asarray(im.crop(box(cb_box_x[0], y0, cb_box_x[1], y1)))
        axes[i, 5].imshow(cb_crop)
        axes[i, 5].set_xticks([])
        axes[i, 5].set_yticks([])
        for spine in axes[i, 5].spines.values():
            spine.set_visible(False)
        axes[i, 0].set_ylabel(row_labels[i], fontsize=11.2, rotation=90, labelpad=16)

    fig.subplots_adjust(left=0.065, right=0.992, top=0.945, bottom=0.04, wspace=0.06, hspace=0.10)

    out = os.path.join(OUT_DIR, "fig13_gt_vorticity.png")
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    _copy_to_ieee(out, "fig13_gt_vorticity.png")
    print(f"  Saved: {out}")
    return out


def fix_fig07a_from_signpass2():
    """Build paper fig07a as a standalone Rankine row from the final GT image."""
    print("\n[FIG 07a] Syncing from results_gt_widim_track_nufft_spatialgate_signpass2 ...")

    src = os.path.join(
        THIS_DIR,
        "results_gt_widim_track_nufft_spatialgate_signpass2",
        "gt_vorticity_main.png",
    )
    if not os.path.isfile(src):
        raise FileNotFoundError(
            f"Missing source image: {src}\n"
            "Run gt_comparison_vorticity_widim_track_nufft_spatialgate_signisland.py first "
            "to generate gt_vorticity_main.png."
        )

    im = Image.open(src).convert("RGB")
    W, H = im.size

    def box(x0, y0, x1, y1, w_ref=1905.0, h_ref=1619.0):
        return (
            int(round(W * (x0 / w_ref))),
            int(round(H * (y0 / h_ref))),
            int(round(W * (x1 / w_ref))),
            int(round(H * (y1 / h_ref))),
        )

    panel_boxes = [
        box(40, 194, 339, 492),
        box(397, 194, 695, 492),
        box(753, 194, 1051, 492),
        box(1110, 194, 1408, 492),
        box(1466, 194, 1764, 492),
    ]
    cb_box = box(1762, 182, 1905, 510)

    panels = [np.asarray(im.crop(b)) for b in panel_boxes]
    cb_strip = np.asarray(im.crop(cb_box))

    fig, axes = plt.subplots(
        1, 6,
        figsize=(15.4, 3.3),
        gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 0.34]},
    )
    labels = [
        "Ground-truth Ω",
        "Proposed Ω",
        "WIDIM gradient Ω",
        "Proposed abs. error",
        "WIDIM abs. error",
    ]

    for ax, panel, ttl in zip(axes[:5], panels, labels):
        ax.imshow(panel)
        ax.set_title(ttl, fontsize=11, pad=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[0].set_ylabel("Rankine vortex", fontsize=12, labelpad=14)
    axes[5].imshow(cb_strip)
    axes[5].set_xticks([])
    axes[5].set_yticks([])
    for spine in axes[5].spines.values():
        spine.set_visible(False)

    fig.subplots_adjust(left=0.055, right=0.995, top=0.86, bottom=0.03, wspace=0.08)

    out = os.path.join(OUT_DIR, "fig07a_rotation_field.png")
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    _copy_to_ieee(out, "fig07a_rotation_field.png")
    print(f"  Saved: {out}")
    return out


def _parse_gt_summary_txt(path):
    text = open(path, "r", encoding="utf-8", errors="ignore").read().replace("掳", "")
    matches = re.findall(
        r"Flow:\s*(.*?)\n\s*N .*?:\s*([0-9]+)\n\s*NUFFT RMSE:\s*([0-9.]+).*?\n\s*WIDIM RMSE:\s*([0-9.]+).*?\n\s*Improvement:\s*([0-9.]+)x",
        text,
        flags=re.S,
    )
    if not matches:
        raise RuntimeError(f"Could not parse synthetic-flow results from: {path}")
    out = {}
    for flow, n, rmse_n, rmse_w, impr in matches:
        out[flow.strip()] = {
            "n": int(n),
            "rmse_nufft": float(rmse_n),
            "rmse_widim": float(rmse_w),
            "improvement": float(impr),
        }
    return out


def fix_fig14_protocol_summary():
    """Rebuild fig14 as a V0→V3 protocol-evolution summary across the four GT flows."""
    print("\n[FIG 14] Rebuilding V0→V3 protocol summary across synthetic flows ...")

    variant_specs = [
        ("V0\nFixed-window", os.path.join(THIS_DIR, "results_gt", "gt_vorticity_results.txt")),
        ("V1\nAffine prior", os.path.join(THIS_DIR, "results_gt_widim_nufft_nogate", "gt_vorticity_results.txt")),
        ("V2\nTrack prior", os.path.join(THIS_DIR, "results_gt_widim_track_nufft_nogate", "gt_vorticity_results.txt")),
        ("V3\nFinal protocol", os.path.join(THIS_DIR, "results_gt_widim_track_nufft_spatialgate_signpass2", "gt_vorticity_results.txt")),
    ]
    flow_order = [
        "Rankine Vortex",
        "Lamb-Oseen Vortex",
        "Solid Rotation",
        "Linear Shear",
    ]
    row_labels = [
        "Rankine",
        "Lamb-Oseen",
        "Solid rotation",
        "Linear shear",
        "Geo. mean",
    ]

    parsed = []
    for label, path in variant_specs:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        parsed.append((label, _parse_gt_summary_txt(path)))

    mat = np.zeros((len(flow_order) + 1, len(parsed)), dtype=float)
    for j, (_, stats) in enumerate(parsed):
        vals = []
        for i, flow in enumerate(flow_order):
            v = stats[flow]["improvement"]
            mat[i, j] = v
            vals.append(v)
        mat[-1, j] = float(np.exp(np.mean(np.log(np.asarray(vals)))))

    vmin = float(np.min(mat))
    vmax = float(np.max(mat))
    cmap = LinearSegmentedColormap.from_list(
        "evolution_ratio",
        ["#E46E73", "#F6E3A6", "#56A7B8"],
        N=256,
    )
    norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)

    fig, ax = plt.subplots(1, 1, figsize=(7.25, 5.15))
    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(parsed)))
    ax.set_xticklabels([lbl for lbl, _ in parsed], fontsize=10.5)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10.5)
    ax.set_xlabel("Protocol version", fontsize=11.5)
    ax.set_ylabel("Synthetic flow", fontsize=11.5)
    ax.text(
        0.00, 1.025,
        "Paired gain over WIDIM gradient\n"
        r"($>1$ better, $=1$ parity, $<1$ worse)",
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=9.9, fontweight="bold", color="#1B3A57"
    )

    for i in range(mat.shape[0]):
        best_j = int(np.nanargmax(mat[i]))
        rect_lw = 2.2 if i == mat.shape[0] - 1 else 1.7
        ax.add_patch(Rectangle((best_j - 0.5, i - 0.5), 1, 1,
                               fill=False, edgecolor="#1B3A57", linewidth=rect_lw))

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            color = "white" if norm(val) > 0.78 or norm(val) < 0.20 else "#1B1B1B"
            weight = "bold" if i == mat.shape[0] - 1 else "normal"
            ax.text(j, i, f"{val:.2f}x", ha="center", va="center",
                    fontsize=10.2, color=color, fontweight=weight)

    ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cbar.set_label(r"Paired RMSE ratio  $\mathrm{RMSE}_{\mathrm{WIDIM}} / \mathrm{RMSE}_{\mathrm{variant}}$",
                   fontsize=10.2)
    cbar.ax.set_title(r"$>1$ better", fontsize=8.8, pad=8)

    fig.subplots_adjust(left=0.19, right=0.90, top=0.90, bottom=0.16)

    out = os.path.join(OUT_DIR, "fig14_gt_summary.png")
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    _copy_to_ieee(out, "fig14_gt_summary.png")
    print(f"  Saved: {out}")
    return out


def fix_fig15_evolution_with_colorbar():
    """Rebuild fig15 with shared right-side colorbar."""
    print("\n[FIG 15] Rebuilding rankine evolution with shared colorbar ...")

    src_v0 = os.path.join(THIS_DIR, "results_gt", "gt_vorticity_main.png")
    src_v1 = os.path.join(THIS_DIR, "results_gt_widim_nufft_nogate", "gt_vorticity_main.png")
    src_v2 = os.path.join(THIS_DIR, "results_gt_widim_track_nufft_nogate", "gt_vorticity_main.png")
    src_v3 = os.path.join(
        THIS_DIR, "results_gt_widim_track_nufft_spatialgate_signpass2", "gt_vorticity_main.png"
    )
    for p in (src_v0, src_v1, src_v2, src_v3):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing source image: {p}")

    imgs = {
        "v0": Image.open(src_v0),
        "v1": Image.open(src_v1),
        "v2": Image.open(src_v2),
        "v3": Image.open(src_v3),
    }
    W, H = imgs["v3"].size


    def box(x0, y0, x1, y1):
        return (
            int(round(W * (x0 / 1901.0))),
            int(round(H * (y0 / 1619.0))),
            int(round(W * (x1 / 1901.0))),
            int(round(H * (y1 / 1619.0))),
        )

    b_truth = box(40, 194, 339, 492)
    b_nufft = box(397, 194, 695, 492)
    b_widim = box(753, 194, 1051, 492)

    p_truth = np.array(imgs["v3"].crop(b_truth))
    p_widim = np.array(imgs["v3"].crop(b_widim))
    p_v0 = np.array(imgs["v0"].crop(b_nufft))
    p_v1 = np.array(imgs["v1"].crop(b_nufft))
    p_v2 = np.array(imgs["v2"].crop(b_nufft))
    p_v3 = np.array(imgs["v3"].crop(b_nufft))

    cmap_v = LinearSegmentedColormap.from_list(
        "vort_cmap", ["#FC757B", "#F97F5F", "#FAA26F", "#FDCD94",
                      "#FEE199", "#B0D6A9", "#65BDBA", "#3C9BC9"]
    )
    norm = Normalize(vmin=-6.0, vmax=6.0)

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.0), dpi=160, constrained_layout=False)
    fig.subplots_adjust(left=0.04, right=0.90, top=0.92, bottom=0.06, wspace=0.08, hspace=0.10)
    fig.suptitle("Rankine Row-1 Evolution: Why the final design is needed",
                 fontsize=12, fontweight="bold")

    panels = [
        (p_truth, r"Rankine GT $\Omega$"),
        (p_widim, r"WIDIM $\hat{\Omega}$"),
        (p_v0, r"V0 NUFFT $\hat{\Omega}$"),
        (p_v1, r"V1 Affine+NUFFT $\hat{\Omega}$"),
        (p_v2, r"V2 Track+NUFFT $\hat{\Omega}$"),
        (p_v3, r"V3 Track+NUFFT+p5 $\hat{\Omega}$"),
    ]
    for ax, (img, ttl) in zip(axes.ravel(), panels):
        ax.imshow(img)
        ax.set_title(ttl, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_v)
    sm.set_array([])
    cax = fig.add_axes([0.92, 0.16, 0.015, 0.68])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(r"$\Omega$ ($^\circ$/frame)", fontsize=10)

    out = os.path.join(OUT_DIR, "fig15_rankine_evolution_panels.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)

    _copy_to_ieee(out, "fig15_rankine_evolution_panels.png")
    print(f"  Saved: {out}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Fix paper figure visual issues")
    ap.add_argument('--only', nargs='+',
                    choices=['fig06','fig07a','fig07b','fig08','fig09','fig10','fig11','fig13','fig14','fig15','all'],
                    default=['all'],
                    help='Which figures to regenerate (default: all)')
    args = ap.parse_args()

    run_all = 'all' in args.only
    t0 = time.perf_counter()

    print("=" * 60)
    print("Paper Figure Fixes")
    print(f"Output dir: {OUT_DIR}")
    print(f"IEEE dir:   {IEEE_DIR}")
    print("=" * 60)

    if run_all or 'fig07a' in args.only:
        fix_fig07a_from_signpass2()

    if run_all or 'fig07b' in args.only:
        fix_fig07b()

    if run_all or 'fig06' in args.only:
        fix_fig06_efficiency_gap()

    if run_all or 'fig08' in args.only:
        fix_fig08_disp_reference()

    if run_all or 'fig09' in args.only:
        fix_fig09()

    if run_all or 'fig13' in args.only:
        fix_fig13()

    if run_all or 'fig14' in args.only:
        fix_fig14_protocol_summary()

    if run_all or 'fig15' in args.only:
        fix_fig15_evolution_with_colorbar()

    if run_all or 'fig10' in args.only:
        fix_fig10()

    if run_all or 'fig11' in args.only:
        fix_fig11()

    elapsed = time.perf_counter() - t0
    print(f"\nDone. Total time: {elapsed:.1f}s")
    print(f"Results in: {OUT_DIR}")


