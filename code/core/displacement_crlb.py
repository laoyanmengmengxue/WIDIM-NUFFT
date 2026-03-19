import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from comparison import (
    make_uniform_flow, widim_fullfield,
    _search_ncc_window, _residual_ncc,
    WS, STEP, SEARCH_RADIUS,
)
from poc_common import generate_speckle_field, DEFAULT_PARAMS, set_plot_style, COLORS

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results_displacement_crlb")
os.makedirs(OUT_DIR, exist_ok=True)


NOISE_LEVELS = [0.002, 0.005, 0.01, 0.015, 0.02]
N_MC         = 200
SEED_BASE    = 42
FS           = DEFAULT_PARAMS["field_size"]
HALF         = WS // 2
U0, V0       = 4.37, 3.21


def compute_crlb(I1_window, sigma_img):


    I = I1_window.astype(np.float64)
    gx = np.gradient(I, axis=1)
    gy = np.gradient(I, axis=0)
    Gx = float(np.sum(gx ** 2))
    Gy = float(np.sum(gy ** 2))
    eps = 1e-30
    crlb_x = sigma_img ** 2 / (Gx + eps)
    crlb_y = sigma_img ** 2 / (Gy + eps)
    return crlb_x, crlb_y


def _ncc_subpixel_estimate(W1, I2, cr, cc, half, H, W_img,
                           search_radius=SEARCH_RADIUS):


    raw = _search_ncc_window(W1, I2, cr, cc, half, H, W_img,
                             search_radius, 1)
    dx_int = int(round(raw["dx"]))
    dy_int = int(round(raw["dy"]))

    cr2 = max(half, min(H     - half, cr + dy_int))
    cc2 = max(half, min(W_img - half, cc + dx_int))
    W2_best = I2[cr2 - half: cr2 - half + WS,
                 cc2 - half: cc2 - half + WS]

    dx_sub, dy_sub = _residual_ncc(W1, W2_best, res_radius=3)
    return float(dx_int) + dx_sub, float(dy_int) + dy_sub


def run_experiment():
    set_plot_style()
    print("=" * 65)
    print("Displacement CRLB Verification: NCC vs WIDIM")
    print("=" * 65)
    print(f"\n  纯平移流场: u={U0:.2f}px, v={V0:.2f}px（亚像素分量）")
    print(f"  窗口大小: {WS}×{WS}px  图像大小: {FS}×{FS}px")
    print(f"  N_MC = {N_MC}  噪声水平: {NOISE_LEVELS}")

    rng = np.random.default_rng(SEED_BASE)


    I1_clean, I2_clean, _, _, _, _ = make_uniform_flow(
        seed=SEED_BASE, u0=U0, v0=V0)


    cr, cc = FS // 2, FS // 2
    W1_clean = I1_clean[cr - HALF: cr - HALF + WS,
                         cc - HALF: cc - HALF + WS]


    centers = list(range(HALF, FS - HALF + 1, STEP))
    idx_c   = len(centers) // 2

    results = {}

    for sigma_s in NOISE_LEVELS:
        print(f"\n  [σ_s={sigma_s:.3f}]  运行 {N_MC} 次 Monte Carlo ...")
        t0 = time.perf_counter()


        crlb_x, crlb_y = compute_crlb(W1_clean, sigma_s)
        crlb_rmse = float(np.sqrt((crlb_x + crlb_y) / 2.0))

        rmse_ncc_list = []
        rmse_wdm_list = []

        for mc in range(N_MC):

            noise1 = rng.normal(0, sigma_s, I1_clean.shape).astype(np.float32)
            noise2 = rng.normal(0, sigma_s, I2_clean.shape).astype(np.float32)
            I1n = np.clip(I1_clean + noise1, 0.0, 1.0)
            I2n = np.clip(I2_clean + noise2, 0.0, 1.0)


            cr_c, cc_c = FS // 2, FS // 2
            W1n = I1n[cr_c - HALF: cr_c - HALF + WS,
                      cc_c - HALF: cc_c - HALF + WS]
            u_ncc, v_ncc = _ncc_subpixel_estimate(
                W1n, I2n, cr_c, cc_c, HALF, FS, FS)
            rmse_ncc_list.append(
                float(np.sqrt((u_ncc - U0)**2 + (v_ncc - V0)**2)))


            wdm = widim_fullfield(I1n, I2n, n_iter=3)
            u_wdm = float(wdm["u_wdm"][idx_c, idx_c])
            v_wdm = float(wdm["v_wdm"][idx_c, idx_c])
            rmse_wdm_list.append(
                float(np.sqrt((u_wdm - U0)**2 + (v_wdm - V0)**2)))

        rmse_ncc = float(np.sqrt(np.mean(np.array(rmse_ncc_list)**2)))
        rmse_wdm = float(np.sqrt(np.mean(np.array(rmse_wdm_list)**2)))


        eta_ncc = min((crlb_rmse / rmse_ncc)**2, 1.0) if rmse_ncc > 1e-9 else 0.0
        eta_wdm = min((crlb_rmse / rmse_wdm)**2, 1.0) if rmse_wdm > 1e-9 else 0.0

        results[sigma_s] = dict(
            crlb_rmse=crlb_rmse,
            rmse_ncc=rmse_ncc,
            rmse_wdm=rmse_wdm,
            eta_ncc=eta_ncc,
            eta_wdm=eta_wdm,
        )

        elapsed = time.perf_counter() - t0
        print(f"    CRLB_rmse = {crlb_rmse*1000:.2f} mpx")
        print(f"    NCC  RMSE = {rmse_ncc*1000:.2f} mpx  η = {eta_ncc*100:.1f}%")
        print(f"    WIDIM RMSE = {rmse_wdm*1000:.2f} mpx  η = {eta_wdm*100:.1f}%")
        print(f"    ({elapsed:.1f}s)")


    print()
    print(f"  {'σ_s':>6} {'CRLB(mpx)':>10} {'NCC(mpx)':>10} "
          f"{'η_NCC':>8} {'WIDIM(mpx)':>11} {'η_WIDIM':>9}")
    print("  " + "-" * 60)
    for s in NOISE_LEVELS:
        r = results[s]
        print(f"  {s:>6.3f} {r['crlb_rmse']*1000:>10.2f} "
              f"{r['rmse_ncc']*1000:>10.2f} {r['eta_ncc']*100:>7.1f}% "
              f"{r['rmse_wdm']*1000:>10.2f} {r['eta_wdm']*100:>8.1f}%")

    _plot(results)
    return results


def _plot(results):
    sigmas    = NOISE_LEVELS
    crlb_vals = [results[s]["crlb_rmse"] for s in sigmas]
    r_ncc     = [results[s]["rmse_ncc"]  for s in sigmas]
    r_wdm     = [results[s]["rmse_wdm"]  for s in sigmas]
    eta_ncc   = [results[s]["eta_ncc"] * 100 for s in sigmas]
    eta_wdm   = [results[s]["eta_wdm"] * 100 for s in sigmas]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))


    ax = axes[0]
    ax.loglog(sigmas, crlb_vals, "k--", lw=2.5, label="CRLB (theory)")
    ax.loglog(sigmas, r_ncc,     "-o", color=COLORS[0], lw=1.8, ms=6, label="NCC (sub-px Gauss)")
    ax.loglog(sigmas, r_wdm,     "-s", color=COLORS[7], lw=1.8, ms=6, label="WIDIM (3-iter)")
    ax.set_xlabel("Image noise level σ_s", fontsize=11)
    ax.set_ylabel("Displacement RMSE (px)", fontsize=11)
    ax.set_title("Displacement RMSE vs Noise Level\n(NCC: Gaussian sub-px fitting, WS=64px)",
                 fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)


    ax2 = axes[1]
    ax2.semilogx(sigmas, eta_ncc, "-o", color=COLORS[0], lw=1.8, ms=6, label="NCC (sub-px)")
    ax2.semilogx(sigmas, eta_wdm, "-s", color=COLORS[7], lw=1.8, ms=6, label="WIDIM (3-iter)")
    ax2.axhline(100, color="k", ls="--", lw=1.5, label="CRLB 100%")
    ax2.set_xlabel("Image noise level σ_s", fontsize=11)
    ax2.set_ylabel("CRLB Efficiency η (%)", fontsize=11)
    ax2.set_title("Displacement CRLB Efficiency vs Noise\n(Higher η = closer to optimal)",
                  fontsize=10)
    ax2.set_ylim(0, 120)
    ax2.legend(fontsize=10)
    ax2.grid(True, which="both", alpha=0.3)


    ax3 = axes[2]
    ref_s   = 0.01
    e_ncc   = results[ref_s]["eta_ncc"] * 100
    e_wdm   = results[ref_s]["eta_wdm"] * 100

    e_nufft_rot = 93.0

    bars = ax3.bar(
        ["NCC(sub-px)\n(disp.)", "WIDIM(3-iter)\n(disp.)", "NUFFT\n(rot., DIC)"],
        [e_ncc, e_wdm, e_nufft_rot],
        color=[COLORS[0], COLORS[7], COLORS[5]],
        width=0.5,
        alpha=0.85
    )
    ax3.axhline(100, color="k", ls="--", lw=1.5, label="CRLB=100%")
    ax3.set_ylabel("CRLB Efficiency η (%)", fontsize=11)
    ax3.set_title(f"CRLB Efficiency at σ_s=0.01\n(Dual-CRLB summary for WIDIM+NUFFT)",
                  fontsize=10)
    ax3.set_ylim(0, 120)
    for bar, e in zip(bars, [e_ncc, e_wdm, e_nufft_rot]):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 2,
                 f"{e:.1f}%", ha="center", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=10)
    ax3.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Displacement CRLB Verification: NCC vs WIDIM\n"
        f"Pure translation (u={U0:.2f}px, v={V0:.2f}px), WS={WS}px, N_MC={N_MC}",
        fontsize=12, fontweight="bold"
    )
    fig.tight_layout()

    fname = os.path.join(OUT_DIR, "disp_crlb_comparison.png")
    fig.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {fname}")


if __name__ == "__main__":
    run_experiment()
