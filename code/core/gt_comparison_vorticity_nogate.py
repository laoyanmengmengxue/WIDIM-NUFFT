import os, sys, contextlib, io as _io, time, shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import map_coordinates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from poc_common import (
    generate_speckle_field, apply_affine, extract_window,
    DEFAULT_PARAMS, set_plot_style, COLORS,
)
from poc_point3 import estimate_rotation, OPT_R_MIN, OPT_WEIGHT, OPT_POC


FS     = DEFAULT_PARAMS["field_size"]
WS     = DEFAULT_PARAMS["window_size"]
PPP    = DEFAULT_PARAMS["ppp"]
DTAU   = DEFAULT_PARAMS["d_tau"]
STEP   = WS // 2
SIGMA_S = 0.01
SNR_MIN = 5.0
ALPHA_MAX = 8.0
N_SEEDS = 30
N_BOOT  = 1000
SEARCH_RADIUS = 6
SEARCH_STEP   = 1

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_gt_nogate")
os.makedirs(OUT_DIR, exist_ok=True)
IEEE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "IEEE-Transactions-LaTeX2e-templates-and-instructions")


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comparison import (
    _warp_by_flow, _ncc, _search_ncc_window, _compute_gradients,
    _normalized_median_test, _replace_outliers, _deform_window,
    _residual_ncc, _gauss_peak_1d,
)


def make_rankine(seed=42, Gamma=1500.0, R_core=50.0, sigma_s=SIGMA_S):

    I1_clean = generate_speckle_field(FS, PPP, DTAU, 1.0, seed)
    rng = np.random.default_rng(seed + 10000)
    I1 = np.clip(I1_clean + rng.normal(0, sigma_s, I1_clean.shape), 0, 1)

    Y, X = np.mgrid[0:FS, 0:FS].astype(float)
    rx = X - FS / 2.0;  ry = Y - FS / 2.0
    r  = np.hypot(rx, ry) + 1e-10
    Omega = Gamma / (2.0 * np.pi * R_core ** 2)
    v_t   = np.where(r <= R_core, Omega * r, Gamma / (2.0 * np.pi * r))
    u_f   = -v_t * ry / r;  v_f = v_t * rx / r

    I2_clean = _warp_by_flow(I1_clean, u_f, v_f)
    I2 = np.clip(I2_clean + rng.normal(0, sigma_s, I2_clean.shape), 0, 1)


    omega_true = np.where(r <= R_core, np.degrees(Omega), 0.0)
    label = f"Rankine (Γ={int(Gamma)}, R={int(R_core)} px)"
    return I1, I2, u_f, v_f, omega_true, label


def make_lamb_oseen(seed=42, Gamma=1500.0, sigma_core=40.0, sigma_s=SIGMA_S):


    I1_clean = generate_speckle_field(FS, PPP, DTAU, 1.0, seed)
    rng = np.random.default_rng(seed + 20000)
    I1 = np.clip(I1_clean + rng.normal(0, sigma_s, I1_clean.shape), 0, 1)

    Y, X = np.mgrid[0:FS, 0:FS].astype(float)
    rx = X - FS / 2.0;  ry = Y - FS / 2.0
    r2 = rx**2 + ry**2 + 1e-10
    r  = np.sqrt(r2)


    v_t = (Gamma / (2.0 * np.pi * r)) * (1.0 - np.exp(-r2 / sigma_core**2))
    u_f = -v_t * ry / r;  v_f = v_t * rx / r

    I2_clean = _warp_by_flow(I1_clean, u_f, v_f)
    I2 = np.clip(I2_clean + rng.normal(0, sigma_s, I2_clean.shape), 0, 1)


    omega_rad = (Gamma / (np.pi * sigma_core**2)) * np.exp(-r2 / sigma_core**2)
    omega_true = np.degrees(omega_rad)
    label = f"Lamb-Oseen (Γ={int(Gamma)}, σ={int(sigma_core)} px)"
    return I1, I2, u_f, v_f, omega_true, label


def make_solid_rotation(seed=42, alpha_deg=5.0, sigma_s=SIGMA_S):

    I1_clean = generate_speckle_field(FS, PPP, DTAU, 1.0, seed)
    rng = np.random.default_rng(seed + 30000)
    I1 = np.clip(I1_clean + rng.normal(0, sigma_s, I1_clean.shape), 0, 1)

    from scipy.ndimage import rotate as sp_rotate
    I2_clean = sp_rotate(I1_clean, angle=-alpha_deg, reshape=False,
                         order=3, mode='constant', cval=0.0)
    I2 = np.clip(I2_clean + rng.normal(0, sigma_s, I2_clean.shape), 0, 1)


    omega_true = np.full((FS, FS), alpha_deg)
    u_f = np.zeros((FS, FS))
    v_f = np.zeros((FS, FS))
    label = f"Solid Rotation (α={alpha_deg:.1f}°)"
    return I1, I2, u_f, v_f, omega_true, label


def make_shear(seed=42, U_max=4.0, sigma_s=SIGMA_S):

    I1_clean = generate_speckle_field(FS, PPP, DTAU, 1.0, seed)
    rng = np.random.default_rng(seed + 40000)
    I1 = np.clip(I1_clean + rng.normal(0, sigma_s, I1_clean.shape), 0, 1)

    Y, _ = np.mgrid[0:FS, 0:FS]
    u_f = U_max * Y.astype(float) / (FS - 1)
    v_f = np.zeros_like(u_f)
    I2_clean = _warp_by_flow(I1_clean, u_f, v_f)
    I2 = np.clip(I2_clean + rng.normal(0, sigma_s, I2_clean.shape), 0, 1)

    omega_true = np.full((FS, FS), np.degrees(U_max / (2.0 * FS)))
    label = f"Shear (U_max={U_max:.1f} px)"
    return I1, I2, u_f, v_f, omega_true, label


def _run_widim(I1, I2, n_iter=3, subset_size=WS, step=STEP):
    H, W_img = I1.shape
    half     = subset_size // 2
    centers  = list(range(half, H - half + 1, step))
    n        = len(centers)

    u_wdm = np.zeros((n, n))
    v_wdm = np.zeros((n, n))


    for ri, cr in enumerate(centers):
        for ci, cc in enumerate(centers):
            W1  = I1[cr - half: cr - half + subset_size,
                     cc - half: cc - half + subset_size]
            raw = _search_ncc_window(W1, I2, cr, cc, half, H, W_img,
                                     SEARCH_RADIUS, SEARCH_STEP)
            u_wdm[ri, ci] = raw["dx"]
            v_wdm[ri, ci] = raw["dy"]


    for _ in range(n_iter):
        du_dx, du_dy, dv_dx, dv_dy = _compute_gradients(u_wdm, v_wdm, step)
        om = _normalized_median_test(u_wdm, v_wdm)
        if om.any():
            u_wdm, v_wdm = _replace_outliers(u_wdm, v_wdm, om)
            du_dx, du_dy, dv_dx, dv_dy = _compute_gradients(u_wdm, v_wdm, step)
        du_res = np.zeros((n, n));  dv_res = np.zeros((n, n))
        for ri, cr in enumerate(centers):
            for ci, cc in enumerate(centers):
                W1     = I1[cr - half: cr - half + subset_size,
                            cc - half: cc - half + subset_size]
                W2_def = _deform_window(I2, cr, cc, half, H, W_img,
                                        u_wdm[ri, ci], v_wdm[ri, ci],
                                        du_dx[ri, ci], du_dy[ri, ci],
                                        dv_dx[ri, ci], dv_dy[ri, ci])
                dx_r, dy_r = _residual_ncc(W1, W2_def, 3)
                du_res[ri, ci] = dx_r;  dv_res[ri, ci] = dy_r
        u_wdm += du_res;  v_wdm += dv_res

    return centers, n, u_wdm, v_wdm


def _eval_one_seed(flow_fn, seed):

    I1, I2, u_f, v_f, omega_field, _ = flow_fn(seed=seed)
    H = I1.shape[0];  half = WS // 2


    centers, n, u_wdm, v_wdm = _run_widim(I1, I2)

    du_dx, du_dy, dv_dx, dv_dy = _compute_gradients(u_wdm, v_wdm, STEP)
    omega_widim_map = np.degrees(0.5 * (dv_dx - du_dy))


    omega_nufft_list   = []
    omega_widim_list   = []
    omega_true_list    = []

    for ri, cr in enumerate(centers):
        for ci, cc in enumerate(centers):
            if cr - half < 0 or cr + half > H or cc - half < 0 or cc + half > H:
                continue
            W1 = I1[cr - half: cr - half + WS, cc - half: cc - half + WS]
            W2 = I2[cr - half: cr - half + WS, cc - half: cc - half + WS]

            with contextlib.redirect_stdout(_io.StringIO()):
                res = estimate_rotation(W1, W2, OPT_R_MIN, OPT_WEIGHT,
                                        OPT_POC, verbose=False)

            snr = res["snr"]
            ang = res["angle_est"]


            true_val = omega_field[cr, cc]

            if True:


                omega_nufft_list.append(-ang)
                omega_widim_list.append(omega_widim_map[ri, ci])
                omega_true_list.append(true_val)

    if len(omega_true_list) == 0:
        return None

    return dict(
        nufft = np.array(omega_nufft_list),
        widim = np.array(omega_widim_list),
        true  = np.array(omega_true_list),
    )


def _bootstrap_ci(errors, n_boot=N_BOOT, ci=95.0):

    rng = np.random.default_rng(0)
    rmse_vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(errors), len(errors))
        rmse_vals.append(np.sqrt(np.mean(errors[idx]**2)))
    lo = np.percentile(rmse_vals, (100 - ci) / 2)
    hi = np.percentile(rmse_vals, 100 - (100 - ci) / 2)
    return lo, hi


def run_mc(flow_fn, n_seeds=N_SEEDS, label=""):

    print(f"  [{label}] running {n_seeds} seeds ...", flush=True)
    all_nufft = [];  all_widim = []

    for s in range(n_seeds):
        result = _eval_one_seed(flow_fn, seed=s)
        if result is None:
            continue
        all_nufft.append(result["nufft"] - result["true"])
        all_widim.append(result["widim"] - result["true"])
        if (s + 1) % 10 == 0:
            print(f"    seed {s+1}/{n_seeds}", flush=True)

    if len(all_nufft) == 0:
        return None

    err_nufft = np.concatenate(all_nufft)
    err_widim = np.concatenate(all_widim)

    rmse_n = np.sqrt(np.mean(err_nufft**2))
    rmse_w = np.sqrt(np.mean(err_widim**2))
    ci_n   = _bootstrap_ci(err_nufft)
    ci_w   = _bootstrap_ci(err_widim)
    N      = len(err_nufft)

    print(f"    N={N}  NUFFT RMSE={rmse_n:.3f}° [{ci_n[0]:.3f},{ci_n[1]:.3f}]  "
          f"WIDIM RMSE={rmse_w:.3f}° [{ci_w[0]:.3f},{ci_w[1]:.3f}]  "
          f"改善={rmse_w/max(rmse_n,1e-6):.2f}×", flush=True)

    return dict(label=label,
                rmse_nufft=rmse_n, ci_nufft=ci_n,
                rmse_widim=rmse_w, ci_widim=ci_w,
                N=N,
                improve=rmse_w / max(rmse_n, 1e-6))


def _viz_one_scene(ax_true, ax_nufft, ax_widim, ax_err_n, ax_err_w,
                   flow_fn, seed=42, label="", step_viz=STEP):

    I1, I2, u_f, v_f, omega_field, _ = flow_fn(seed=seed)
    H = I1.shape[0];  half = WS // 2

    centers, n, u_wdm, v_wdm = _run_widim(I1, I2, n_iter=1, step=step_viz)
    du_dx, du_dy, dv_dx, dv_dy = _compute_gradients(u_wdm, v_wdm, step_viz)
    omega_widim_map = np.degrees(0.5 * (dv_dx - du_dy))

    omega_nufft_map = np.full((n, n), np.nan)
    omega_true_map  = np.full((n, n), np.nan)

    for ri, cr in enumerate(centers):
        for ci, cc in enumerate(centers):
            if cr - half < 0 or cr + half > H or cc - half < 0 or cc + half > H:
                continue
            W1 = I1[cr - half: cr - half + WS, cc - half: cc - half + WS]
            W2 = I2[cr - half: cr - half + WS, cc - half: cc - half + WS]
            with contextlib.redirect_stdout(_io.StringIO()):
                res = estimate_rotation(W1, W2, OPT_R_MIN, OPT_WEIGHT,
                                        OPT_POC, verbose=False)
            snr = res["snr"];  ang = res["angle_est"]
            if True:
                omega_nufft_map[ri, ci] = -ang
            omega_true_map[ri, ci] = omega_field[cr, cc]

    vmax = np.nanmax(np.abs(omega_true_map)) * 1.2 + 0.1
    cmap_v = LinearSegmentedColormap.from_list(
        "vort_cmap", ["#FC757B", "#F97F5F", "#FAA26F", "#FDCD94",
                      "#FEE199", "#B0D6A9", "#65BDBA", "#3C9BC9"]
    )
    kw = dict(cmap=cmap_v, vmin=-vmax, vmax=vmax, aspect='equal')

    im_v = ax_true.imshow(omega_true_map,  **kw);  ax_true.set_title("Ground Truth Ω",  fontsize=8)
    ax_nufft.imshow(omega_nufft_map, **kw);  ax_nufft.set_title("NUFFT Direct Ω",  fontsize=8)
    ax_widim.imshow(omega_widim_map, **kw);  ax_widim.set_title("WIDIM Gradient Ω", fontsize=8)

    err_nufft = omega_nufft_map - omega_true_map
    err_widim = omega_widim_map - omega_true_map
    emax = max(np.nanmax(np.abs(err_nufft)), np.nanmax(np.abs(err_widim))) * 1.1 + 0.1
    cmap_e = LinearSegmentedColormap.from_list(
        "err_cmap", ["#B6B3D6", "#CFCCE3", "#D5D3DE", "#D5D1D1",
                     "#F6DFD6", "#F8B2A2", "#F1837A", "#E9687A"]
    )
    ekw  = dict(cmap=cmap_e, vmin=0.0, vmax=emax, aspect='equal')
    err_abs_n = np.abs(err_nufft)
    err_abs_w = np.abs(err_widim)
    im_en = ax_err_n.imshow(err_abs_n, **ekw)
    im_ew = ax_err_w.imshow(err_abs_w, **ekw)

    rmse_n = np.sqrt(np.nanmean(err_nufft**2))
    rmse_w = np.sqrt(np.nanmean(err_widim**2))
    ax_err_n.set_title(f"NUFFT Abs Error\n(RMSE={rmse_n:.2f}°)", fontsize=7)
    ax_err_w.set_title(f"WIDIM Abs Error\n(RMSE={rmse_w:.2f}°)", fontsize=7)
    for ax in (ax_true, ax_nufft, ax_widim, ax_err_n, ax_err_w):
        ax.set_xlabel(label, fontsize=7);  ax.set_xticks([]);  ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box')
    return im_v, im_en, im_ew


def plot_main(results, viz_fns, seed_viz=42, step_viz=10):

    n_flows = len(viz_fns)
    fig, axes = plt.subplots(n_flows, 5, figsize=(15.5, 3.0 * n_flows),
                              dpi=150, constrained_layout=False)
    fig.subplots_adjust(right=0.88, wspace=0.08, hspace=0.22)
    fig.suptitle("Ground-Truth Vorticity Comparison: NUFFT vs WIDIM Gradient\n"
                 f"(σ_s={SIGMA_S}, WS={WS} px, MC N={N_SEEDS} seeds)",
                 fontsize=10, fontweight='bold')

    for row, (fn, label) in enumerate(viz_fns):
        axs = axes[row] if n_flows > 1 else axes
        im_v, im_en, im_ew = _viz_one_scene(axs[0], axs[1], axs[2], axs[3], axs[4],
                                            fn, seed=seed_viz, label=label, step_viz=step_viz)
        axs[0].set_ylabel(label, fontsize=8)
        pos_l = axs[0].get_position()
        pos_r = axs[4].get_position()
        height = pos_l.y1 - pos_l.y0
        x_cb = pos_r.x1 + 0.012
        cb_w = 0.012
        cb_h = height * 0.46
        cb1 = fig.add_axes([x_cb, pos_l.y0 + height * 0.52, cb_w, cb_h])
        cb2 = fig.add_axes([x_cb, pos_l.y0 + height * 0.02, cb_w, cb_h])
        cbar1 = fig.colorbar(im_v, cax=cb1)
        cbar1.ax.tick_params(labelsize=6)
        cbar1.set_label("Vorticity (deg/frame)", fontsize=6)
        cbar2 = fig.colorbar(im_en, cax=cb2)
        cbar2.ax.tick_params(labelsize=6)
        cbar2.set_label("Abs Error (deg/frame)", fontsize=6)

    out = os.path.join(OUT_DIR, "gt_vorticity_main.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  图已保存: {out}")
    return out


def plot_summary(results_list):

    valid = [r for r in results_list if r is not None]
    if not valid:
        print("  无有效结果，跳过汇总图")
        return

    labels = [r["label"] for r in valid]
    rmse_n = [r["rmse_nufft"] for r in valid]
    rmse_w = [r["rmse_widim"] for r in valid]
    ci_n   = [(r["ci_nufft"][1] - r["rmse_nufft"],
               r["rmse_nufft"] - r["ci_nufft"][0]) for r in valid]
    ci_w   = [(r["ci_widim"][1] - r["rmse_widim"],
               r["rmse_widim"] - r["ci_widim"][0]) for r in valid]

    x = np.arange(len(labels));  w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4), dpi=150)

    bars_n = ax.bar(x - w/2, rmse_n, w, label='NUFFT Direct',
                    color='#FC757B', alpha=0.85,
                    yerr=np.array(ci_n).T, capsize=4, error_kw={'elinewidth':1.2})
    bars_w = ax.bar(x + w/2, rmse_w, w, label='WIDIM Gradient',
                    color='#FEE199', alpha=0.85,
                    yerr=np.array(ci_w).T, capsize=4, error_kw={'elinewidth':1.2})

    for i, r in enumerate(valid):
        ymax = max(rmse_n[i], rmse_w[i])
        ax.text(i, ymax * 1.08,
                f'{r["improve"]:.1f}×', ha='center', va='bottom',
                fontsize=8.5, color='#154360', fontweight='bold')

    ax.set_xlabel("Flow Type", fontsize=10)
    ax.set_ylabel("Vorticity RMSE (°/frame)", fontsize=10)
    ax.set_title("Ground-Truth Vorticity RMSE: NUFFT vs WIDIM Gradient\n"
                 f"(95% Bootstrap CI, N={N_SEEDS} seeds each, σ_s={SIGMA_S})",
                 fontsize=10)
    ax.set_xticks(x);  ax.set_xticklabels(labels, fontsize=8)
    ax.legend(fontsize=9);  ax.grid(axis='y', alpha=0.4)
    ax.set_ylim(0, max(rmse_w) * 1.35)

    out = os.path.join(OUT_DIR, "gt_summary_table.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  图已保存: {out}")
    return out


def plot_summary_from_txt(txt_path):
    lines = Path(txt_path).read_text(encoding="utf-8").splitlines()
    results = []
    label = None
    rmse_n = rmse_w = None
    ci_n = ci_w = None
    improve = None
    for line in lines:
        if line.startswith("Flow: "):
            if label and rmse_n is not None:
                results.append(dict(
                    label=label,
                    rmse_nufft=rmse_n,
                    ci_nufft=ci_n,
                    rmse_widim=rmse_w,
                    ci_widim=ci_w,
                    improve=improve,
                ))
            label = line.split("Flow: ")[1].strip()
            rmse_n = rmse_w = None
            ci_n = ci_w = None
            improve = None
        elif line.strip().startswith("NUFFT RMSE:"):
            s = line.split("NUFFT RMSE: ")[1]
            val = s.split("°")[0].strip()
            rng = s.split("[")[1].split("]")[0].split(",")
            rmse_n = float(val)
            ci_n = (float(rng[0]), float(rng[1]))
        elif line.strip().startswith("WIDIM RMSE:"):
            s = line.split("WIDIM RMSE: ")[1]
            val = s.split("°")[0].strip()
            rng = s.split("[")[1].split("]")[0].split(",")
            rmse_w = float(val)
            ci_w = (float(rng[0]), float(rng[1]))
        elif line.strip().startswith("Improvement:"):
            improve = float(line.split(":")[1].replace("x", "").strip())
    if label and rmse_n is not None:
        results.append(dict(
            label=label,
            rmse_nufft=rmse_n,
            ci_nufft=ci_n,
            rmse_widim=rmse_w,
            ci_widim=ci_w,
            improve=improve,
        ))
    return plot_summary(results)


def _set_cn_font():
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                            "Arial Unicode MS", "DejaVu Sans"],
        "font.family": "sans-serif",
        "axes.unicode_minus": False,
    })


def generate_fig1_motivation(seed=42, out_name="fig01_motivation.png", step_fig=10):
    set_plot_style()

    I1, I2, _, _, omega_true, _ = make_rankine(seed=seed)
    H = I1.shape[0]
    half = WS // 2

    centers, n, u_wdm, v_wdm = _run_widim(I1, I2, n_iter=1, step=step_fig)
    du_dx, du_dy, dv_dx, dv_dy = _compute_gradients(u_wdm, v_wdm, step_fig)
    omega_widim_map = np.degrees(0.5 * (dv_dx - du_dy))

    omega_nufft_map = np.full((n, n), np.nan)
    omega_true_map = np.full((n, n), np.nan)

    for ri, cr in enumerate(centers):
        for ci, cc in enumerate(centers):
            if cr - half < 0 or cr + half > H or cc - half < 0 or cc + half > H:
                continue
            W1 = I1[cr - half: cr - half + WS, cc - half: cc - half + WS]
            W2 = I2[cr - half: cr - half + WS, cc - half: cc - half + WS]
            with contextlib.redirect_stdout(_io.StringIO()):
                res = estimate_rotation(W1, W2, OPT_R_MIN, OPT_WEIGHT,
                                        OPT_POC, verbose=False)
            snr = res["snr"]
            ang = res["angle_est"]
            if True:
                omega_nufft_map[ri, ci] = -ang
            omega_true_map[ri, ci] = omega_true[cr, cc]

    vmax = np.nanmax(np.abs(omega_true_map)) * 1.2 + 0.1
    diff = np.where(np.isfinite(omega_nufft_map),
                    omega_widim_map - omega_nufft_map, np.nan)
    err_abs = np.abs(diff)
    emax = np.nanmax(err_abs) * 1.1 + 0.1

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), dpi=150,
                             constrained_layout=True)
    cmap_v = LinearSegmentedColormap.from_list(
        "vort_cmap", ["#FC757B", "#F97F5F", "#FAA26F", "#FDCD94",
                      "#FEE199", "#B0D6A9", "#65BDBA", "#3C9BC9"]
    )
    cmap_e = LinearSegmentedColormap.from_list(
        "err_cmap", ["#B6B3D6", "#CFCCE3", "#D5D3DE", "#D5D1D1",
                     "#F6DFD6", "#F8B2A2", "#F1837A", "#E9687A"]
    )
    im0 = axes[0].imshow(omega_widim_map, cmap=cmap_v,
                         vmin=-vmax, vmax=vmax, aspect="auto")
    axes[0].set_title("Finite-Difference Vorticity (WIDIM)", fontsize=10)

    im1 = axes[1].imshow(omega_nufft_map, cmap=cmap_v,
                         vmin=-vmax, vmax=vmax, aspect="auto")
    axes[1].set_title("WIDIM+NUFFT Direct Vorticity", fontsize=10)

    im2 = axes[2].imshow(err_abs, cmap=cmap_e,
                         vmin=0.0, vmax=emax, aspect="auto")
    axes[2].set_title("Rankine Error Map (WIDIM−NUFFT)", fontsize=10)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(im0, ax=axes[:2], shrink=0.85, label="Vorticity (deg/frame)")
    fig.colorbar(im2, ax=axes[2], shrink=0.85, label="Error (deg/frame)")

    out = os.path.join(OUT_DIR, out_name)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    if os.path.isdir(IEEE_DIR):
        dst = os.path.join(IEEE_DIR, out_name)
        shutil.copy2(out, dst)
        print(f"  已复制到 IEEE: {dst}")
    print(f"  图已保存: {out}")
    return out


def save_results_txt(results_list, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\n")
        f.write("Ground-Truth Vorticity Comparison Results (NO GATE)\n")
        f.write(f"sigma_s={SIGMA_S}, WS={WS}px, MC N_SEEDS={N_SEEDS}, "
                f"Bootstrap N_BOOT={N_BOOT}\n")
        f.write("=" * 72 + "\n\n")
        for r in results_list:
            if r is None:
                continue
            f.write(f"Flow: {r['label']}\n")
            f.write(f"  N (all windows, no gate): {r['N']}\n")
            f.write(f"  NUFFT RMSE: {r['rmse_nufft']:.3f}° "
                    f"[{r['ci_nufft'][0]:.3f}, {r['ci_nufft'][1]:.3f}] 95%CI\n")
            f.write(f"  WIDIM RMSE: {r['rmse_widim']:.3f}° "
                    f"[{r['ci_widim'][0]:.3f}, {r['ci_widim'][1]:.3f}] 95%CI\n")
            f.write(f"  Improvement: {r['improve']:.2f}x\n\n")
        f.write("\n% LaTeX table row format:\n")
        f.write("% Flow & NUFFT RMSE (95\\%CI) & WIDIM RMSE (95\\%CI) & Improvement \\\\\n")
        for r in results_list:
            if r is None:
                continue
            f.write(
                f"% {r['label']} & "
                f"{r['rmse_nufft']:.3f} [{r['ci_nufft'][0]:.3f},{r['ci_nufft'][1]:.3f}] & "
                f"{r['rmse_widim']:.3f} [{r['ci_widim'][0]:.3f},{r['ci_widim'][1]:.3f}] & "
                f"{r['improve']:.2f}x \\\\\n"
            )
    print(f"  结果已保存: {out_path}")


def main():
    t0 = time.perf_counter()
    print("=" * 60)
    print("Ground-Truth Vorticity Comparison (NO GATE)")
    print(f"sigma_s={SIGMA_S}, WS={WS}px, N_SEEDS={N_SEEDS}")
    print("=" * 60)

    print("\n[Step 0] Generate Fig.1 motivation ...")
    generate_fig1_motivation(seed=42, step_fig=10)


    flow_specs = [
        (make_rankine,      "Rankine Vortex"),
        (make_lamb_oseen,   "Lamb-Oseen Vortex"),
        (make_solid_rotation, "Solid Rotation"),
        (make_shear,        "Linear Shear"),
    ]

    print("\n[Step 1] Monte Carlo 聚合评估 ...")
    results_list = []
    for fn, label in flow_specs:
        r = run_mc(fn, n_seeds=N_SEEDS, label=label)
        results_list.append(r)

    print("\n[Step 2] 绘制代表性可视化（seed=42）...")
    viz_fns = [(fn, label) for fn, label in flow_specs]
    plot_main(results_list, viz_fns, seed_viz=42)

    print("\n[Step 3] 绘制 RMSE 汇总图 ...")
    plot_summary(results_list)

    print("\n[Step 4] 保存数字结果 ...")
    txt_path = os.path.join(OUT_DIR, "gt_vorticity_results.txt")
    save_results_txt(results_list, txt_path)

    dt = time.perf_counter() - t0
    print(f"\n完成。总耗时: {dt:.1f}s")
    print(f"输出目录: {OUT_DIR}")

    return results_list


if __name__ == "__main__":
    main()
