import os, sys, contextlib, io as _io, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import map_coordinates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from poc_common import (
    generate_speckle_field, apply_affine, extract_window, DEFAULT_PARAMS,
    set_plot_style, COLORS,
)
from poc_point3 import estimate_rotation, OPT_R_MIN, OPT_WEIGHT, OPT_POC

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_comparison")
os.makedirs(OUT_DIR, exist_ok=True)

FS            = DEFAULT_PARAMS["field_size"]
WS            = DEFAULT_PARAMS["window_size"]
PPP           = DEFAULT_PARAMS["ppp"]
DTAU          = DEFAULT_PARAMS["d_tau"]
STEP          = 10
SEARCH_RADIUS = 6
SEARCH_STEP   = 1
N_SEEDS       = 3


def _warp_by_flow(I1, u, v):


    H, W = I1.shape
    Y, X = np.mgrid[0:H, 0:W].astype(float)
    return map_coordinates(I1, [Y - v, X - u],
                           order=3, mode='constant', cval=0.0)


def make_rankine_vortex(seed=42, Gamma=1500.0, R_core=50):


    I1 = generate_speckle_field(FS, PPP, DTAU, 1.0, seed)
    Y, X = np.mgrid[0:FS, 0:FS].astype(float)
    rx = X - FS / 2.0
    ry = Y - FS / 2.0
    r  = np.hypot(rx, ry) + 1e-10

    Omega = Gamma / (2.0 * np.pi * R_core ** 2)
    v_t   = np.where(r <= R_core,
                     Omega * r,
                     Gamma / (2.0 * np.pi * r))

    u_field = -v_t * ry / r
    v_field =  v_t * rx / r

    I2 = _warp_by_flow(I1, u_field, v_field)

    omega_deg = np.where(r <= R_core, np.degrees(Omega), 0.0)

    return I1, I2, u_field, v_field, omega_deg, f"RankineVortex_G{int(Gamma)}_Rc{R_core}"


def make_shear_flow(seed=42, U_max=4.0):

    I1 = generate_speckle_field(FS, PPP, DTAU, 1.0, seed)
    Y, _ = np.mgrid[0:FS, 0:FS]
    u_field = U_max * Y.astype(float) / (FS - 1)
    v_field = np.zeros_like(u_field)
    I2 = _warp_by_flow(I1, u_field, v_field)
    omega_deg = np.full((FS, FS), np.degrees(U_max / FS))
    return I1, I2, u_field, v_field, omega_deg, f"ShearFlow_Umax{int(U_max)}"


def make_uniform_flow(seed=42, u0=4.0, v0=3.0):

    I1 = generate_speckle_field(FS, PPP, DTAU, 1.0, seed)
    u_field = np.full((FS, FS), u0)
    v_field = np.full((FS, FS), v0)
    I2 = _warp_by_flow(I1, u_field, v_field)
    omega_deg = np.zeros((FS, FS))
    return I1, I2, u_field, v_field, omega_deg, f"UniformFlow_u{int(u0)}_v{int(v0)}"


def _ncc(W1, W2):

    w1 = W1 - W1.mean()
    w2 = W2 - W2.mean()
    d = np.sqrt(np.sum(w1 ** 2) * np.sum(w2 ** 2))
    return float(np.sum(w1 * w2) / d) if d > 1e-30 else 0.0


def _search_ncc_window(W1, I2, cr, cc, half, H, W, radius, step):

    best_ncc = -1.0
    best_dx, best_dy = 0.0, 0.0
    for dy in range(-radius, radius + 1, step):
        for dx in range(-radius, radius + 1, step):
            r2, c2 = cr + dy, cc + dx
            if r2 - half < 0 or r2 + half > H or c2 - half < 0 or c2 + half > W:
                continue
            W2s = I2[r2 - half: r2 - half + WS, c2 - half: c2 - half + WS]
            ncc = _ncc(W1, W2s)
            if ncc > best_ncc:
                best_ncc = ncc
                best_dx, best_dy = float(dx), float(dy)
    return dict(dx=best_dx, dy=best_dy, ncc=best_ncc)


def _search_nufft_ncc_window(W1, I2, cr, cc, half, H, W, radius, step):


    W2_c = I2[cr - half: cr - half + WS, cc - half: cc - half + WS]

    with contextlib.redirect_stdout(_io.StringIO()):
        res = estimate_rotation(W1, W2_c, OPT_R_MIN, OPT_WEIGHT,
                                OPT_POC, verbose=False)
    angle_est = res["angle_est"]
    snr       = res["snr"]

    best_ncc = -1.0
    best_dx, best_dy = 0.0, 0.0
    for dy in range(-radius, radius + 1, step):
        for dx in range(-radius, radius + 1, step):
            r2, c2 = cr + dy, cc + dx
            if r2 - half < 0 or r2 + half > H or c2 - half < 0 or c2 + half > W:
                continue
            W2s   = I2[r2 - half: r2 - half + WS, c2 - half: c2 - half + WS]
            W2_cr = apply_affine(W2s, dx=0, dy=0, angle_deg=-angle_est)
            ncc   = _ncc(W1, W2_cr)
            if ncc > best_ncc:
                best_ncc = ncc
                best_dx, best_dy = float(dx), float(dy)

    return dict(dx=best_dx, dy=best_dy, ncc=best_ncc,
                angle=angle_est, snr=snr)


def _compute_gradients(u_map, v_map, step):


    du_dx = np.zeros_like(u_map)
    du_dy = np.zeros_like(u_map)
    dv_dx = np.zeros_like(v_map)
    dv_dy = np.zeros_like(v_map)


    du_dx[:, 1:-1] = (u_map[:, 2:] - u_map[:, :-2]) / (2.0 * step)
    du_dy[1:-1, :] = (u_map[2:, :] - u_map[:-2, :]) / (2.0 * step)
    dv_dx[:, 1:-1] = (v_map[:, 2:] - v_map[:, :-2]) / (2.0 * step)
    dv_dy[1:-1, :] = (v_map[2:, :] - v_map[:-2, :]) / (2.0 * step)


    du_dx[:, 0]  = (u_map[:, 1]  - u_map[:, 0])  / step
    du_dx[:, -1] = (u_map[:, -1] - u_map[:, -2]) / step
    du_dy[0, :]  = (u_map[1, :]  - u_map[0, :])  / step
    du_dy[-1, :] = (u_map[-1, :] - u_map[-2, :]) / step
    dv_dx[:, 0]  = (v_map[:, 1]  - v_map[:, 0])  / step
    dv_dx[:, -1] = (v_map[:, -1] - v_map[:, -2]) / step
    dv_dy[0, :]  = (v_map[1, :]  - v_map[0, :])  / step
    dv_dy[-1, :] = (v_map[-1, :] - v_map[-2, :]) / step

    return du_dx, du_dy, dv_dx, dv_dy


def _normalized_median_test(u_map, v_map, threshold=2.0, eps=0.1):


    n = u_map.shape[0]
    outlier_mask = np.zeros((n, n), dtype=bool)

    for i in range(n):
        for j in range(n):
            i0, i1 = max(i - 1, 0), min(i + 2, n)
            j0, j1 = max(j - 1, 0), min(j + 2, n)

            nb_u = u_map[i0:i1, j0:j1].ravel()
            nb_v = v_map[i0:i1, j0:j1].ravel()
            flat_idx = (i - i0) * (j1 - j0) + (j - j0)
            nb_u = np.delete(nb_u, flat_idx)
            nb_v = np.delete(nb_v, flat_idx)

            if len(nb_u) == 0:
                continue

            med_u = np.median(nb_u)
            med_v = np.median(nb_v)
            r_u = abs(u_map[i, j] - med_u) / (np.median(np.abs(nb_u - med_u)) + eps)
            r_v = abs(v_map[i, j] - med_v) / (np.median(np.abs(nb_v - med_v)) + eps)

            if r_u > threshold or r_v > threshold:
                outlier_mask[i, j] = True

    return outlier_mask


def _replace_outliers(u_map, v_map, outlier_mask):


    n = u_map.shape[0]
    u_new = u_map.copy()
    v_new = v_map.copy()

    for i in range(n):
        for j in range(n):
            if not outlier_mask[i, j]:
                continue
            i0, i1 = max(i - 1, 0), min(i + 2, n)
            j0, j1 = max(j - 1, 0), min(j + 2, n)
            valid = ~outlier_mask[i0:i1, j0:j1]
            nb_u  = u_map[i0:i1, j0:j1][valid]
            nb_v  = v_map[i0:i1, j0:j1][valid]
            if len(nb_u) > 0:
                u_new[i, j] = np.median(nb_u)
                v_new[i, j] = np.median(nb_v)

    return u_new, v_new


def _deform_window(I2, cr, cc, half, H, W_img, u, v, du_dx, du_dy, dv_dx, dv_dy):


    yr = np.arange(-half, half)
    xr = np.arange(-half, half)
    YR, XR = np.meshgrid(yr, xr, indexing='ij')

    Y2 = (cr + YR) + v + dv_dy * YR + dv_dx * XR
    X2 = (cc + XR) + u + du_dx * XR + du_dy * YR

    Y2 = np.clip(Y2, 0, H - 1)
    X2 = np.clip(X2, 0, W_img - 1)

    return map_coordinates(I2, [Y2, X2], order=3, mode='nearest')


def _gauss_peak_1d(arr, pi):


    n = len(arr)
    if pi <= 0 or pi >= n - 1:
        return float(pi)
    l, c, r = arr[pi - 1], arr[pi], arr[pi + 1]
    if c <= 0 or l <= 0 or r <= 0:
        return float(pi)
    try:
        denom = np.log(l) - 2.0 * np.log(c) + np.log(r)
        if abs(denom) < 1e-12:
            return float(pi)
        offset = 0.5 * (np.log(l) - np.log(r)) / denom
    except Exception:
        offset = 0.0
    return float(pi) + offset


def _residual_ncc(W1, W2_def, res_radius=3):


    ws  = W1.shape[0]
    sz  = 2 * res_radius + 1
    ncc_grid = np.zeros((sz, sz))
    best_ncc     = -np.inf
    best_dy_int  = 0
    best_dx_int  = 0

    for idy, dy in enumerate(range(-res_radius, res_radius + 1)):
        for idx, dx in enumerate(range(-res_radius, res_radius + 1)):
            r0  = max(0,  dy);  r1  = min(ws, ws + dy)
            c0  = max(0,  dx);  c1  = min(ws, ws + dx)
            r0s = max(0, -dy);  r1s = r0s + (r1 - r0)
            c0s = max(0, -dx);  c1s = c0s + (c1 - c0)

            if r1 <= r0 or c1 <= c0:
                continue

            v = _ncc(W1[r0s:r1s, c0s:c1s], W2_def[r0:r1, c0:c1])
            ncc_grid[idy, idx] = v

            if v > best_ncc:
                best_ncc    = v
                best_dy_int = dy
                best_dx_int = dx

    idy_peak = best_dy_int + res_radius
    idx_peak = best_dx_int + res_radius

    dy_sub = _gauss_peak_1d(ncc_grid[:, idx_peak], idy_peak) - res_radius
    dx_sub = _gauss_peak_1d(ncc_grid[idy_peak, :], idx_peak) - res_radius

    return float(dx_sub), float(dy_sub)


def widim_fullfield(I1, I2, subset_size=WS, step=STEP,
                    search_radius=SEARCH_RADIUS, search_step=SEARCH_STEP,
                    n_iter=3, res_radius=3, nmt_threshold=2.0, nmt_eps=0.1):


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
                                     search_radius, search_step)
            u_wdm[ri, ci] = raw["dx"]
            v_wdm[ri, ci] = raw["dy"]


    for _it in range(1, n_iter + 1):

        du_dx, du_dy, dv_dx, dv_dy = _compute_gradients(u_wdm, v_wdm, step)


        outlier_mask = _normalized_median_test(u_wdm, v_wdm,
                                               threshold=nmt_threshold,
                                               eps=nmt_eps)
        if outlier_mask.any():
            u_wdm, v_wdm = _replace_outliers(u_wdm, v_wdm, outlier_mask)
            du_dx, du_dy, dv_dx, dv_dy = _compute_gradients(u_wdm, v_wdm, step)


        du_res = np.zeros((n, n))
        dv_res = np.zeros((n, n))

        for ri, cr in enumerate(centers):
            for ci, cc in enumerate(centers):
                W1     = I1[cr - half: cr - half + subset_size,
                            cc - half: cc - half + subset_size]
                W2_def = _deform_window(
                    I2, cr, cc, half, H, W_img,
                    u_wdm[ri, ci], v_wdm[ri, ci],
                    du_dx[ri, ci], du_dy[ri, ci],
                    dv_dx[ri, ci], dv_dy[ri, ci])
                dx_r, dy_r = _residual_ncc(W1, W2_def, res_radius)
                du_res[ri, ci] = dx_r
                dv_res[ri, ci] = dy_r


        u_wdm += du_res
        v_wdm += dv_res

    return dict(centers=centers, n=n, u_wdm=u_wdm, v_wdm=v_wdm)


def fullfield_compare(I1, I2, step=STEP,
                      search_radius=SEARCH_RADIUS, search_step=SEARCH_STEP):

    H, W   = I1.shape
    half   = WS // 2
    centers = list(range(half, H - half + 1, step))
    n      = len(centers)

    u_ncc  = np.full((n, n), np.nan);  v_ncc  = np.full((n, n), np.nan)
    u_nft  = np.full((n, n), np.nan);  v_nft  = np.full((n, n), np.nan)
    angle_map = np.full((n, n), np.nan)
    snr_map   = np.full((n, n), np.nan)

    for ri, cr in enumerate(centers):
        for ci, cc in enumerate(centers):
            W1 = I1[cr - half: cr - half + WS, cc - half: cc - half + WS]

            raw = _search_ncc_window(
                W1, I2, cr, cc, half, H, W, search_radius, search_step)
            nft = _search_nufft_ncc_window(
                W1, I2, cr, cc, half, H, W, search_radius, search_step)

            u_ncc[ri, ci] = raw["dx"];  v_ncc[ri, ci] = raw["dy"]
            u_nft[ri, ci] = nft["dx"];  v_nft[ri, ci] = nft["dy"]
            angle_map[ri, ci] = nft["angle"]
            snr_map[ri, ci]   = nft["snr"]

    return dict(centers=centers, n=n,
                u_ncc=u_ncc, v_ncc=v_ncc,
                u_nft=u_nft, v_nft=v_nft,
                angle=angle_map, snr=snr_map)


def run_scenario(name, I1, I2, u_true, v_true, omega_deg):
    print(f"\n  [{name}]")
    t0 = time.perf_counter()


    print(f"    运行 WIDIM (n_iter=3) ...")
    wdm = widim_fullfield(I1, I2, n_iter=3)
    wdn = dict(centers=wdm["centers"], n=wdm["n"],
               u_wdm=wdm["u_wdm"].copy(), v_wdm=wdm["v_wdm"].copy())

    centers = wdm["centers"]


    u_gt = np.array([[u_true[cr, cc] for cc in centers] for cr in centers])
    v_gt = np.array([[v_true[cr, cc] for cc in centers] for cr in centers])
    omega_gt = np.array([[omega_deg[cr, cc] for cc in centers] for cr in centers])


    err_wdm = np.sqrt((wdm["u_wdm"] - u_gt) ** 2 + (wdm["v_wdm"] - v_gt) ** 2)
    err_wdn = np.sqrt((wdn["u_wdm"] - u_gt) ** 2 + (wdn["v_wdm"] - v_gt) ** 2)

    rmse_wdm = float(np.nanmean(err_wdm))
    rmse_wdn = float(np.nanmean(err_wdn))
    diff_wdn = (rmse_wdn - rmse_wdm) / rmse_wdm * 100.0 if rmse_wdm > 1e-6 else 0.0

    elapsed = time.perf_counter() - t0

    print(f"    WIDIM        RMSE = {rmse_wdm:.4f} px")
    print(f"    WIDIM+NUFFT  RMSE = {rmse_wdn:.4f} px  (差异 {diff_wdn:.3f}%)")
    print(f"    局部旋转率均值     = {np.nanmean(omega_gt):.2f} deg/frame")
    print(f"    用时               = {elapsed:.1f} s")

    _plot_comparison(I1, I2, wdm, wdn, u_gt, v_gt,
                     rmse_wdm, rmse_wdn, diff_wdn, name)

    return dict(name=name,
                rmse_wdm=rmse_wdm, rmse_wdn=rmse_wdn, diff_wdn=diff_wdn,
                elapsed=elapsed, omega_mean=float(np.nanmean(omega_gt)))


def _plot_comparison(I1, I2, wdm, wdn, u_gt, v_gt,
                     rmse_wdm, rmse_wdn, diff_wdn, name):
    centers = wdm["centers"]
    CC, RR  = np.meshgrid(centers, centers)

    u_max = max(np.nanmax(np.abs(u_gt)), np.nanmax(np.abs(v_gt)), 1.0)
    scale = u_max * 8.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(I2, cmap='gray', alpha=0.35, vmin=0, vmax=1)
    axes[0].quiver(CC, RR, wdm["u_wdm"], wdm["v_wdm"],
                   color=COLORS[7], scale=scale, width=0.003,
                   headwidth=4, headlength=4)
    axes[0].set_title(f'WIDIM\nRMSE = {rmse_wdm:.4f} px', fontsize=11)
    axes[0].axis('off')

    axes[1].imshow(I2, cmap='gray', alpha=0.35, vmin=0, vmax=1)
    axes[1].quiver(CC, RR, wdn["u_wdm"], wdn["v_wdm"],
                   color=COLORS[5], scale=scale, width=0.003,
                   headwidth=4, headlength=4)
    axes[1].set_title(f'WIDIM+NUFFT\nRMSE = {rmse_wdn:.4f} px', fontsize=11)
    axes[1].axis('off')

    fig.suptitle(
        f'{name}\nWIDIM+NUFFT − WIDIM = {diff_wdn:.3f}%',
        fontsize=12)
    fig.tight_layout()
    fname = os.path.join(OUT_DIR, f'{name}_disp_orthogonality.png')
    fig.savefig(fname, dpi=130)
    plt.close(fig)
    print(f"    Saved: results_comparison/{name}_disp_orthogonality.png")


def _plot_summary(all_stats):
    names     = [s["name"].split("_")[0] for s in all_stats]
    rmse_wdm  = [s["rmse_wdm"] for s in all_stats]
    rmse_wdn  = [s["rmse_wdn"] for s in all_stats]
    diff_wdn  = [s["diff_wdn"] for s in all_stats]
    omega     = [s["omega_mean"] for s in all_stats]

    x = np.arange(len(names))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    b1 = axes[0].bar(x - w/2, rmse_wdm, w, label='WIDIM',
                     color=COLORS[7], alpha=0.85, edgecolor='k')
    b2 = axes[0].bar(x + w/2, rmse_wdn, w, label='WIDIM+NUFFT',
                     color=COLORS[5], alpha=0.85, edgecolor='k')
    for bar, v in zip(list(b1) + list(b2), rmse_wdm + rmse_wdn):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.002,
                     f'{v:.4f}', ha='center', va='bottom', fontsize=8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(
        [f'{n}\n(omega≈{o:.1f} deg/fr)' for n, o in zip(names, omega)], fontsize=9)
    axes[0].set_ylabel('Displacement RMSE (px)')
    axes[0].set_title('Displacement RMSE comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    b3 = axes[1].bar(x, diff_wdn, w, color=COLORS[5], alpha=0.85, edgecolor='k')
    for bar, v in zip(b3, diff_wdn):
        y_pos = bar.get_height() + 0.01 if v >= 0 else bar.get_height() - 0.03
        axes[1].text(bar.get_x() + bar.get_width() / 2, y_pos,
                     f'{v:.3f}%', ha='center', va='bottom', fontsize=8)
    axes[1].axhline(0, color='k', lw=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    axes[1].set_ylabel('RMSE difference (%)')
    axes[1].set_title('WIDIM+NUFFT − WIDIM')
    axes[1].grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fname = os.path.join(OUT_DIR, 'comparison_summary.png')
    fig.savefig(fname, dpi=130)
    plt.close(fig)
    print(f"\n  Saved summary: results_comparison/comparison_summary.png")


def run_rotation_window_search_comparison():
    set_plot_style()
    print("\n" + "=" * 76)
    print("WIDIM vs WIDIM+NUFFT displacement accuracy (synthetic PIV, known truth)")
    print(f"  场尺寸: {FS}×{FS}  窗口: {WS}×{WS}  步长: {STEP}px")
    print(f"  搜索半径: ±{SEARCH_RADIUS}px  搜索步长: {SEARCH_STEP}px")
    print(f"  WIDIM: n_iter=3, res_radius=3, NMT_threshold=2.0")
    print("=" * 76)

    scenarios = [
        ("RankineVortex", make_rankine_vortex(seed=42, Gamma=1500.0, R_core=50)),
        ("ShearFlow",     make_shear_flow(seed=42, U_max=4.0)),
        ("UniformFlow",   make_uniform_flow(seed=42, u0=4.0, v0=3.0)),
    ]

    all_stats = []
    for scene_name, (I1, I2, u_true, v_true, omega_deg, _) in scenarios:
        stats = run_scenario(scene_name, I1, I2, u_true, v_true, omega_deg)
        all_stats.append(stats)

    _plot_summary(all_stats)

    print("\n" + "=" * 76)
    print("Summary")
    print(f"  {'Flow':<20s}  {'WIDIM':>10s}  {'WIDIM+NUFFT':>14s}  "
          f"{'diff(%)':>9s}  {'omega(deg/fr)':>12s}")
    print("  " + "-" * 76)
    for s in all_stats:
        print(f"  {s['name']:<20s}  {s['rmse_wdm']:>10.4f}  {s['rmse_wdn']:>14.4f}  "
              f"{s['diff_wdn']:>8.3f}%  {s['omega_mean']:>12.2f}")
    print("=" * 76)

    return all_stats


if __name__ == "__main__":
    run_rotation_window_search_comparison()
