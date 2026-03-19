import sys, os, time, contextlib, io as _io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from poc_common import (
    generate_speckle_field, apply_affine, extract_window,
    DEFAULT_PARAMS, set_plot_style, COLORS,
)
from poc_point3 import estimate_rotation, OPT_R_MIN, OPT_WEIGHT, OPT_POC

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_fullfield")
os.makedirs(OUT_DIR, exist_ok=True)

FS            = DEFAULT_PARAMS["field_size"]
WS            = DEFAULT_PARAMS["window_size"]
PPP           = DEFAULT_PARAMS["ppp"]
DTAU          = DEFAULT_PARAMS["d_tau"]
STEP          = WS // 2
SEARCH_RADIUS = 6
SEARCH_STEP   = 1
SNR_MIN       = 3.0


def _warp_by_flow(I1, u, v):

    H, W = I1.shape
    Y, X = np.mgrid[0:H, 0:W].astype(float)
    return map_coordinates(I1, [Y - v, X - u],
                           order=3, mode='constant', cval=0.0)


def make_rankine_vortex_flow(seed=42, Gamma=1500.0, R_core=50):


    I1 = generate_speckle_field(FS, PPP, DTAU, 1.0, seed)
    Y, X = np.mgrid[0:FS, 0:FS].astype(float)
    rx = X - FS / 2.0
    ry = Y - FS / 2.0
    r  = np.hypot(rx, ry) + 1e-10

    Omega = Gamma / (2.0 * np.pi * R_core ** 2)
    v_t   = np.where(r <= R_core, Omega * r, Gamma / (2.0 * np.pi * r))

    u_field = -v_t * ry / r
    v_field =  v_t * rx / r
    omega_deg = np.where(r <= R_core, np.degrees(Omega), 0.0)

    I2 = _warp_by_flow(I1, u_field, v_field)
    return I1, I2, u_field, v_field, omega_deg, f"RankineVortex_G{int(Gamma)}_Rc{R_core}"


def make_shear_flow(seed=42, U_max=4.0):

    I1 = generate_speckle_field(FS, PPP, DTAU, 1.0, seed)
    Y, _ = np.mgrid[0:FS, 0:FS]
    u_field = U_max * Y.astype(float) / (FS - 1)
    v_field = np.zeros_like(u_field)
    omega_deg = np.full((FS, FS), np.degrees(U_max / FS))
    I2 = _warp_by_flow(I1, u_field, v_field)
    return I1, I2, u_field, v_field, omega_deg, f"ShearFlow_Umax{int(U_max)}"


def make_uniform_flow(seed=42, u0=4.0, v0=3.0):

    I1 = generate_speckle_field(FS, PPP, DTAU, 1.0, seed)
    u_field = np.full((FS, FS), u0)
    v_field = np.full((FS, FS), v0)
    omega_deg = np.zeros((FS, FS))
    I2 = _warp_by_flow(I1, u_field, v_field)
    return I1, I2, u_field, v_field, omega_deg, f"UniformFlow_u{int(u0)}_v{int(v0)}"


def _ncc_zero(W1, W2):
    w1 = W1 - W1.mean()
    w2 = W2 - W2.mean()
    denom = np.sqrt(np.sum(w1 ** 2) * np.sum(w2 ** 2))
    return float(np.sum(w1 * w2) / denom) if denom > 1e-30 else 0.0


def rotation_window_search(I1, I2, center, subset_size,
                           search_radius=SEARCH_RADIUS, search_step=SEARCH_STEP):


    H, W  = I1.shape
    half  = subset_size // 2
    cr, cc = center

    W1   = I1[cr - half: cr - half + subset_size, cc - half: cc - half + subset_size]
    W2_c = I2[cr - half: cr - half + subset_size, cc - half: cc - half + subset_size]


    with contextlib.redirect_stdout(_io.StringIO()):
        res = estimate_rotation(W1, W2_c, OPT_R_MIN, OPT_WEIGHT,
                                OPT_POC, verbose=False)
    angle_est = res["angle_est"]
    snr       = res["snr"]

    best_nft = dict(ncc=-1.0, dx=0.0, dy=0.0)
    best_raw = dict(ncc=-1.0, dx=0.0, dy=0.0)


    for dy in range(-search_radius, search_radius + 1, search_step):
        for dx in range(-search_radius, search_radius + 1, search_step):
            cr2, cc2 = cr + dy, cc + dx
            if (cr2 - half < 0 or cr2 + half > H
                    or cc2 - half < 0 or cc2 + half > W):
                continue
            W2s = I2[cr2 - half: cr2 - half + subset_size,
                     cc2 - half: cc2 - half + subset_size]


            ncc_raw = _ncc_zero(W1, W2s)
            if ncc_raw > best_raw["ncc"]:
                best_raw = dict(ncc=ncc_raw, dx=float(dx), dy=float(dy))


            W2_corr  = apply_affine(W2s, dx=0, dy=0, angle_deg=-angle_est)
            ncc_corr = _ncc_zero(W1, W2_corr)
            if ncc_corr > best_nft["ncc"]:
                best_nft = dict(ncc=ncc_corr, dx=float(dx), dy=float(dy))

    return dict(

        dx=best_nft["dx"], dy=best_nft["dy"],
        ncc=best_nft["ncc"], angle=angle_est, snr=snr,

        dx_raw=best_raw["dx"], dy_raw=best_raw["dy"],
        ncc_raw=best_raw["ncc"],
        ncc_gain=best_nft["ncc"] - best_raw["ncc"],
    )


def fullfield_dic(I1, I2, subset_size=WS, step=STEP,
                  search_radius=SEARCH_RADIUS, search_step=SEARCH_STEP):

    H, W   = I1.shape
    half   = subset_size // 2
    centers = list(range(half, H - half + 1, step))
    n      = len(centers)

    u_nft = np.full((n, n), np.nan);  v_nft = np.full((n, n), np.nan)
    u_raw = np.full((n, n), np.nan);  v_raw = np.full((n, n), np.nan)
    angle_map  = np.full((n, n), np.nan)
    snr_map    = np.full((n, n), np.nan)
    ncc_map    = np.full((n, n), np.nan)
    ncc_raw_map= np.full((n, n), np.nan)
    ncc_gain_map = np.full((n, n), np.nan)

    for ri, cr in enumerate(centers):
        for ci, cc in enumerate(centers):
            res = rotation_window_search(
                I1, I2, (cr, cc), subset_size, search_radius, search_step)

            u_nft[ri, ci] = res["dx"];    v_nft[ri, ci] = res["dy"]
            u_raw[ri, ci] = res["dx_raw"]; v_raw[ri, ci] = res["dy_raw"]
            angle_map[ri, ci]  = res["angle"]
            snr_map[ri, ci]    = res["snr"]
            ncc_map[ri, ci]    = res["ncc"]
            ncc_raw_map[ri, ci]= res["ncc_raw"]
            ncc_gain_map[ri, ci] = res["ncc_gain"]

    return dict(
        u=u_nft, v=v_nft, u_raw=u_raw, v_raw=v_raw,
        angle=angle_map, snr=snr_map,
        ncc=ncc_map, ncc_raw=ncc_raw_map, ncc_gain=ncc_gain_map,
        centers_r=centers, centers_c=centers, n=n,
    )


def fullfield_ncc(I1, I2, subset_size=WS, step=STEP,
                  search_radius=SEARCH_RADIUS, search_step=SEARCH_STEP):

    H, W   = I1.shape
    half   = subset_size // 2
    centers = list(range(half, H - half + 1, step))
    n      = len(centers)
    u_map  = np.full((n, n), np.nan)
    v_map  = np.full((n, n), np.nan)
    ncc_map= np.full((n, n), np.nan)

    for ri, cr in enumerate(centers):
        for ci, cc in enumerate(centers):
            W1       = I1[cr - half: cr - half + subset_size,
                          cc - half: cc - half + subset_size]
            best_ncc = -1.0
            best_dx, best_dy = 0.0, 0.0
            for dy in range(-search_radius, search_radius + 1, search_step):
                for dx in range(-search_radius, search_radius + 1, search_step):
                    cr2, cc2 = cr + dy, cc + dx
                    if (cr2 - half < 0 or cr2 + half > H
                            or cc2 - half < 0 or cc2 + half > W):
                        continue
                    W2s = I2[cr2 - half: cr2 - half + subset_size,
                             cc2 - half: cc2 - half + subset_size]
                    ncc = _ncc_zero(W1, W2s)
                    if ncc > best_ncc:
                        best_ncc = ncc
                        best_dx, best_dy = float(dx), float(dy)
            u_map[ri, ci]  = best_dx
            v_map[ri, ci]  = best_dy
            ncc_map[ri, ci]= best_ncc

    return dict(u=u_map, v=v_map, ncc=ncc_map,
                centers_r=centers, centers_c=centers, n=n)


def _plot_scene(I1, I2, result, u_gt, v_gt, scene_name, elapsed,
                rmse_nft, rmse_raw):
    n        = result["n"]
    cr_arr   = result["centers_r"]
    cc_arr   = result["centers_c"]
    CC, RR   = np.meshgrid(cc_arr, cr_arr)

    u_max = max(np.nanmax(np.abs(u_gt)), np.nanmax(np.abs(v_gt)), 1.0)
    scale = u_max * 8.0

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))


    axes[0, 0].imshow(I1, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('I1：参考图像')
    axes[0, 0].axis('off')


    axes[0, 1].imshow(I2, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('I2：变形图像')
    axes[0, 1].axis('off')


    axes[0, 2].imshow(I1, cmap='gray', alpha=0.35)
    axes[0, 2].quiver(CC, RR, u_gt, v_gt,
                      color=COLORS[4], scale=scale, width=0.003)
    axes[0, 2].set_title('真值位移场 (u_true, v_true)')
    axes[0, 2].axis('off')


    axes[1, 0].imshow(I2, cmap='gray', alpha=0.35)
    axes[1, 0].quiver(CC, RR, result["u_raw"], result["v_raw"],
                      color=COLORS[0], scale=scale, width=0.003)
    axes[1, 0].set_title(f'NCC（标准互相关）\nRMSE = {rmse_raw:.4f} px')
    axes[1, 0].axis('off')


    axes[1, 1].imshow(I2, cmap='gray', alpha=0.35)
    axes[1, 1].quiver(CC, RR, result["u"], result["v"],
                      color=COLORS[5], scale=scale, width=0.003)
    axes[1, 1].set_title(f'NCC+NUFFT（旋转校正）\nRMSE = {rmse_nft:.4f} px')
    axes[1, 1].axis('off')


    err_ncc = np.sqrt((result["u_raw"] - u_gt) ** 2 +
                      (result["v_raw"] - v_gt) ** 2)
    err_nft = np.sqrt((result["u"] - u_gt) ** 2 +
                      (result["v"] - v_gt) ** 2)
    diff = err_ncc - err_nft
    vmax_d = max(np.nanmax(np.abs(diff)), 0.1)
    im = axes[1, 2].imshow(diff, cmap='user_cmap',
                            vmin=-vmax_d, vmax=vmax_d,
                            extent=[0, FS, FS, 0], aspect='equal')
    improve = (rmse_raw - rmse_nft) / rmse_raw * 100 if rmse_raw > 1e-6 else 0.0
    axes[1, 2].set_title(f'误差改善 (NCC − NCC+NUFFT) px\n改善 {improve:.1f}%')
    plt.colorbar(im, ax=axes[1, 2], shrink=0.85)

    fig.suptitle(f'{scene_name}  |  {n}×{n} 窗口  |  t={elapsed:.1f}s', fontsize=11)
    fig.tight_layout()
    fname = os.path.join(OUT_DIR, f'{scene_name}.png')
    fig.savefig(fname, dpi=130)
    plt.close(fig)
    print(f"    图已保存: results_fullfield/{scene_name}.png")


def _plot_summary(all_stats):
    names    = [s["name"].split("_")[0] for s in all_stats]
    rmse_raw = [s["rmse_raw"] for s in all_stats]
    rmse_nft = [s["rmse_nft"] for s in all_stats]
    improve  = [s["improve"]  for s in all_stats]

    x = np.arange(len(names)); w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    b1 = axes[0].bar(x - w/2, rmse_raw, w, label='NCC',
                     color=COLORS[0], alpha=0.85, edgecolor='k')
    b2 = axes[0].bar(x + w/2, rmse_nft, w, label='NCC+NUFFT',
                     color=COLORS[7], alpha=0.85, edgecolor='k')
    for bar, v in zip(list(b1) + list(b2), rmse_raw + rmse_nft):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.002,
                     f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    axes[0].set_xticks(x); axes[0].set_xticklabels(names)
    axes[0].set_ylabel('位移 RMSE (px)')
    axes[0].set_title('全场位移精度：NCC vs NCC+NUFFT')
    axes[0].legend(); axes[0].grid(True, alpha=0.3, axis='y')

    bars = axes[1].bar(x, improve,
                       color=[COLORS[5] if v > 0 else COLORS[0] for v in improve],
                       alpha=0.85, edgecolor='k')
    for bar, v in zip(bars, improve):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.3,
                     f'{v:.1f}%', ha='center', va='bottom',
                     fontsize=10, fontweight='bold')
    axes[1].axhline(0, color='k', lw=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(names)
    axes[1].set_ylabel('RMSE 改善幅度 (%)')
    axes[1].set_title('NCC+NUFFT 改善幅度（正值=更优）')
    axes[1].grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fname = os.path.join(OUT_DIR, 'fullfield_summary.png')
    fig.savefig(fname, dpi=130)
    plt.close(fig)
    print(f"\n  汇总图已保存: results_fullfield/fullfield_summary.png")


def run_all_scenes():
    set_plot_style()
    print("\n" + "=" * 68)
    print("全场PIV位移场：NCC vs NCC+NUFFT（合成流场，已知真值）")
    print(f"  场尺寸: {FS}×{FS}  窗口: {WS}×{WS}  步长: {STEP}px")
    print("=" * 68)

    scenes = [
        make_rankine_vortex_flow(seed=42, Gamma=1500.0, R_core=50),
        make_shear_flow(seed=42, U_max=4.0),
        make_uniform_flow(seed=42, u0=4.0, v0=3.0),
    ]

    all_stats = []
    for I1, I2, u_true, v_true, omega_deg, scene_name in scenes:
        print(f"\n  [{scene_name}]")
        t0 = time.perf_counter()

        result  = fullfield_dic(I1, I2)
        elapsed = time.perf_counter() - t0

        centers = result["centers_r"]
        u_gt = np.array([[u_true[cr, cc] for cc in centers] for cr in centers])
        v_gt = np.array([[v_true[cr, cc] for cc in centers] for cr in centers])

        err_nft = np.sqrt((result["u"] - u_gt) ** 2 +
                          (result["v"] - v_gt) ** 2)
        err_raw = np.sqrt((result["u_raw"] - u_gt) ** 2 +
                          (result["v_raw"] - v_gt) ** 2)
        rmse_nft = float(np.nanmean(err_nft))
        rmse_raw = float(np.nanmean(err_raw))
        improve  = (rmse_raw - rmse_nft) / rmse_raw * 100 if rmse_raw > 1e-6 else 0.0
        omega_mean = float(np.mean(omega_deg))

        print(f"    NCC       RMSE = {rmse_raw:.4f} px")
        print(f"    NCC+NUFFT RMSE = {rmse_nft:.4f} px  (改善 {improve:.1f}%)")
        print(f"    局部旋转率均值 = {omega_mean:.3f} deg/frame")
        print(f"    用时           = {elapsed:.2f} s")

        _plot_scene(I1, I2, result, u_gt, v_gt,
                    scene_name, elapsed, rmse_nft, rmse_raw)

        all_stats.append(dict(name=scene_name,
                              rmse_raw=rmse_raw, rmse_nft=rmse_nft,
                              improve=improve, elapsed=elapsed,
                              omega_mean=omega_mean))

    _plot_summary(all_stats)

    print("\n" + "=" * 68)
    print("全场汇总")
    print(f"  {'流场':<35s}  {'NCC':>8s}  {'NCC+NUFFT':>10s}  {'改善':>7s}  {'旋转率':>8s}")
    print("  " + "-" * 75)
    for s in all_stats:
        print(f"  {s['name']:<35s}  {s['rmse_raw']:>8.4f}  {s['rmse_nft']:>10.4f}  "
              f"  {s['improve']:>5.1f}%  {s['omega_mean']:>6.3f}°/fr")
    print("=" * 68)

    return all_stats


if __name__ == "__main__":
    run_all_scenes()
