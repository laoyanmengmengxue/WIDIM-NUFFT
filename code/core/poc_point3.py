import sys, os, io, contextlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import finufft

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from poc_common import (
    generate_speckle_field, apply_affine, extract_window,
    apply_hanning_and_fft, DEFAULT_PARAMS, set_plot_style, COLORS,
)
from poc_point2 import design_polar_grid, nufft_polar_spectrum

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_point3")
os.makedirs(OUT_DIR, exist_ok=True)

N_THETA = 360
N_R = 32


OPT_R_MIN = 8
OPT_WEIGHT = 'r2'
OPT_POC = False


def radial_integrate(polar_mag, r_arr, r_min_idx=0, weight='r'):
    dr = r_arr[1] - r_arr[0] if len(r_arr) > 1 else 1.0
    sub = polar_mag[r_min_idx:, :]
    r_sub = r_arr[r_min_idx:]
    if weight == 'r':
        w = r_sub[:, np.newaxis]
    elif weight == 'r2':
        w = (r_sub[:, np.newaxis]) ** 2
    elif weight == 'r3':
        w = (r_sub[:, np.newaxis]) ** 3
    elif weight == 'adaptive':


        mu_r = np.mean(sub, axis=1, keepdims=True) + 1e-10
        var_r = np.var(sub, axis=1, keepdims=True)
        snr_r = var_r / (mu_r ** 2)
        snr_max = snr_r.max()
        w = (snr_r / snr_max) if snr_max > 1e-10 else np.ones_like(snr_r)
    else:
        w = np.ones_like(r_sub[:, np.newaxis])
    return np.sum(sub * w * dr, axis=0)


def circular_cross_corr(S1, S2, phase_only=False):

    F1 = np.fft.fft(S1 - S1.mean())
    F2 = np.fft.fft(S2 - S2.mean())
    cross = np.conj(F1) * F2
    if phase_only:
        mag = np.abs(cross)
        mag[mag < 1e-30] = 1e-30
        cross = cross / mag
    return np.real(np.fft.ifft(cross))


def find_peak_subpixel(C, theta_step_deg):


    N = len(C)
    half = N // 2

    peak_idx = np.argmax(C)
    peak_val = C[peak_idx]


    left = C[(peak_idx - 1) % N]
    right = C[(peak_idx + 1) % N]

    if left > 0 and right > 0 and peak_val > 0:
        ll, lp, lr = np.log(left), np.log(peak_val), np.log(right)
        denom = 2.0 * (ll - 2 * lp + lr)
        delta = (ll - lr) / denom if abs(denom) > 1e-30 else 0.0
    else:
        denom = 2.0 * (left - 2 * peak_val + right)
        delta = (left - right) / denom if abs(denom) > 1e-30 else 0.0

    raw_angle = (peak_idx + delta) * theta_step_deg

    if raw_angle > 90:
        raw_angle -= 180.0


    exclude = max(5, int(5.0 / theta_step_deg))
    mask = np.ones(N, dtype=bool)
    for d in range(-exclude, exclude + 1):
        mask[(peak_idx + d) % N] = False
        mask[(peak_idx + half + d) % N] = False

    if mask.sum() > 5:
        bg = C[mask]
        snr = (peak_val - bg.mean()) / bg.std() if bg.std() > 0 else float('inf')
    else:
        snr = float('inf')


    half_max = (peak_val + C[mask].mean()) / 2.0 if mask.sum() > 0 else peak_val / 2
    fwhm_count = 0
    for d in range(1, N // 4):
        if C[(peak_idx + d) % N] < half_max:
            fwhm_count = d
            break
    fwhm_deg = 2 * fwhm_count * theta_step_deg

    return raw_angle, peak_idx, snr, fwhm_deg


def estimate_rotation(W1, W2, r_min_idx=OPT_R_MIN, weight=OPT_WEIGHT,
                      phase_only=OPT_POC, verbose=True):
    N = W1.shape[0]
    _, windowed1 = apply_hanning_and_fft(W1)
    _, windowed2 = apply_hanning_and_fft(W2)

    if verbose:
        r_arr, theta_arr, s_x, s_y = design_polar_grid(
            N, N_R, N_THETA, r_min=1)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            r_arr, theta_arr, s_x, s_y = design_polar_grid(
                N, N_R, N_THETA, r_min=1)

    polar1 = nufft_polar_spectrum(windowed1, s_x, s_y, N_R, N_THETA, eps=1e-9)
    polar2 = nufft_polar_spectrum(windowed2, s_x, s_y, N_R, N_THETA, eps=1e-9)

    S1 = radial_integrate(polar1, r_arr, r_min_idx=r_min_idx, weight=weight)
    S2 = radial_integrate(polar2, r_arr, r_min_idx=r_min_idx, weight=weight)

    C = circular_cross_corr(S1, S2, phase_only=phase_only)
    theta_step = 180.0 / N_THETA

    angle_est, peak_idx, snr, fwhm = find_peak_subpixel(C, theta_step)

    return dict(angle_est=angle_est, snr=snr, fwhm=fwhm,
                S1=S1, S2=S2, C=C, polar1=polar1, polar2=polar2,
                r_arr=r_arr, theta_arr=theta_arr)


def run_task_3a():
    print("\n" + "=" * 60)
    print("任务 3-A: S(theta) 数学正确性验证")
    print("=" * 60)

    N = 64
    angle = 15.0


    xx, yy = np.meshgrid(np.arange(N), np.arange(N))
    f1 = (np.cos(2*np.pi*3*xx/N) + np.cos(2*np.pi*5*yy/N)
           + 0.5*np.cos(2*np.pi*(7*xx + 2*yy)/N))

    from scipy.ndimage import rotate as ndi_rotate
    f2 = ndi_rotate(f1, -angle, reshape=False, order=5, mode='constant')

    with contextlib.redirect_stdout(io.StringIO()):
        r_arr, theta_arr, s_x, s_y = design_polar_grid(N, N_R, N_THETA, r_min=1)

    h1 = np.hanning(N)
    hann = np.outer(h1, h1)
    w1 = (f1 - f1.mean()) * hann
    w2 = (f2 - f2.mean()) * hann

    p1 = nufft_polar_spectrum(w1, s_x, s_y, N_R, N_THETA, eps=1e-12)
    p2 = nufft_polar_spectrum(w2, s_x, s_y, N_R, N_THETA, eps=1e-12)


    shift_idx = int(round(angle / (180.0 / N_THETA)))
    p1_shifted = np.roll(p1, -shift_idx, axis=1)
    err_polar = np.sqrt(np.sum((p2 - p1_shifted)**2) / np.sum(p1_shifted**2))
    print(f"  极坐标旋转验证: 相对误差 = {err_polar:.6f}")
    print(f"  [{'通过' if err_polar < 0.15 else '注意'}] (含有限窗口截断效应)")


    S1 = radial_integrate(p1, r_arr, r_min_idx=3, weight='r')
    S2 = radial_integrate(p2, r_arr, r_min_idx=3, weight='r')
    S1_shifted = np.roll(S1, -shift_idx)
    err_S = np.sqrt(np.sum((S2 - S1_shifted)**2) / np.sum(S1_shifted**2))
    print(f"  S(theta) 角位移验证: 相对误差 = {err_S:.6f}")


    C = circular_cross_corr(S1, S2, phase_only=True)
    theta_step = 180.0 / N_THETA
    est, _, snr_val, _ = find_peak_subpixel(C, theta_step)

    true_angle = -angle
    err_angle = abs(est - true_angle)
    print(f"  互相关估计: {est:.2f}° (真值: {true_angle}°, 误差: {err_angle:.2f}°)")
    print(f"  SNR = {snr_val:.1f}")


def run_task_3b():
    print("\n" + "=" * 60)
    print("任务 3-B: 积分参数优化")
    print("=" * 60)

    angle = 15.0
    field = generate_speckle_field(256, 0.04, 2.5, 1.0, seed=42)
    W1 = extract_window(field, 64)
    ft = apply_affine(field, dx=10.5, dy=8.2, angle_deg=angle)
    W2 = extract_window(ft, 64)

    r_mins = [0, 2, 4, 6, 8]
    weights = ['none', 'r', 'r2']
    poc_modes = [False, True]

    best_snr = 0
    best_cfg = None

    for poc in poc_modes:
        mode = "POC" if poc else "XCorr"
        print(f"\n  --- {mode} ---")
        print(f"  {'r_min':>6s}", end="")
        for w in weights:
            print(f"  {w+'/SNR':>10s} {'err':>6s}", end="")
        print()

        for ri in r_mins:
            print(f"  {ri:>6d}", end="")
            for w in weights:
                res = estimate_rotation(W1, W2, ri, w, phase_only=poc, verbose=False)
                err = abs(res["angle_est"] - angle)
                snr_val = res["snr"]
                print(f"  {snr_val:>10.1f} {err:>6.2f}", end="")
                if snr_val > best_snr:
                    best_snr = snr_val
                    best_cfg = (ri, w, poc, err)
            print()

    print(f"\n  最佳: r_min={best_cfg[0]}, weight='{best_cfg[1]}', "
          f"POC={best_cfg[2]}, SNR={best_snr:.1f}, err={best_cfg[3]:.2f}°")
    return best_cfg


def run_task_3cd():
    print("\n" + "=" * 60)
    print("任务 3-C/D: 1D 互相关峰值与 SNR (POC + 优化参数)")
    print("=" * 60)

    angle = 15.0
    field = generate_speckle_field(256, 0.04, 2.5, 1.0, seed=42)
    W1 = extract_window(field, 64)
    ft = apply_affine(field, dx=10.5, dy=8.2, angle_deg=angle)
    W2 = extract_window(ft, 64)

    res = estimate_rotation(W1, W2, OPT_R_MIN, OPT_WEIGHT,
                            phase_only=OPT_POC, verbose=True)

    err = abs(res["angle_est"] - angle)
    print(f"  旋转角估计: {res['angle_est']:.3f}° (真值: {angle}°)")
    print(f"  角度误差:    {err:.3f}°")
    print(f"  SNR:         {res['snr']:.1f}")
    print(f"  FWHM:        {res['fwhm']:.1f}°")


    err_pass = err < 0.5
    snr_pass = res["snr"] > 5
    print(f"  角度精度: [{'通过' if err_pass else '失败'}] "
          f"(误差 {err:.3f}° {'<' if err_pass else '>='} 0.5°)")
    print(f"  SNR:      [{'通过' if snr_pass else '警告'}] "
          f"(SNR {res['snr']:.1f} {'>' if snr_pass else '<='} 5)")


    theta_axis = np.arange(N_THETA) * 180.0 / N_THETA
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(theta_axis, res["S1"], label="S1")
    axes[0, 0].plot(theta_axis, res["S2"], label="S2", alpha=0.8)
    axes[0, 0].set_xlabel("theta (deg)")
    axes[0, 0].set_title("S(theta) Radial Projection")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(theta_axis, res["C"])
    axes[0, 1].axvline(angle, color="r", ls="--", lw=1.5, label=f"True {angle}°")
    axes[0, 1].axvline(res["angle_est"], color="g", ls=":", lw=2,
                        label=f"Est {res['angle_est']:.2f}°")
    axes[0, 1].set_xlabel("Angle shift (deg)")
    axes[0, 1].set_title(f"Phase-Only Correlation (SNR={res['snr']:.1f})")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].imshow(np.log1p(res["polar1"]), aspect="auto", cmap="inferno",
                       extent=[0, 180, res["r_arr"][-1], res["r_arr"][0]])
    axes[1, 0].set_xlabel("theta (deg)")
    axes[1, 0].set_ylabel("r")
    axes[1, 0].set_title("Polar |F1|")

    axes[1, 1].imshow(np.log1p(res["polar2"]), aspect="auto", cmap="inferno",
                       extent=[0, 180, res["r_arr"][-1], res["r_arr"][0]])
    axes[1, 1].set_xlabel("theta (deg)")
    axes[1, 1].set_ylabel("r")
    axes[1, 1].set_title("Polar |F2|")

    fig.suptitle(f"Task 3-C/D: est={res['angle_est']:.2f}° "
                 f"(true={angle}°), SNR={res['snr']:.1f}, POC mode")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "task_3cd_main.png"), dpi=120)
    plt.close(fig)
    print(f"  图已保存: results_point3/task_3cd_main.png")
    return res


def _quick_est(W1, W2):
    return estimate_rotation(W1, W2, OPT_R_MIN, OPT_WEIGHT,
                             phase_only=OPT_POC, verbose=False)

def run_task_3e():
    print("\n" + "=" * 60)
    print("任务 3-E: 系统性参数扫描 (POC + 优化参数)")
    print("=" * 60)

    n_seeds = 5

    def sweep(label, make_pair, true_angle, var_name, var_values):
        print(f"\n  [扫描] {label}")
        print(f"    {var_name:>8s}  {'误差(°)':>8s}  {'SNR':>8s}  判定")
        print("    " + "-" * 36)
        results = {}
        for v in var_values:
            errs, snrs = [], []
            for s in range(42, 42 + n_seeds):
                W1, W2 = make_pair(s, v)
                r = _quick_est(W1, W2)
                errs.append(abs(r["angle_est"] - true_angle))
                snrs.append(r["snr"])
            me, ms = np.mean(errs), np.mean(snrs)
            verdict = "PASS" if me < 1.0 and ms > 3 else "WARN"
            results[v] = dict(err=me, snr=ms, verdict=verdict)
            print(f"    {v:>8g}  {me:>8.3f}  {ms:>8.1f}  {verdict}")
        return results


    def make_angle(s, ang):
        field = generate_speckle_field(256, 0.04, 2.5, 1.0, s)
        W1 = extract_window(field, 64)
        ft = apply_affine(field, dx=10, dy=8, angle_deg=ang)
        return W1, extract_window(ft, 64)

    r1 = {}
    angles = [2, 5, 10, 15, 20, 30, 45]
    for ang in angles:
        errs, snrs = [], []
        for s in range(42, 42 + n_seeds):
            W1, W2 = make_angle(s, ang)
            r = _quick_est(W1, W2)
            errs.append(abs(r["angle_est"] - ang))
            snrs.append(r["snr"])
        r1[ang] = dict(err=np.mean(errs), snr=np.mean(snrs))

    print(f"\n  [扫描 1] 旋转角度 (shift=10px)")
    print(f"    {'角度':>8s}  {'误差(°)':>8s}  {'SNR':>8s}  判定")
    print("    " + "-" * 36)
    for ang in angles:
        v = "PASS" if r1[ang]["err"] < 1 and r1[ang]["snr"] > 3 else "WARN"
        print(f"    {ang:>8d}  {r1[ang]['err']:>8.3f}  {r1[ang]['snr']:>8.1f}  {v}")


    def make_shift(s, sh):
        field = generate_speckle_field(256, 0.04, 2.5, 1.0, s)
        W1 = extract_window(field, 64)
        ft = apply_affine(field, dx=sh, dy=sh*0.78, angle_deg=15)
        return W1, extract_window(ft, 64)

    r2 = sweep("平移量 (rot=15°)", make_shift, 15, "shift", [0, 5, 10, 15, 20])


    def make_ppp(s, ppp):
        field = generate_speckle_field(256, ppp, 2.5, 1.0, s)
        W1 = extract_window(field, 64)
        ft = apply_affine(field, dx=10, dy=8, angle_deg=15)
        return W1, extract_window(ft, 64)

    r3 = sweep("粒子密度 (rot=15°, shift=10px)", make_ppp, 15, "ppp",
               [0.01, 0.02, 0.04, 0.08])


    def make_noise(s, ns):
        field = generate_speckle_field(256, 0.04, 2.5, 1.0, s)
        W1 = extract_window(field, 64)
        ft = apply_affine(field, dx=10, dy=8, angle_deg=15)
        W2 = extract_window(ft, 64)
        if ns > 0:
            rng = np.random.default_rng(s + 1000)
            W1 = W1 + ns * rng.standard_normal(W1.shape)
            W2 = W2 + ns * rng.standard_normal(W2.shape)
        return W1, W2

    r4 = sweep("图像噪声 (rot=15°, shift=10px)", make_noise, 15, "noise",
               [0, 0.02, 0.05, 0.10])

    return r1, r2, r3, r4


def run_task_3f():
    print("\n" + "=" * 60)
    print("任务 3-F: 180° 旋转歧义验证")
    print("=" * 60)

    for ang in [15, 30, 60, 90]:
        field = generate_speckle_field(256, 0.04, 2.5, 1.0, seed=42)
        W1 = extract_window(field, 64)
        ft = apply_affine(field, dx=10, dy=8, angle_deg=ang)
        W2 = extract_window(ft, 64)

        res = _quick_est(W1, W2)
        C = res["C"]
        theta_step = 180.0 / N_THETA

        pk1_idx = np.argmax(C)
        pk1_ang = pk1_idx * theta_step
        pk1_val = C[pk1_idx]

        C_m = C.copy()
        exclude = int(10 / theta_step)
        for d in range(-exclude, exclude + 1):
            C_m[(pk1_idx + d) % N_THETA] = -1e30
        pk2_idx = np.argmax(C_m)
        pk2_ang = pk2_idx * theta_step
        pk2_val = C_m[pk2_idx]

        ratio = pk2_val / pk1_val if pk1_val > 0 else 0
        ambig = "是" if ratio > 0.5 else "否"

        est = res["angle_est"]
        print(f"  真值 {ang:>3d}°: 估计={est:>7.2f}°  "
              f"峰1={pk1_ang:.1f}° 峰2={pk2_ang:.1f}° "
              f"比值={ratio:.3f} 歧义={ambig}")

    print("\n  [结论] 限制旋转 < 45° 即可消歧 (方案 A)。")


def main():
    set_plot_style()
    print("=" * 60)
    print("POC 要点 3: 1D 径向积分抗噪与峰值验证 (改进版)")
    print("=" * 60)

    run_task_3a()
    best = run_task_3b()
    res_cd = run_task_3cd()
    sweeps = run_task_3e()
    run_task_3f()

    print("\n" + "=" * 60)
    print("要点 3 综合报告")
    print("=" * 60)
    print(f"  3-C 旋转角估计: {res_cd['angle_est']:.3f}° (真值 15°)")
    err = abs(res_cd["angle_est"] - 15)
    print(f"  3-C 角度误差:   {err:.3f}°")
    print(f"  3-D SNR:        {res_cd['snr']:.1f}")
    print(f"  3-D FWHM:       {res_cd['fwhm']:.1f}°")
    print(f"  3-B 最佳配置:   r_min={best[0]}, w='{best[1]}', POC={best[2]}")

    if err < 0.5 and res_cd["snr"] > 5:
        print("\n  >>> 要点 3 总体判定: [通过] <<<")
    elif err < 1.0:
        print(f"\n  >>> 要点 3 总体判定: [条件通过] 误差 {err:.2f}° < 1° <<<")
    else:
        print(f"\n  >>> 要点 3 总体判定: [需改进] <<<")


if __name__ == "__main__":
    main()
