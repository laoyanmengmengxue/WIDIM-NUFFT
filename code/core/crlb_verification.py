import sys, os, io, contextlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from poc_common import (
    generate_speckle_field, apply_affine, extract_window,
    apply_hanning_and_fft, mag_to_polar_bicubic, DEFAULT_PARAMS,
    set_plot_style, COLORS,
)
from poc_point2 import design_polar_grid, nufft_polar_spectrum
from poc_point3 import (
    radial_integrate, circular_cross_corr, find_peak_subpixel,
    estimate_rotation, N_THETA, N_R, OPT_R_MIN, OPT_WEIGHT,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_crlb")
os.makedirs(OUT_DIR, exist_ok=True)


TRUE_ANGLE = 5.0
NOISE_LEVELS = [0.0, 0.005, 0.01, 0.02, 0.05]
N_MC = 200

FS = DEFAULT_PARAMS["field_size"]
WS = DEFAULT_PARAMS["window_size"]


def compute_polar_spectrum_ref(seed=42):

    field = generate_speckle_field(FS, DEFAULT_PARAMS["ppp"],
                                   DEFAULT_PARAMS["d_tau"], 1.0, seed)
    W1 = extract_window(field, WS)
    mag_shifted, windowed = apply_hanning_and_fft(W1)
    with contextlib.redirect_stdout(io.StringIO()):
        r_arr, theta_arr, s_x, s_y = design_polar_grid(WS, N_R, N_THETA, r_min=1)
    polar_mag = nufft_polar_spectrum(windowed, s_x, s_y, N_R, N_THETA, eps=1e-9)
    return polar_mag, r_arr, theta_arr, windowed, s_x, s_y


def compute_angular_gradient(polar_mag, theta_arr):

    d_theta = theta_arr[1] - theta_arr[0]
    dA_dtheta = (np.roll(polar_mag, -1, axis=1)
                 - np.roll(polar_mag, 1, axis=1)) / (2.0 * d_theta)
    return dA_dtheta


def compute_fisher_2d(dA_dtheta, sigma, r_min_idx=8):

    sub = dA_dtheta[r_min_idx:, :]
    return np.sum(sub ** 2) / (sigma ** 2)


def compute_fisher_1d(dA_dtheta, sigma, weight_vec, r_min_idx=8):

    sub = dA_dtheta[r_min_idx:, :]
    weighted_sum = weight_vec @ sub
    numerator = np.sum(weighted_sum ** 2)
    denominator = sigma ** 2 * np.sum(weight_vec ** 2)
    return numerator / denominator


def compute_gradient_covariance_matrix(dA_dtheta, r_min_idx=8):


    sub = dA_dtheta[r_min_idx:, :]
    M = sub @ sub.T
    eigenvalues, eigenvectors = np.linalg.eigh(M)

    W_star = eigenvectors[:, -1]
    if np.sum(W_star) < 0:
        W_star = -W_star
    return M, W_star, eigenvalues


def compute_rms_gradient(dA_dtheta, r_min_idx=8):


    sub = dA_dtheta[r_min_idx:, :]
    return np.sqrt(np.mean(sub ** 2, axis=1))


def make_weight_vec(r_arr, r_min_idx, scheme, W_star=None):

    r_sub = r_arr[r_min_idx:]
    if scheme == 'r0':
        return np.ones_like(r_sub)
    elif scheme == 'r1':
        return r_sub.copy()
    elif scheme == 'r2':
        return r_sub ** 2
    elif scheme == 'r3':
        return r_sub ** 3
    elif scheme == 'Wstar':
        assert W_star is not None
        return W_star.copy()
    else:
        raise ValueError(f"Unknown scheme: {scheme}")


def sigma_spatial_to_freq(sigma_s, N):

    return sigma_s * N / np.sqrt(2.0)


def radial_integrate_custom(polar_mag, r_arr, r_min_idx, weight_vec):

    dr = r_arr[1] - r_arr[0] if len(r_arr) > 1 else 1.0
    sub = polar_mag[r_min_idx:, :]
    w = weight_vec[:, np.newaxis]
    return np.sum(sub * w * dr, axis=0)


def estimate_rotation_bicubic(W1, W2, r_min_idx=8, weight='r2'):


    mag1, _ = apply_hanning_and_fft(W1)
    mag2, _ = apply_hanning_and_fft(W2)
    polar1, r_arr, _ = mag_to_polar_bicubic(mag1, N_R, N_THETA, r_min=1)
    polar2, _, _ = mag_to_polar_bicubic(mag2, N_R, N_THETA, r_min=1)
    S1 = radial_integrate(polar1, r_arr, r_min_idx=r_min_idx, weight=weight)
    S2 = radial_integrate(polar2, r_arr, r_min_idx=r_min_idx, weight=weight)
    C = circular_cross_corr(S1, S2, phase_only=False)
    theta_step = 180.0 / N_THETA
    angle_est, _, snr, fwhm = find_peak_subpixel(C, theta_step)

    return dict(angle_est=-angle_est, snr=snr, fwhm=fwhm)


def make_pair_piv(seed, dx, dy, angle_deg):


    field = generate_speckle_field(FS, DEFAULT_PARAMS["ppp"],
                                   DEFAULT_PARAMS["d_tau"], 1.0, seed)
    W1 = extract_window(field, WS)
    ft = apply_affine(field, dx=dx, dy=dy, angle_deg=angle_deg)

    W2 = extract_window(ft, WS)
    return W1, W2


def run_crlb1():

    print("\n" + "=" * 65)
    print("CRLB-1: Fisher 信息与 CRLB 层级关系")
    print("=" * 65)


    polar_mag, r_arr, theta_arr, windowed, s_x, s_y = \
        compute_polar_spectrum_ref()
    dA = compute_angular_gradient(polar_mag, theta_arr)


    sigma_2img = sigma_spatial_to_freq(0.01 * np.sqrt(2), WS)


    I_2D = compute_fisher_2d(dA, sigma_2img, r_min_idx=OPT_R_MIN)
    crlb_2d = 1.0 / I_2D
    print(f"  sigma_s = 0.01, sigma_F(2img) = {sigma_2img:.4f}")
    print(f"  I_2D = {I_2D:.4e}")
    print(f"  CRLB_2D = {crlb_2d:.4e} rad^2 "
          f"= {np.degrees(np.sqrt(crlb_2d)):.6f} deg")


    M, W_star, eigenvalues = compute_gradient_covariance_matrix(
        dA, OPT_R_MIN)
    lambda_max = eigenvalues[-1]
    I_1D_max = lambda_max / sigma_2img ** 2
    print(f"  lambda_max(M) = {lambda_max:.4e}")
    print(f"  I_1D(W*) = {I_1D_max:.4e}")


    schemes = ['r0', 'r1', 'r2', 'r3', 'Wstar']
    labels = ['W=1', 'W=r', 'W=r^2', 'W=r^3', 'W=W*']
    crlb_1d = {}
    fisher_1d = {}

    print(f"\n  {'Weight':>8s}  {'I_1D':>12s}  {'CRLB_1D(deg)':>13s}  "
          f"{'eta=I1D/I2D':>12s}  Hierarchy")
    print("  " + "-" * 62)

    for scheme, label in zip(schemes, labels):
        w = make_weight_vec(r_arr, OPT_R_MIN, scheme, W_star)
        I_1D = compute_fisher_1d(dA, sigma_2img, w, OPT_R_MIN)
        c1d = 1.0 / I_1D
        eta = I_1D / I_2D
        fisher_1d[scheme] = I_1D
        crlb_1d[scheme] = c1d
        ok = "OK  CRLB_1D >= CRLB_2D" if c1d >= crlb_2d * 0.999 \
            else "VIOLATED"
        print(f"  {label:>8s}  {I_1D:>12.4e}  "
              f"{np.degrees(np.sqrt(c1d)):>13.6f}  {eta:>12.4f}  {ok}")


    sub = dA[OPT_R_MIN:, :]
    r_sub = r_arr[OPT_R_MIN:]
    fisher_per_r = np.sum(sub ** 2, axis=1) / sigma_2img ** 2


    g_r = compute_rms_gradient(dA, OPT_R_MIN)
    g_r_norm = g_r / np.max(g_r) if np.max(g_r) > 0 else g_r
    W_star_norm = np.abs(W_star) / np.max(np.abs(W_star))
    corr_Wstar_g = np.corrcoef(W_star_norm, g_r_norm)[0, 1]
    print(f"\n  Corollary 1 validation:")
    print(f"    corr(W*, g(r)) = {corr_Wstar_g:.4f}  "
          f"(separable approx: W* ~ g(r))")

    results = dict(
        polar_mag=polar_mag, r_arr=r_arr, theta_arr=theta_arr,
        dA=dA, M=M, W_star=W_star, eigenvalues=eigenvalues,
        crlb_2d=crlb_2d, crlb_1d=crlb_1d, fisher_2d=I_2D,
        fisher_1d=fisher_1d, fisher_per_r=fisher_per_r,
        r_sub=r_sub, sigma_2img=sigma_2img, g_r=g_r,
        windowed=windowed, s_x=s_x, s_y=s_y,
    )

    plot_crlb1(results, schemes, labels)
    return results


def plot_crlb1(res, schemes, labels):

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))


    crlb_deg = [np.degrees(np.sqrt(res['crlb_1d'][s])) for s in schemes]
    crlb_2d_deg = np.degrees(np.sqrt(res['crlb_2d']))
    colors = [COLORS[7], COLORS[6], COLORS[5], COLORS[2], COLORS[0]]
    axes[0].bar(labels, crlb_deg, color=colors, alpha=0.8, edgecolor='k')
    axes[0].axhline(crlb_2d_deg, color='k', ls='--', lw=2,
                     label=f'CRLB_2D = {crlb_2d_deg:.4f} deg')
    axes[0].set_ylabel('CRLB (deg)')
    axes[0].set_title('(a) CRLB Hierarchy: 1D vs 2D')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, axis='y')


    r_sub = res['r_sub']
    W_star = res['W_star']
    g_r = res['g_r']
    W_star_norm = np.abs(W_star) / np.max(np.abs(W_star))
    g_r_norm = g_r / np.max(g_r) if np.max(g_r) > 0 else g_r
    r2_norm = (r_sub ** 2) / np.max(r_sub ** 2)
    corr_g = np.corrcoef(W_star_norm, g_r_norm)[0, 1]
    axes[1].plot(r_sub, W_star_norm, 'o-', color=COLORS[0], ms=4, lw=1.5,
                 label='W*(r) (Thm 3 eigvec)')
    axes[1].plot(r_sub, g_r_norm, '--', color=COLORS[7], lw=2,
                 label='g(r) RMS gradient')
    axes[1].plot(r_sub, r2_norm, ':', color=COLORS[2], lw=1.5, alpha=0.6,
                 label='r^2 (SCOT, Thm 4)')
    axes[1].set_xlabel('r (frequency)')
    axes[1].set_ylabel('Weight (normalized)')
    axes[1].set_title(f'(b) W* vs g(r): corr = {corr_g:.4f}')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)


    dr = r_sub[1] - r_sub[0] if len(r_sub) > 1 else 1.0
    axes[2].bar(r_sub, res['fisher_per_r'], width=dr * 0.8,
                color=COLORS[7], alpha=0.7, edgecolor='k')
    axes[2].set_xlabel('r (frequency)')
    axes[2].set_ylabel('I_2D(r) per radial bin')
    axes[2].set_title('(c) Fisher Information Radial Distribution')
    axes[2].grid(True, alpha=0.3)

    fig.suptitle('CRLB-1: Information Hierarchy Verification', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'crlb1_hierarchy.png'), dpi=150)
    plt.close(fig)
    print(f"\n  Figure saved: results_crlb/crlb1_hierarchy.png")


def run_crlb2(crlb1_res):


    print("\n" + "=" * 65)
    print("CRLB-2: RMSE vs CRLB (NUFFT vs Bicubic, PIV mode)")
    print("=" * 65)

    dA = crlb1_res['dA']
    r_arr = crlb1_res['r_arr']


    w_r2 = make_weight_vec(r_arr, OPT_R_MIN, 'r2')


    crlb_theory = {}
    for sigma_s in NOISE_LEVELS:
        if sigma_s == 0:
            crlb_theory[sigma_s] = 0.0
        else:
            sig_f = sigma_spatial_to_freq(sigma_s * np.sqrt(2), WS)
            I_1D = compute_fisher_1d(dA, sig_f, w_r2, OPT_R_MIN)
            crlb_theory[sigma_s] = np.degrees(np.sqrt(1.0 / I_1D))

    print(f"\n  Theoretical CRLB_1D(r^2) [reference, ideal observer]:")
    for sigma_s in NOISE_LEVELS:
        print(f"    sigma_s = {sigma_s:.3f}:  "
              f"CRLB = {crlb_theory[sigma_s]:.6f} deg")


    W1_clean, W2_clean = make_pair_piv(42, dx=4.0, dy=3.0,
                                        angle_deg=TRUE_ANGLE)


    print(f"\n  Monte Carlo (PIV mode): {N_MC} trials x "
          f"{len(NOISE_LEVELS)} noise levels")
    print(f"  Note: PIV Eulerian windows -> particle exchange -> eta << 1")

    rmse_nufft = {}
    rmse_bicubic = {}

    for sigma_s in NOISE_LEVELS:
        errors_nufft = []
        errors_bicubic = []

        for mc in range(N_MC):
            if sigma_s > 0:
                rng = np.random.default_rng(mc + 5000)
                W1n = W1_clean + sigma_s * rng.standard_normal(W1_clean.shape)
                W2n = W2_clean + sigma_s * rng.standard_normal(W2_clean.shape)
            else:
                W1n = W1_clean.copy()
                W2n = W2_clean.copy()


            res_n = estimate_rotation(W1n, W2n, OPT_R_MIN, OPT_WEIGHT,
                                      phase_only=False, verbose=False)
            errors_nufft.append(res_n['angle_est'] - TRUE_ANGLE)


            res_b = estimate_rotation_bicubic(W1n, W2n, OPT_R_MIN,
                                              OPT_WEIGHT)
            errors_bicubic.append(res_b['angle_est'] - TRUE_ANGLE)

        rmse_n = np.sqrt(np.mean(np.array(errors_nufft) ** 2))
        rmse_b = np.sqrt(np.mean(np.array(errors_bicubic) ** 2))
        rmse_nufft[sigma_s] = rmse_n
        rmse_bicubic[sigma_s] = rmse_b

        if rmse_n > 0 and crlb_theory[sigma_s] > 0:
            eta_n = (crlb_theory[sigma_s] / rmse_n) ** 2
        else:
            eta_n = 0
        if rmse_b > 0 and crlb_theory[sigma_s] > 0:
            eta_b = (crlb_theory[sigma_s] / rmse_b) ** 2
        else:
            eta_b = 0

        print(f"    sigma = {sigma_s:.3f}:  "
              f"NUFFT RMSE = {rmse_n:.6f} deg (eta={eta_n:.4f}), "
              f"Bicubic RMSE = {rmse_b:.6f} deg (eta={eta_b:.4f})")

    results = dict(
        crlb_theory=crlb_theory,
        rmse_nufft=rmse_nufft,
        rmse_bicubic=rmse_bicubic,
    )

    plot_crlb2(results)
    return results


def plot_crlb2(res):

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    sigmas = np.array(NOISE_LEVELS)
    crlb = np.array([res['crlb_theory'][s] for s in NOISE_LEVELS])
    rmse_n = np.array([res['rmse_nufft'][s] for s in NOISE_LEVELS])
    rmse_b = np.array([res['rmse_bicubic'][s] for s in NOISE_LEVELS])


    floor = 1e-6
    rmse_n_plot = np.maximum(rmse_n, floor)
    rmse_b_plot = np.maximum(rmse_b, floor)


    nz = sigmas > 0


    axes[0].loglog(sigmas[nz], np.maximum(crlb[nz], floor), '-^', color='k', lw=2,
                   ms=7, label='CRLB (theory)')
    axes[0].loglog(sigmas[nz], rmse_n_plot[nz], '-o', color=COLORS[7], lw=2, ms=6,
                   label='NUFFT + r^2')
    axes[0].loglog(sigmas[nz], rmse_b_plot[nz], '-s', color=COLORS[0], lw=2, ms=6,
                   label='Bicubic + r^2')

    axes[0].axhline(rmse_n_plot[0], color=COLORS[7], ls=':', lw=1, alpha=0.5,
                     label=f'NUFFT floor = {rmse_n[0]:.2e} deg')
    axes[0].axhline(rmse_b_plot[0], color=COLORS[0], ls=':', lw=1, alpha=0.5,
                     label=f'Bicubic floor = {rmse_b[0]:.2e} deg')
    axes[0].set_xlabel('Spatial noise sigma_s')
    axes[0].set_ylabel('RMSE (deg)')
    axes[0].set_title('(a) RMSE vs Noise Level')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, which='both')


    eta_n = np.where((rmse_n > 0) & (crlb > 0),
                     (crlb / rmse_n) ** 2, 0)
    eta_b = np.where((rmse_b > 0) & (crlb > 0),
                     (crlb / rmse_b) ** 2, 0)
    axes[1].plot(sigmas[nz], eta_n[nz], '-o', color=COLORS[7], lw=2, ms=6,
                 label='NUFFT + r^2')
    axes[1].plot(sigmas[nz], eta_b[nz], '-s', color=COLORS[0], lw=2, ms=6,
                 label='Bicubic + r^2')
    axes[1].axhline(1.0, color='k', ls='--', lw=1.5,
                     label='eta = 1 (efficient)')
    axes[1].axhline(0.7, color=COLORS[5], ls=':', lw=1,
                     label='eta = 0.7')
    axes[1].set_xlabel('Spatial noise sigma_s')
    axes[1].set_ylabel('Efficiency eta = CRLB / MSE')
    axes[1].set_title('(b) Estimator Efficiency')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([-0.05, 1.3])

    fig.suptitle('CRLB-2: Monte Carlo RMSE vs Cramer-Rao Lower Bound (PIV mode)',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'crlb2_rmse_vs_crlb.png'), dpi=150)
    plt.close(fig)
    print(f"\n  Figure saved: results_crlb/crlb2_rmse_vs_crlb.png")


def run_crlb3(crlb1_res):

    print("\n" + "=" * 65)
    print("CRLB-3: Optimal Weight Verification")
    print("=" * 65)

    dA = crlb1_res['dA']
    r_arr = crlb1_res['r_arr']
    r_sub = crlb1_res['r_sub']
    W_star = crlb1_res['W_star']


    g_r = crlb1_res['g_r']


    print("\n  Pearson correlation of |W*(r)| with reference profiles:")
    W_star_abs = np.abs(W_star)
    g_r_norm = g_r / np.max(g_r) if np.max(g_r) > 0 else g_r
    corr_g = np.corrcoef(W_star_abs, g_r_norm)[0, 1]
    print(f"    corr(W*, g(r))  = {corr_g:.6f}  (Corollary 1 prediction)")
    for n, label in [(1, 'r^1'), (2, 'r^2'), (3, 'r^3')]:
        rn = r_sub ** n
        corr = np.corrcoef(W_star_abs, rn)[0, 1]
        print(f"    corr(W*, {label:>4s}) = {corr:.6f}")


    sigma_s = 0.01
    sig_f = sigma_spatial_to_freq(sigma_s * np.sqrt(2), WS)

    schemes = ['r0', 'r1', 'r2', 'r3', 'Wstar']
    labels = ['W=1', 'W=r', 'W=r^2', 'W=r^3', 'W=W*']


    weight_map = {'r0': 'none', 'r1': 'r', 'r2': 'r2', 'r3': 'r3'}

    crlb_theory = {}
    for scheme in schemes:
        w = make_weight_vec(r_arr, OPT_R_MIN, scheme, W_star)
        I_1D = compute_fisher_1d(dA, sig_f, w, OPT_R_MIN)
        crlb_theory[scheme] = np.degrees(np.sqrt(1.0 / I_1D))


    W1_clean, W2_clean = make_pair_piv(42, dx=4.0, dy=3.0,
                                        angle_deg=TRUE_ANGLE)


    with contextlib.redirect_stdout(io.StringIO()):
        grid_r, grid_t, grid_sx, grid_sy = design_polar_grid(
            WS, N_R, N_THETA, r_min=1)


    N_MC_3 = 100
    rmse_mc = {}
    print(f"\n  MC ({N_MC_3} trials, sigma_s = {sigma_s}):")

    for scheme, label in zip(schemes, labels):
        errors = []
        for mc in range(N_MC_3):
            rng = np.random.default_rng(mc + 7000)
            W1n = W1_clean + sigma_s * rng.standard_normal(W1_clean.shape)
            W2n = W2_clean + sigma_s * rng.standard_normal(W2_clean.shape)

            if scheme == 'Wstar':

                _, windowed1 = apply_hanning_and_fft(W1n)
                _, windowed2 = apply_hanning_and_fft(W2n)
                p1 = nufft_polar_spectrum(windowed1, grid_sx, grid_sy,
                                          N_R, N_THETA)
                p2 = nufft_polar_spectrum(windowed2, grid_sx, grid_sy,
                                          N_R, N_THETA)
                S1 = radial_integrate_custom(p1, grid_r, OPT_R_MIN,
                                             W_star)
                S2 = radial_integrate_custom(p2, grid_r, OPT_R_MIN,
                                             W_star)
                C = circular_cross_corr(S1, S2, phase_only=False)
                theta_step = 180.0 / N_THETA
                angle_est, _, _, _ = find_peak_subpixel(C, theta_step)
            else:
                wn = weight_map[scheme]
                res = estimate_rotation(W1n, W2n, OPT_R_MIN, wn,
                                        phase_only=False, verbose=False)
                angle_est = res['angle_est']

            errors.append(angle_est - TRUE_ANGLE)

        rmse = np.sqrt(np.mean(np.array(errors) ** 2))
        rmse_mc[scheme] = rmse
        eta = (crlb_theory[scheme] / rmse) ** 2 if rmse > 0 else 0
        print(f"    {label:>8s}: CRLB = {crlb_theory[scheme]:.6f} deg, "
              f"RMSE = {rmse:.6f} deg, eta = {eta:.3f}")

    results = dict(
        crlb_theory=crlb_theory,
        rmse_mc=rmse_mc,
        W_star=W_star,
        r_sub=r_sub,
        g_r=g_r,
        schemes=schemes,
        labels=labels,
    )

    plot_crlb3(results)
    return results


def plot_crlb3(res):

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    schemes = res['schemes']
    labels = res['labels']
    x = np.arange(len(schemes))
    crlb_vals = [res['crlb_theory'][s] for s in schemes]
    rmse_vals = [res['rmse_mc'][s] for s in schemes]


    w_bar = 0.35
    axes[0].bar(x - w_bar / 2, crlb_vals, w_bar, label='CRLB (theory)',
                color=COLORS[7], alpha=0.8, edgecolor='k')
    axes[0].bar(x + w_bar / 2, rmse_vals, w_bar, label='RMSE (MC)',
                color=COLORS[0], alpha=0.8, edgecolor='k')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel('CRLB / RMSE (deg)')
    axes[0].set_title('(a) CRLB Theory vs MC RMSE (sigma_s = 0.01)')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, axis='y')


    r_sub = res['r_sub']
    W_star = res['W_star']
    g_r = res['g_r']
    W_abs = np.abs(W_star) / np.max(np.abs(W_star))
    g_r_norm = g_r / np.max(g_r) if np.max(g_r) > 0 else g_r
    r2_norm = (r_sub ** 2) / np.max(r_sub ** 2)

    axes[1].plot(r_sub, W_abs, 'o-', color=COLORS[0], ms=5, lw=2,
                 label='W* (Thm 3 eigvec)')
    axes[1].plot(r_sub, g_r_norm, '-s', color=COLORS[7], ms=3, lw=1.5,
                 label='g(r) RMS gradient')
    axes[1].plot(r_sub, r2_norm, '--', color=COLORS[2], lw=2,
                 label='r^2 (SCOT, Thm 4)')
    corr_g = np.corrcoef(W_abs, g_r_norm)[0, 1]
    axes[1].set_xlabel('r (frequency)')
    axes[1].set_ylabel('Weight (normalized)')
    axes[1].set_title(f'(b) W* vs g(r): corr = {corr_g:.4f}')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('CRLB-3: Optimal Weight Verification', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'crlb3_optimal_weight.png'), dpi=150)
    plt.close(fig)
    print(f"\n  Figure saved: results_crlb/crlb3_optimal_weight.png")


def main():
    set_plot_style()
    print("=" * 65)
    print("CRLB Numerical Verification")
    print("  Cramer-Rao Lower Bounds for Rotation Estimation")
    print("=" * 65)

    res1 = run_crlb1()
    res2 = run_crlb2(res1)
    res3 = run_crlb3(res1)


    print("\n" + "=" * 65)
    print("Summary Verification Report")
    print("=" * 65)


    all_hierarchy = all(
        res1['crlb_1d'][s] >= res1['crlb_2d'] * 0.999
        for s in ['r0', 'r1', 'r2', 'r3', 'Wstar']
    )
    print(f"  [{'PASS' if all_hierarchy else 'FAIL'}] "
          f"Hierarchy: CRLB_1D(W) >= CRLB_2D for all W")


    op_sigmas = [s for s in NOISE_LEVELS if 0 < s <= 0.02]
    etas = []
    for s in op_sigmas:
        if res2['crlb_theory'][s] > 0 and res2['rmse_nufft'][s] > 0:
            etas.append((res2['crlb_theory'][s]
                         / res2['rmse_nufft'][s]) ** 2)
    mean_eta = np.mean(etas) if etas else 0
    print(f"  [{'PASS' if mean_eta > 0.7 else 'WARN'}] "
          f"NUFFT mean efficiency eta = {mean_eta:.3f} "
          f"(sigma <= 0.02 operating range)")


    bic_floor = res2['rmse_bicubic'].get(0.0, 0)
    nufft_floor = res2['rmse_nufft'].get(0.0, 0)
    print(f"  [{'PASS' if bic_floor > nufft_floor * 2 else 'WARN'}] "
          f"Bicubic bias floor = {bic_floor:.6f} deg "
          f"vs NUFFT = {nufft_floor:.6f} deg")


    g_r = res1['g_r']
    W_star = res1['W_star']
    g_r_norm = g_r / np.max(g_r) if np.max(g_r) > 0 else g_r
    W_star_norm = np.abs(W_star) / np.max(np.abs(W_star))
    corr_g = np.corrcoef(W_star_norm, g_r_norm)[0, 1]
    print(f"  [{'PASS' if corr_g > 0.7 else 'WARN'}] "
          f"corr(W*, g(r)) = {corr_g:.4f} (Corollary 1, approx)")


    Wstar_best = (res1['fisher_1d']['Wstar']
                  >= max(res1['fisher_1d'][s]
                         for s in ['r0', 'r1', 'r2', 'r3']))
    print(f"  [{'PASS' if Wstar_best else 'FAIL'}] "
          f"I_1D(W*) >= I_1D(r^n) for all n")

    print(f"\n  Output directory: {OUT_DIR}/")
    print("  Generated figures:")
    print("    crlb1_hierarchy.png      -- CRLB hierarchy")
    print("    crlb2_rmse_vs_crlb.png   -- RMSE vs CRLB")
    print("    crlb3_optimal_weight.png -- Optimal weight verification")
    print("=" * 65)


if __name__ == "__main__":
    main()
