import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import affine_transform, map_coordinates
from scipy.interpolate import RectBivariateSpline
from numpy.fft import fft2, fftshift


def check_environment():

    print("=" * 60)
    print("任务 0-A: 环境与依赖库确认")
    print("=" * 60)
    libs = {}
    for name in ["numpy", "scipy", "matplotlib", "skimage", "finufft"]:
        try:
            mod = __import__(name)
            ver = getattr(mod, "__version__", "unknown")
            libs[name] = ver
            print(f"  [OK] {name:15s} v{ver}")
        except ImportError:
            print(f"  [FAIL] {name:15s} — 未安装!")
            libs[name] = None
    print("-" * 60)

    try:
        import finufft
        x = np.array([0.0, 1.0], dtype=np.float64)
        y = np.array([0.0, 1.0], dtype=np.float64)
        c = np.ones(2, dtype=np.complex128)
        out = finufft.nufft2d1(x, y, c, (4, 4), eps=1e-6)
        print("  [OK] FINUFFT type-1 2D 功能验证通过")
    except Exception as e:
        print(f"  [FAIL] FINUFFT 功能验证失败: {e}")

    print("=" * 60)
    return libs


DEFAULT_PARAMS = dict(
    field_size=256,
    window_size=64,
    ppp=0.04,
    d_tau=2.5,
    intensity=1.0,
    seed=42,
)


def generate_speckle_field(field_size=256, ppp=0.04, d_tau=2.5,
                           intensity=1.0, seed=42):

    rng = np.random.default_rng(seed)
    n_particles = int(ppp * field_size * field_size)
    sigma = d_tau / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    cx = rng.uniform(0, field_size, n_particles)
    cy = rng.uniform(0, field_size, n_particles)

    rad = int(np.ceil(3 * sigma))
    field = np.zeros((field_size, field_size), dtype=np.float64)

    for i in range(n_particles):
        x0, y0 = cx[i], cy[i]
        ix_min = max(0, int(np.floor(x0)) - rad)
        ix_max = min(field_size - 1, int(np.floor(x0)) + rad)
        iy_min = max(0, int(np.floor(y0)) - rad)
        iy_max = min(field_size - 1, int(np.floor(y0)) + rad)
        for iy in range(iy_min, iy_max + 1):
            for ix in range(ix_min, ix_max + 1):
                dx = ix - x0
                dy = iy - y0
                field[iy, ix] += intensity * np.exp(
                    -(dx * dx + dy * dy) / (2.0 * sigma * sigma))

    fmax = field.max()
    if fmax > 0:
        field /= fmax
    return field


def apply_affine(field, dx, dy, angle_deg, center=None):

    H, W = field.shape
    if center is None:
        center = np.array([(H - 1) / 2.0, (W - 1) / 2.0])

    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    R_inv = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    offset = center - R_inv @ center + np.array([dy, dx])

    return affine_transform(field, R_inv, offset=offset,
                            order=3, mode='constant', cval=0.0)


def extract_window(field, window_size=64, center=None):

    H, W = field.shape
    if center is None:
        cr, cc = H // 2, W // 2
    else:
        cr, cc = center
    half = window_size // 2
    return field[cr - half: cr - half + window_size,
                 cc - half: cc - half + window_size].copy()


def apply_hanning_and_fft(window):

    N = window.shape[0]
    w = window - window.mean()
    h1 = np.hanning(N)
    hann2d = np.outer(h1, h1)
    w = w * hann2d
    F = fftshift(fft2(w))
    return np.abs(F), w


def mag_to_polar_bicubic(mag_shifted, N_r=32, N_theta=360,
                         r_min=1, r_max=None):


    N = mag_shifted.shape[0]
    if r_max is None:
        r_max = N / 2

    r_arr = np.linspace(r_min, r_max, N_r)

    theta_arr = np.linspace(0, np.pi, N_theta, endpoint=False)


    center = N // 2
    u_axis = np.arange(N) - center


    spline = RectBivariateSpline(u_axis, u_axis, mag_shifted, kx=3, ky=3)


    rr, tt = np.meshgrid(r_arr, theta_arr, indexing='ij')
    uu = rr * np.cos(tt)
    vv = rr * np.sin(tt)

    polar = spline.ev(vv.ravel(), uu.ravel()).reshape(N_r, N_theta)
    polar = np.maximum(polar, 0)
    return polar, r_arr, theta_arr


def ncc_polar(polar1, polar2, r_arr, r_max, angle_shift_idx=0,
              r_threshold_frac=0.0):


    r_threshold = r_threshold_frac * r_max
    r_mask = r_arr > r_threshold

    p1 = np.roll(polar1, -angle_shift_idx, axis=1)
    a = p1[r_mask, :].ravel().astype(np.float64)
    b = polar2[r_mask, :].ravel().astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    num = np.dot(a, b)
    denom = np.sqrt(np.dot(a, a) * np.dot(b, b))
    return num / denom if denom > 0 else 0.0


def freq_radius_grid(N):

    u = np.arange(N) - N // 2
    uu, vv = np.meshgrid(u, u)
    return np.sqrt(uu ** 2 + vv ** 2)


COLORS = ['#FC757B', '#F97F5F', '#FAA26F', '#FDCD94', '#FEE199', '#B0D6A9', '#65BDBA', '#3C9BC9']

def set_plot_style():

    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=COLORS)


    try:
        cmap = LinearSegmentedColormap.from_list('user_cmap', COLORS)
        plt.colormaps.register(cmap)
    except ValueError:
        pass

    plt.rcParams['image.cmap'] = 'user_cmap'


if __name__ == "__main__":
    check_environment()
