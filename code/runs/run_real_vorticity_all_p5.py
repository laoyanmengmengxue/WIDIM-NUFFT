from __future__ import annotations

import argparse
import contextlib
import csv
import io as _io
import json
import re
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import new_comparison as nc
from comparison import _compute_gradients, widim_fullfield as widim_fullfield_base
from poc_point3 import OPT_POC, OPT_R_MIN, OPT_WEIGHT, estimate_rotation


WS = nc.WS
STEP = 10
SEARCH_RADIUS = nc.SEARCH_RADIUS
SEARCH_STEP = nc.SEARCH_STEP
DEFAULT_GATE_PERCENTILE = 5.0


@dataclass(frozen=True)
class PairItem:
    source: str
    label: str
    p1: Path
    p2: Path


def _safe(v: float | None) -> str:
    if v is None:
        return "nan"
    try:
        if not np.isfinite(v):
            return "nan"
    except Exception:
        return "nan"
    return f"{float(v):.6f}"


def _safe_short(v: float | None) -> str:
    if v is None:
        return "nan"
    try:
        if not np.isfinite(v):
            return "nan"
    except Exception:
        return "nan"
    return f"{float(v):.3f}"


def _slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s.strip())
    return s.strip("_") or "pair"


def _eta_str(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "nan"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _find_data_root(repo_root: Path, user_data_root: str) -> Path:
    if user_data_root:
        p = Path(user_data_root).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"--data-root not found: {p}")
        return p

    default_p = repo_root / "真实数据涡流"
    if default_p.is_dir():
        return default_p

    hits = list(repo_root.rglob("PIV book"))
    if hits:
        return hits[0].parent
    raise FileNotFoundError("Cannot locate data root. Please pass --data-root.")


def _find_subdir(root: Path, name: str) -> Path | None:
    p = root / name
    if p.is_dir():
        return p
    hits = [x for x in root.rglob(name) if x.is_dir()]
    return hits[0] if hits else None


def _collect_piv_c(folder: Path) -> list[PairItem]:
    out: list[PairItem] = []
    for p1 in sorted(folder.rglob("c*a.bmp")):
        if not re.match(r"^c\d+a\.bmp$", p1.name, flags=re.IGNORECASE):
            continue
        p2 = p1.with_name(re.sub(r"a\.bmp$", "b.bmp", p1.name, flags=re.IGNORECASE))
        if p2.exists():
            out.append(PairItem("PIV_C", p1.stem[:-1], p1, p2))
    return out


def _collect_piv_e(folder: Path) -> list[PairItem]:
    out: list[PairItem] = []
    frame_re = re.compile(r"^(E_camera_[^_]+_frame_)(\d+)\.tif$", re.IGNORECASE)
    groups: dict[str, list[tuple[int, Path]]] = {}

    for p in sorted(folder.rglob("E_camera_*_frame_*.tif")):
        m = frame_re.match(p.name)
        if not m:
            continue
        prefix = m.group(1)
        frame = int(m.group(2))
        groups.setdefault(prefix, []).append((frame, p))

    for prefix, seq in groups.items():
        seq.sort(key=lambda x: x[0])
        for i in range(len(seq) - 1):
            f0, p0 = seq[i]
            f1, p1 = seq[i + 1]
            if f1 != f0 + 1:
                continue
            label = f"{prefix}{f0:05d}_{f1:05d}".replace(" ", "_")
            out.append(PairItem("PIV_E", label, p0, p1))

    return out


def _collect_piv_book(folder: Path) -> list[PairItem]:
    out: list[PairItem] = []
    for p1 in sorted(folder.rglob("*a.tif")):
        if not p1.name.lower().endswith("a.tif"):
            continue
        p2 = p1.with_name(p1.name[:-5] + "b.tif")
        if not p2.exists():
            continue
        out.append(PairItem("PIV_book", p1.stem[:-1], p1, p2))
    return out


def collect_all_pairs(data_root: Path, max_pairs_per_source: int | None = None) -> dict[str, list[PairItem]]:
    src_map: dict[str, list[PairItem]] = {"PIV_C": [], "PIV_E": [], "PIV_book": []}

    piv_c = _find_subdir(data_root, "PIV C")
    piv_e = _find_subdir(data_root, "PIV E")
    piv_book = _find_subdir(data_root, "PIV book")

    if piv_c is not None:
        src_map["PIV_C"] = _collect_piv_c(piv_c)
    if piv_e is not None:
        src_map["PIV_E"] = _collect_piv_e(piv_e)
    if piv_book is not None:
        src_map["PIV_book"] = _collect_piv_book(piv_book)

    if max_pairs_per_source is not None:
        for k in list(src_map.keys()):
            src_map[k] = src_map[k][:max_pairs_per_source]
    return src_map


def load_gray(path: Path, max_size: int = 512) -> np.ndarray:
    img = Image.open(path)
    if img.mode != "L":
        img = img.convert("L")
    arr = np.asarray(img, dtype=float) / 255.0

    h, w = arr.shape
    if h > max_size or w > max_size:
        rr = min(h, max_size)
        cc = min(w, max_size)
        r0 = (h - rr) // 2
        c0 = (w - cc) // 2
        arr = arr[r0:r0 + rr, c0:c0 + cc]
    return arr


def abs_corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    xv = x[valid]
    yv = y[valid]
    if xv.size <= 1:
        return float("nan")
    if np.nanstd(xv) <= 1e-12 or np.nanstd(yv) <= 1e-12:
        return float("nan")
    return float(abs(np.corrcoef(xv, yv)[0, 1]))


def rmse(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    d = x[valid] - y[valid]
    if d.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(d * d)))


def mae(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    d = np.abs(x[valid] - y[valid])
    if d.size == 0:
        return float("nan")
    return float(np.mean(d))


def p99_abs(a: np.ndarray, floor: float = 0.05) -> float:
    v = np.asarray(a, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return floor
    return max(float(np.percentile(np.abs(v), 99.0)), floor)


def run_pair(
    i1: np.ndarray,
    i2: np.ndarray,
    gate_percentile: float,
    progress: Callable[[str], None] | None = None,
    heartbeat_rows: int = 5,
) -> dict:
    def _emit(msg: str) -> None:
        if progress is not None:
            progress(msg)

    _emit("stage=widim start")
    wdm = widim_fullfield_base(
        i1,
        i2,
        subset_size=WS,
        step=STEP,
        search_radius=SEARCH_RADIUS,
        search_step=SEARCH_STEP,
        n_iter=3,
    )

    centers = wdm["centers"]
    n = int(wdm["n"])
    u = np.asarray(wdm["u_wdm"], dtype=float)
    v = np.asarray(wdm["v_wdm"], dtype=float)
    _emit(f"stage=widim done grid={n}x{n}")

    du_dx, du_dy, dv_dx, dv_dy = _compute_gradients(u, v, STEP)
    omega_w = np.degrees(0.5 * (dv_dx - du_dy))

    h, w_img = i1.shape
    half = WS // 2
    alpha_raw = np.full((n, n), np.nan, dtype=float)
    snr_map = np.full((n, n), np.nan, dtype=float)

    _emit("stage=nufft start")
    for ri, cr in enumerate(centers):
        for ci, cc in enumerate(centers):
            if cr - half < 0 or cr + half > h or cc - half < 0 or cc + half > w_img:
                continue

            w1 = i1[cr - half: cr - half + WS, cc - half: cc - half + WS]
            w2_t = nc._shift_window_translation(i2, cr, cc, half, h, w_img, u[ri, ci], v[ri, ci])

            with contextlib.redirect_stdout(_io.StringIO()):
                res = estimate_rotation(w1, w2_t, OPT_R_MIN, OPT_WEIGHT, OPT_POC, verbose=False)

            alpha_raw[ri, ci] = -float(res["angle_est"])
            snr_map[ri, ci] = float(res["snr"])

        if heartbeat_rows <= 0:
            heartbeat_rows = 5
        if ((ri + 1) % heartbeat_rows == 0) or (ri == n - 1):
            pct = 100.0 * float(ri + 1) / float(max(n, 1))
            _emit(f"stage=nufft rows {ri + 1}/{n} ({pct:.1f}%)")

    _emit("stage=gate_refill start")
    alpha_new = nc._two_pass_gate_refill(alpha_raw, snr_map)

    finite_snr = np.isfinite(snr_map)
    snr_thr = float(np.percentile(snr_map[finite_snr], gate_percentile)) if np.any(finite_snr) else float("nan")
    gate = finite_snr & (snr_map >= snr_thr)

    corr_raw_gate = abs_corr(alpha_raw, omega_w, gate)
    corr_new_gate = abs_corr(alpha_new, omega_w, gate)
    rmse_raw_gate = rmse(alpha_raw, omega_w, gate)
    rmse_new_gate = rmse(alpha_new, omega_w, gate)
    mae_raw_gate = mae(alpha_raw, omega_w, gate)
    mae_new_gate = mae(alpha_new, omega_w, gate)

    delta_corr_gate = (
        corr_new_gate - corr_raw_gate
        if np.isfinite(corr_new_gate) and np.isfinite(corr_raw_gate)
        else float("nan")
    )
    delta_rmse_gate = (
        rmse_raw_gate - rmse_new_gate
        if np.isfinite(rmse_raw_gate) and np.isfinite(rmse_new_gate)
        else float("nan")
    )
    delta_mae_gate = (
        mae_raw_gate - mae_new_gate
        if np.isfinite(mae_raw_gate) and np.isfinite(mae_new_gate)
        else float("nan")
    )

    valid_windows = np.isfinite(alpha_raw) & np.isfinite(alpha_new) & np.isfinite(omega_w)
    n_windows = int(np.sum(valid_windows))
    n_pass = int(np.sum(valid_windows & gate))
    pass_rate = float(100.0 * n_pass / max(n_windows, 1))
    _emit("stage=metrics done")

    return {
        "centers": centers,
        "omega_w": omega_w,
        "alpha_raw": alpha_raw,
        "alpha_new": alpha_new,
        "snr": snr_map,
        "gate": gate,
        "snr_thr": snr_thr,
        "snr_mean": float(np.nanmean(snr_map)),
        "corr_raw_gate": corr_raw_gate,
        "corr_new_gate": corr_new_gate,
        "delta_corr_gate": delta_corr_gate,
        "rmse_raw_gate": rmse_raw_gate,
        "rmse_new_gate": rmse_new_gate,
        "delta_rmse_gate": delta_rmse_gate,
        "mae_raw_gate": mae_raw_gate,
        "mae_new_gate": mae_new_gate,
        "delta_mae_gate": delta_mae_gate,
        "n_windows": n_windows,
        "n_pass": n_pass,
        "pass_rate": pass_rate,
    }


def plot_pair(i1: np.ndarray, i2: np.ndarray, r: dict, title_name: str, out_path: Path) -> None:
    omega_w = r["omega_w"]
    alpha_raw = r["alpha_raw"]
    alpha_new = r["alpha_new"]
    snr_map = r["snr"]
    gate = r["gate"]

    err_raw = np.where(gate, alpha_raw - omega_w, np.nan)
    err_new = np.where(gate, alpha_new - omega_w, np.nan)
    delta_abs = np.abs(err_new) - np.abs(err_raw)

    vmax_v = p99_abs(np.concatenate([omega_w.reshape(-1), alpha_new.reshape(-1)]), floor=1.0)
    vmax_e = p99_abs(np.concatenate([err_raw.reshape(-1), err_new.reshape(-1)]), floor=0.1)
    vmax_d = p99_abs(delta_abs, floor=0.05)

    fig, axes = plt.subplots(3, 3, figsize=(16, 14))

    axes[0, 0].imshow(i1, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("I1")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(i2, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("I2")
    axes[0, 1].axis("off")

    im_snr = axes[0, 2].imshow(snr_map, cmap="viridis")
    y_bad, x_bad = np.where(~gate)
    if x_bad.size > 0:
        axes[0, 2].plot(x_bad, y_bad, "r.", ms=2.0, alpha=0.7)
    axes[0, 2].set_title(f"SNR map (p-thr={_safe_short(r['snr_thr'])}, pass={r['pass_rate']:.1f}%)")
    plt.colorbar(im_snr, ax=axes[0, 2], shrink=0.82)

    im_w = axes[1, 0].imshow(omega_w, cmap="coolwarm", vmin=-vmax_v, vmax=vmax_v)
    axes[1, 0].set_title("WIDIM vorticity")
    plt.colorbar(im_w, ax=axes[1, 0], shrink=0.82)

    im_ar = axes[1, 1].imshow(alpha_raw, cmap="coolwarm", vmin=-vmax_v, vmax=vmax_v)
    axes[1, 1].set_title("NUFFT vorticity (raw)")
    plt.colorbar(im_ar, ax=axes[1, 1], shrink=0.82)

    im_an = axes[1, 2].imshow(alpha_new, cmap="coolwarm", vmin=-vmax_v, vmax=vmax_v)
    axes[1, 2].set_title("NUFFT vorticity (new)")
    plt.colorbar(im_an, ax=axes[1, 2], shrink=0.82)

    im_er = axes[2, 0].imshow(err_raw, cmap="bwr", vmin=-vmax_e, vmax=vmax_e)
    axes[2, 0].set_title("Gated error raw")
    plt.colorbar(im_er, ax=axes[2, 0], shrink=0.82)

    im_en = axes[2, 1].imshow(err_new, cmap="bwr", vmin=-vmax_e, vmax=vmax_e)
    axes[2, 1].set_title("Gated error new")
    plt.colorbar(im_en, ax=axes[2, 1], shrink=0.82)

    im_de = axes[2, 2].imshow(delta_abs, cmap="bwr", vmin=-vmax_d, vmax=vmax_d)
    axes[2, 2].set_title("|err_new|-|err_raw| (negative better)")
    plt.colorbar(im_de, ax=axes[2, 2], shrink=0.82)

    for ax in axes.flat:
        if ax not in (axes[0, 0], axes[0, 1]):
            ax.set_xlabel("grid-x")
            ax.set_ylabel("grid-y")

    fig.suptitle(
        f"{title_name} | Corr_gate: {_safe_short(r['corr_raw_gate'])}->{_safe_short(r['corr_new_gate'])} "
        f"(d={_safe_short(r['delta_corr_gate'])}) | RMSE_gate: {_safe_short(r['rmse_raw_gate'])}"
        f"->{_safe_short(r['rmse_new_gate'])} (d={_safe_short(r['delta_rmse_gate'])})",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _mean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr))


def build_source_summary(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for r in rows:
        grouped.setdefault(r["source"], []).append(r)

    out: list[dict] = []
    for src in sorted(grouped.keys()):
        rr = grouped[src]
        out.append(
            {
                "source": src,
                "n_pairs": len(rr),
                "mean_delta_corr_gate": _mean_or_nan([x["delta_corr_gate"] for x in rr]),
                "mean_delta_rmse_gate": _mean_or_nan([x["delta_rmse_gate"] for x in rr]),
                "mean_delta_mae_gate": _mean_or_nan([x["delta_mae_gate"] for x in rr]),
                "mean_pass_rate": _mean_or_nan([x["pass_rate"] for x in rr]),
                "mean_snr_thr": _mean_or_nan([x["snr_thr_p"] for x in rr]),
                "mean_time_s": _mean_or_nan([x["time_s"] for x in rr]),
                "pos_delta_corr": int(np.sum(np.asarray([x["delta_corr_gate"] for x in rr], dtype=float) > 0)),
                "pos_delta_rmse": int(np.sum(np.asarray([x["delta_rmse_gate"] for x in rr], dtype=float) > 0)),
            }
        )
    return out


def save_csv_all(rows: list[dict], out_csv: Path) -> None:
    fields = [
        "source",
        "label",
        "pair",
        "corr_raw_gate",
        "corr_new_gate",
        "delta_corr_gate",
        "rmse_raw_gate",
        "rmse_new_gate",
        "delta_rmse_gate",
        "mae_raw_gate",
        "mae_new_gate",
        "delta_mae_gate",
        "snr_thr_p",
        "snr_mean",
        "pass_rate",
        "n_pass",
        "n_windows",
        "time_s",
        "fig",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def save_csv_summary(rows: list[dict], out_csv: Path) -> None:
    fields = [
        "source",
        "n_pairs",
        "mean_delta_corr_gate",
        "mean_delta_rmse_gate",
        "mean_delta_mae_gate",
        "mean_pass_rate",
        "mean_snr_thr",
        "mean_time_s",
        "pos_delta_corr",
        "pos_delta_rmse",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_summary_by_source(summary_rows: list[dict], out_png: Path) -> None:
    if not summary_rows:
        return
    labels = [r["source"] for r in summary_rows]
    d_corr = np.asarray([r["mean_delta_corr_gate"] for r in summary_rows], dtype=float)
    d_rmse = np.asarray([r["mean_delta_rmse_gate"] for r in summary_rows], dtype=float)
    pass_rate = np.asarray([r["mean_pass_rate"] for r in summary_rows], dtype=float)

    x = np.arange(len(labels))
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].bar(x - w / 2, d_corr, width=w, color="#65BDBA", edgecolor="k", label="mean dCorr")
    axes[0].bar(x + w / 2, d_rmse, width=w, color="#E9687A", edgecolor="k", label="mean dRMSE")
    axes[0].axhline(0.0, color="k", lw=1, alpha=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_title("Mean improvements by source")
    axes[0].set_ylabel("dMetric (dRMSE > 0 means better)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(x, pass_rate, width=0.45, color="#3C9BC9", edgecolor="k")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0, max(100.0, float(np.nanmax(pass_rate) * 1.1)))
    axes[1].set_title("Mean gate pass rate by source")
    axes[1].set_ylabel("pass rate (%)")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="", help="Folder containing PIV C / PIV E / PIV book")
    parser.add_argument("--out-dir", type=str, default="", help="Output root directory")
    parser.add_argument("--max-size", type=int, default=512, help="Center crop max size for loading")
    parser.add_argument("--gate-percentile", type=float, default=DEFAULT_GATE_PERCENTILE, help="Adaptive SNR percentile")
    parser.add_argument("--max-pairs-per-source", type=int, default=None, help="Debug limit per source")
    parser.add_argument("--heartbeat-rows", type=int, default=5, help="NUFFT row heartbeat interval inside one pair")
    parser.add_argument("--no-resume", action="store_true", help="Disable cache resume")
    parser.add_argument("--no-fig", action="store_true", help="Skip per-pair figure generation")
    args = parser.parse_args()

    t0 = time.perf_counter()
    repo_root = Path(__file__).resolve().parent
    data_root = _find_data_root(repo_root, args.data_root)
    out_root = Path(args.out_dir).resolve() if args.out_dir else (repo_root / "results_vortex_all" / "real_vortex_all_p5")
    out_root.mkdir(parents=True, exist_ok=True)

    src_pairs = collect_all_pairs(data_root, args.max_pairs_per_source)
    all_pairs: list[PairItem] = []
    for key in ("PIV_C", "PIV_E", "PIV_book"):
        all_pairs.extend(src_pairs.get(key, []))

    if not all_pairs:
        raise RuntimeError(f"No valid pairs found under data root: {data_root}")

    print("=" * 110, flush=True)
    print("Real vortex full-batch runner: WIDIM-track + NUFFT + new gate/refill + adaptive SNR percentile gate", flush=True)
    print(f"repo_root={repo_root}", flush=True)
    print(f"data_root={data_root}", flush=True)
    print(f"out_root={out_root}", flush=True)
    print(f"gate_percentile={args.gate_percentile}", flush=True)
    print(f"WS={WS}, STEP={STEP}, SEARCH_RADIUS={SEARCH_RADIUS}, SEARCH_STEP={SEARCH_STEP}", flush=True)
    print(
        f"resume={'off' if args.no_resume else 'on'} | save_fig={'off' if args.no_fig else 'on'} | "
        f"heartbeat_rows={args.heartbeat_rows}",
        flush=True,
    )
    print(
        f"pairs_total={len(all_pairs)} | PIV_C={len(src_pairs['PIV_C'])} | PIV_E={len(src_pairs['PIV_E'])} | "
        f"PIV_book={len(src_pairs['PIV_book'])}",
        flush=True,
    )
    print("=" * 110, flush=True)

    rows: list[dict] = []
    total = len(all_pairs)
    done = 0

    for src_name in ("PIV_C", "PIV_E", "PIV_book"):
        pairs = src_pairs.get(src_name, [])
        if not pairs:
            continue
        src_out = out_root / src_name
        src_out.mkdir(parents=True, exist_ok=True)
        print(f"\n[Source] {src_name} | pairs={len(pairs)} | out={src_out}", flush=True)

        for k, pair in enumerate(pairs, 1):
            pair_slug = _slug(pair.label)
            json_path = src_out / f"{pair_slug}_metrics.json"
            fig_path = src_out / f"{pair_slug}_result.png"

            row: dict
            pair_t0 = time.perf_counter()
            print(
                f"  -> start [{done + 1:03d}/{total:03d}] {src_name} {k:03d}/{len(pairs):03d} "
                f"{pair.p1.name}+{pair.p2.name}",
                flush=True,
            )

            if (not args.no_resume) and json_path.exists():
                row = _read_json(json_path)
                row["cached"] = True
                if "fig" not in row:
                    row["fig"] = fig_path.name if fig_path.exists() else ""
                print("     stage=resume cache hit", flush=True)
            else:
                i1 = load_gray(pair.p1, max_size=args.max_size)
                i2 = load_gray(pair.p2, max_size=args.max_size)

                def _pair_progress(msg: str, _src=src_name, _label=pair.label) -> None:
                    print(f"     {_src}:{_label} | {msg}", flush=True)

                r = run_pair(
                    i1,
                    i2,
                    gate_percentile=args.gate_percentile,
                    progress=_pair_progress,
                    heartbeat_rows=args.heartbeat_rows,
                )
                if not args.no_fig:
                    print("     stage=plot start", flush=True)
                    plot_pair(i1, i2, r, f"{pair.source}:{pair.label}", fig_path)
                    print("     stage=plot done", flush=True)

                dt = time.perf_counter() - pair_t0
                row = {
                    "source": pair.source,
                    "label": pair.label,
                    "pair": f"{pair.p1.name}+{pair.p2.name}",
                    "p1": str(pair.p1),
                    "p2": str(pair.p2),
                    "corr_raw_gate": float(r["corr_raw_gate"]),
                    "corr_new_gate": float(r["corr_new_gate"]),
                    "delta_corr_gate": float(r["delta_corr_gate"]),
                    "rmse_raw_gate": float(r["rmse_raw_gate"]),
                    "rmse_new_gate": float(r["rmse_new_gate"]),
                    "delta_rmse_gate": float(r["delta_rmse_gate"]),
                    "mae_raw_gate": float(r["mae_raw_gate"]),
                    "mae_new_gate": float(r["mae_new_gate"]),
                    "delta_mae_gate": float(r["delta_mae_gate"]),
                    "snr_thr_p": float(r["snr_thr"]),
                    "snr_mean": float(r["snr_mean"]),
                    "pass_rate": float(r["pass_rate"]),
                    "n_pass": int(r["n_pass"]),
                    "n_windows": int(r["n_windows"]),
                    "time_s": float(dt),
                    "fig": fig_path.name if (fig_path.exists() and (not args.no_fig)) else "",
                    "cached": False,
                }
                _write_json(json_path, row)

            rows.append(row)
            done += 1
            elapsed = time.perf_counter() - t0
            eta = (elapsed / max(done, 1)) * (total - done)

            print(
                f"[{done:03d}/{total:03d}] {src_name} {k:03d}/{len(pairs):03d} "
                f"{pair.p1.name}+{pair.p2.name} | "
                f"dCorr={_safe_short(row.get('delta_corr_gate'))} "
                f"dRMSE={_safe_short(row.get('delta_rmse_gate'))} "
                f"pass={_safe_short(row.get('pass_rate'))}% "
                f"p={_safe_short(row.get('snr_thr_p'))} "
                f"ETA={_eta_str(eta)} "
                f"{'(cached)' if row.get('cached') else ''}"
                ,
                flush=True,
            )

    summary_rows = build_source_summary(rows)
    save_csv_all(rows, out_root / "metrics_all_pairs.csv")
    save_csv_summary(summary_rows, out_root / "metrics_by_source.csv")
    plot_summary_by_source(summary_rows, out_root / "summary_by_source.png")

    report_name = f"{date.today().year}.{date.today().month}.{date.today().day}.txt"
    report_path = repo_root / report_name
    with report_path.open("w", encoding="utf-8") as f:
        f.write("Real-vortex full-batch report\n")
        f.write("=" * 110 + "\n")
        f.write(f"date={date.today().isoformat()}\n")
        f.write(f"repo_root={repo_root}\n")
        f.write(f"data_root={data_root}\n")
        f.write(f"out_root={out_root}\n")
        f.write(f"gate_percentile={args.gate_percentile}\n")
        f.write(f"WS={WS}, STEP={STEP}, SEARCH_RADIUS={SEARCH_RADIUS}, SEARCH_STEP={SEARCH_STEP}\n")
        f.write(
            "pipeline=WIDIM displacement -> WIDIM vorticity -> translation-only window tracking -> "
            "NUFFT rotation -> two-pass gate/refill -> adaptive SNR percentile gate\n"
        )
        f.write("\n[source_summary]\n")
        for s in summary_rows:
            f.write(
                f"{s['source']} n={s['n_pairs']} "
                f"mean_dCorr={_safe(s['mean_delta_corr_gate'])} "
                f"mean_dRMSE={_safe(s['mean_delta_rmse_gate'])} "
                f"mean_dMAE={_safe(s['mean_delta_mae_gate'])} "
                f"mean_pass={_safe(s['mean_pass_rate'])}% "
                f"mean_p={_safe(s['mean_snr_thr'])} "
                f"pos_dCorr={s['pos_delta_corr']}/{s['n_pairs']} "
                f"pos_dRMSE={s['pos_delta_rmse']}/{s['n_pairs']}\n"
            )
        f.write("\n[pair_metrics]\n")
        for r in rows:
            f.write(
                f"{r['source']} {r['pair']} "
                f"dCorr={_safe(r['delta_corr_gate'])} "
                f"dRMSE={_safe(r['delta_rmse_gate'])} "
                f"dMAE={_safe(r['delta_mae_gate'])} "
                f"pass={_safe(r['pass_rate'])}% "
                f"p={_safe(r['snr_thr_p'])} "
                f"N={r['n_pass']}/{r['n_windows']} "
                f"time={_safe(r.get('time_s'))} "
                f"cached={r.get('cached', False)} "
                f"fig={r.get('fig', '')}\n"
            )
        f.write("\n[files]\n")
        f.write(f"pairs_csv={out_root / 'metrics_all_pairs.csv'}\n")
        f.write(f"summary_csv={out_root / 'metrics_by_source.csv'}\n")
        f.write(f"summary_png={out_root / 'summary_by_source.png'}\n")

    total_dt = time.perf_counter() - t0
    print("\n" + "=" * 110, flush=True)
    print("Finished full-batch script.", flush=True)
    print(f"elapsed={total_dt:.1f}s", flush=True)
    print(f"report={report_path}", flush=True)
    print(f"pairs_csv={out_root / 'metrics_all_pairs.csv'}", flush=True)
    print(f"summary_csv={out_root / 'metrics_by_source.csv'}", flush=True)
    print(f"summary_png={out_root / 'summary_by_source.png'}", flush=True)
    print("=" * 110, flush=True)


if __name__ == "__main__":
    main()
