from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bridge_config import CASES_DIR, FIGS_DIR, FLOWS, FULL_SEEDS, LEVELS, LOGS_DIR, METRICS_DIR, PILOT_SEEDS, RESULTS_ROOT, SAMPLE_CASES
from bridge_flow_truth import make_truth_case
from bridge_metrics import BridgeEval, evaluate_bridge_case
from bridge_renderer import render_case
import build_bridge_figs


def _ensure_dirs() -> None:
    for path in (RESULTS_ROOT, METRICS_DIR, FIGS_DIR, CASES_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _record_from_eval(flow_name: str, level_name: str, seed: int, flow_label: str, eval_r: BridgeEval) -> dict:
    return {
        "flow_name": flow_name,
        "flow_label": flow_label,
        "level_name": level_name,
        "seed": seed,
        "rmse_true_widim": eval_r.rmse_true_widim,
        "rmse_true_raw": eval_r.rmse_true_raw,
        "rmse_true_new": eval_r.rmse_true_new,
        "mae_true_widim": eval_r.mae_true_widim,
        "mae_true_raw": eval_r.mae_true_raw,
        "mae_true_new": eval_r.mae_true_new,
        "corr_true_widim": eval_r.corr_true_widim,
        "corr_true_raw": eval_r.corr_true_raw,
        "corr_true_new": eval_r.corr_true_new,
        "delta_rmse_true": eval_r.delta_rmse_true,
        "delta_rmse_true_bridge": eval_r.delta_rmse_true_bridge,
        "delta_mae_true": eval_r.delta_mae_true,
        "delta_mae_true_bridge": eval_r.delta_mae_true_bridge,
        "delta_corr_true": eval_r.delta_corr_true,
        "delta_corr_true_bridge": eval_r.delta_corr_true_bridge,
        "corr_raw_gate": eval_r.corr_raw_gate,
        "corr_new_gate": eval_r.corr_new_gate,
        "rmse_raw_gate": eval_r.rmse_raw_gate,
        "rmse_new_gate": eval_r.rmse_new_gate,
        "mae_raw_gate": eval_r.mae_raw_gate,
        "mae_new_gate": eval_r.mae_new_gate,
        "delta_rmse_g": eval_r.delta_rmse_g,
        "delta_mae_g": eval_r.delta_mae_g,
        "delta_corr_g": eval_r.delta_corr_g,
        "n_eval": eval_r.n_eval,
        "pass_rate": eval_r.pass_rate,
        "snr_thr": eval_r.snr_thr,
    }


def _save_sample_case(flow_name: str, level_name: str, seed: int, flow_label: str, rendered, eval_r: BridgeEval) -> None:
    tag = f"{flow_name}_{level_name}_seed{seed}_sample.npz"
    out_path = CASES_DIR / tag
    np.savez_compressed(
        out_path,
        flow_name=flow_name,
        flow_label=flow_label,
        level_name=level_name,
        seed=seed,
        i1=rendered.i1,
        i2=rendered.i2,
        i1_clean=rendered.i1_clean,
        i2_clean=rendered.i2_clean,
        omega_truth_grid=eval_r.omega_truth_grid,
        omega_widim=eval_r.omega_widim,
        alpha_raw=eval_r.alpha_raw,
        alpha_new=eval_r.alpha_new,
        snr_map=eval_r.snr_map,
        gate=eval_r.gate,
    )


def _sign_agreement(a: pd.Series, b: pd.Series) -> float:
    x = a.to_numpy(dtype=float)
    y = b.to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(valid)) == 0:
        return float("nan")
    return float(np.mean(np.sign(x[valid]) == np.sign(y[valid])))


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    x = a.to_numpy(dtype=float)
    y = b.to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(valid)) < 3:
        return float("nan")
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def _write_summary(df: pd.DataFrame, out_txt: Path, seeds: int) -> dict:
    grouped = (
        df.groupby(["flow_name", "level_name"], as_index=False)[
            [
                "rmse_true_widim",
                "rmse_true_raw",
                "rmse_true_new",
                "delta_rmse_true",
                "delta_rmse_true_bridge",
                "delta_rmse_g",
                "delta_mae_g",
                "delta_corr_g",
                "pass_rate",
            ]
        ]
        .mean()
    )

    corr_rmse = _safe_corr(df["delta_rmse_g"], df["delta_rmse_true"])
    corr_mae = _safe_corr(df["delta_mae_g"], df["delta_rmse_true"])
    corr_corr = _safe_corr(df["delta_corr_g"], df["delta_rmse_true"])
    sign_rmse = _sign_agreement(df["delta_rmse_g"], df["delta_rmse_true"])
    sign_mae = _sign_agreement(df["delta_mae_g"], df["delta_rmse_true"])
    sign_corr = _sign_agreement(df["delta_corr_g"], df["delta_rmse_true"])

    corr_rmse_raw = _safe_corr(df["delta_rmse_g"], df["delta_rmse_true_bridge"])
    corr_mae_raw = _safe_corr(df["delta_mae_g"], df["delta_rmse_true_bridge"])
    corr_corr_raw = _safe_corr(df["delta_corr_g"], df["delta_rmse_true_bridge"])

    success = bool(
        np.nanmean(df["delta_rmse_true"]) > 0
        and float(np.mean(grouped["delta_rmse_true"] > 0)) >= 0.66
    )

    lines = []
    lines.append("近真实真值桥接实验记录")
    lines.append("=" * 90)
    lines.append(f"cases={len(df)} | flows={len(FLOWS)} | levels={len(LEVELS)} | seeds={seeds}")
    lines.append("")
    lines.append("[整体桥接指标]")
    lines.append("主真值目标：WIDIM -> final")
    mean_widim = float(df["rmse_true_widim"].mean())
    mean_final = float(df["rmse_true_new"].mean())
    mean_gain = float(df["delta_rmse_true"].mean())
    rel_drop = 100.0 * mean_gain / max(mean_widim, 1e-9)
    case_pos = 100.0 * float(np.mean(df["delta_rmse_true"] > 0))
    group_pos = 100.0 * float(np.mean(grouped["delta_rmse_true"] > 0))
    lines.append(f"overall mean RMSE_true: WIDIM {mean_widim:.3f} -> final {mean_final:.3f}")
    lines.append(f"overall absolute gain ΔRMSE_true = {mean_gain:.3f}")
    lines.append(f"overall relative reduction = {rel_drop:.1f}%")
    lines.append(f"case-wise positive ratio = {case_pos:.1f}%")
    lines.append(f"flow-level positive ratio = {group_pos:.1f}%")
    lines.append("")
    lines.append("说明：当前 gated proxy 指标并不直接标定 WIDIM->final 的真值差，")
    lines.append("因此这些 proxy 的相关性不作为主桥接成功判据。")
    lines.append("")
    lines.append("辅助真值目标：raw direct estimator -> final（仅用于解释当前 proxy 指标）")
    lines.append(f"corr(ΔRMSE_g, ΔRMSE_true_bridge) = {corr_rmse_raw:.4f}")
    lines.append(f"corr(ΔMAE_g,  ΔRMSE_true_bridge) = {corr_mae_raw:.4f}")
    lines.append(f"corr(ΔCorr_g, ΔRMSE_true_bridge) = {corr_corr_raw:.4f}")
    lines.append("")
    lines.append("[按流场与 level 汇总]")
    for _, row in grouped.iterrows():
        lines.append(
            f"{row['flow_name']:>14s} | {row['level_name']} | "
            f"RMSE_true WIDIM {row['rmse_true_widim']:.3f}->{row['rmse_true_new']:.3f} "
            f"(Δ={row['delta_rmse_true']:.3f}) | "
            f"raw->new {row['rmse_true_raw']:.3f}->{row['rmse_true_new']:.3f} "
            f"(Δ={row['delta_rmse_true_bridge']:.3f}) | "
            f"ΔRMSE_g={row['delta_rmse_g']:.3f} | "
            f"ΔMAE_g={row['delta_mae_g']:.3f} | "
            f"ΔCorr_g={row['delta_corr_g']:.4f} | "
            f"pass={row['pass_rate']:.1f}%"
        )
    lines.append("")
    lines.append("[结论判断]")
    if success:
        lines.append("bridge_status=SUCCESS")
        lines.append("判定理由：WIDIM->final 的 truth gain 在总体与大多数 flow-level 组合上为正。")
        lines.append("建议：可将该桥接实验写入正文，放在多流场真值对比之后、真实数据之前。")
    else:
        lines.append("bridge_status=NOT_READY")
        lines.append("判定理由：尽管 WIDIM->final 的 truth gain 可能总体为正，但桥接关系和稳定性仍需更多 seeds 或更稳的 realism 设计。")
        lines.append("建议：继续调 realism knob 或增加 seeds 后再决定是否进入正文。")

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")

    return {
        "success": success,
        "corr_rmse": corr_rmse,
        "corr_mae": corr_mae,
        "corr_corr": corr_corr,
        "corr_rmse_raw": corr_rmse_raw,
        "corr_mae_raw": corr_mae_raw,
        "corr_corr_raw": corr_corr_raw,
        "sign_rmse": sign_rmse,
        "sign_mae": sign_mae,
        "sign_corr": sign_corr,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=PILOT_SEEDS)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    seeds = FULL_SEEDS if args.full else int(args.n_seeds)
    _ensure_dirs()

    rows: list[dict] = []
    total = len(FLOWS) * len(LEVELS) * seeds
    case_idx = 0

    print("=" * 110, flush=True)
    print("Near-real truth bridge benchmark", flush=True)
    print(f"repo_root={RESULTS_ROOT.parent}", flush=True)
    print(f"results_root={RESULTS_ROOT}", flush=True)
    print(f"flows={list(FLOWS)} | levels={[l.name for l in LEVELS]} | seeds={seeds}", flush=True)
    print("=" * 110, flush=True)

    for flow_name in FLOWS:
        for level in LEVELS:
            for seed in range(seeds):
                case_idx += 1
                print(
                    f"[{case_idx:03d}/{total:03d}] flow={flow_name} level={level.name} seed={seed}",
                    flush=True,
                )
                truth = make_truth_case(flow_name, level, seed)
                rendered = render_case(truth, level, seed)
                eval_r = evaluate_bridge_case(rendered.i1, rendered.i2, truth.omega_deg)
                rows.append(_record_from_eval(flow_name, level.name, seed, truth.flow_label, eval_r))

                if (flow_name, level.name, seed) in SAMPLE_CASES:
                    _save_sample_case(flow_name, level.name, seed, truth.flow_label, rendered, eval_r)

    df = pd.DataFrame(rows)
    all_csv = METRICS_DIR / "all_cases.csv"
    df.to_csv(all_csv, index=False, encoding="utf-8-sig")

    summary = _write_summary(df, RESULTS_ROOT.parent / "桥接实验.txt", seeds)
    (METRICS_DIR / "bridge_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    build_bridge_figs.main()
    print("bridge benchmark finished", flush=True)
    print(f"all_cases={all_csv}", flush=True)
    print(f"summary_txt={RESULTS_ROOT.parent / '桥接实验.txt'}", flush=True)


if __name__ == "__main__":
    main()
