from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = REPO_ROOT / "results_bridge_truth"
FIGS_DIR = RESULTS_ROOT / "figs"
METRICS_DIR = RESULTS_ROOT / "metrics"
CASES_DIR = RESULTS_ROOT / "cases"
LOGS_DIR = RESULTS_ROOT / "logs"


@dataclass(frozen=True)
class LevelSpec:
    name: str
    label: str
    p_drop: float
    rho_var: float
    a_illum: float
    sigma_bg: float
    blur_sigma: float
    flow_perturb: float


LEVELS: tuple[LevelSpec, ...] = (
    LevelSpec("L0", "L0 ideal", 0.00, 0.00, 0.00, 0.005, 0.00, 0.00),
    LevelSpec("L1", "L1 turnover", 0.10, 0.15, 0.10, 0.010, 0.30, 0.00),
    LevelSpec("L2", "L2 imaging", 0.20, 0.25, 0.20, 0.020, 0.55, 0.03),
    LevelSpec("L3", "L3 near-real", 0.30, 0.35, 0.30, 0.030, 0.85, 0.08),
)


FLOWS: tuple[str, ...] = (
    "rankine",
    "lamb_oseen",
    "solid_rotation",
    "mixed_vortex",
)


PILOT_SEEDS = 8
FULL_SEEDS = 30


PRIMARY_GATE_PERCENTILE = 5.0
PRIMARY_STEP = 10


SAMPLE_CASES = (
    ("rankine", "L3", 0),
    ("lamb_oseen", "L3", 0),
    ("solid_rotation", "L3", 0),
    ("mixed_vortex", "L3", 0),
)
