# -*- coding: utf-8 -*-
"""One-command reproduction entry point for the journal evidence package."""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
PROJECT_ROOT = str(Path(HERE).resolve().parent)
EXTENDED_RESULTS = os.path.join(RESULTS, "tmc_extended")
DERIVED_RESULTS = os.path.join(EXTENDED_RESULTS, "derived")
FIGURE_SCRIPT = os.path.join(
    PROJECT_ROOT,
    "analysis",
    "figure_generation_tmc_advanced.py",
)
FIGURE_MODULE_NAMES = (
    "figure_generation_tmc_advanced.py",
    "figure_panels_v3.py",
    "figure_baseline_v3.py",
    "figure_service_v3.py",
)
FIGURE_MODULES = tuple(
    os.path.join(PROJECT_ROOT, "analysis", name)
    for name in FIGURE_MODULE_NAMES
)
CONCEPT_FIGURE_SCRIPT = os.path.join(
    PROJECT_ROOT,
    "analysis",
    "figure_generation.py",
)
CONCEPT_FIGURE_SOURCE_DIR = os.path.join(
    PROJECT_ROOT,
    "analysis",
    "freshsky_tmc_figures",
)
FIGURE_DIR = os.path.join(
    PROJECT_ROOT,
    "figures",
)
MEASURED_INPUT = os.path.join(
    HERE,
    "data",
    "measured_a2g_urban_3p6ghz.xlsx",
)
SCRIPTS = (
    "run_tmc_evidence.py",
    "belief_whittle.py",
    "bayes_hmm_experiment.py",
    "bayes_hmm_delayed_ack.py",
    "bayes_hmm_delayed_aoi_belief.py",
    "bayes_hmm_grid_sensitivity.py",
    "measured_a2g_replay.py",
    "tmc_calibration_strict.py",
    "tmc_measured_strict.py",
    "tmc_runtime_scaling.py",
    "tmc_extended_evidence.py",
    "test_zhao25_specialization.py",
    "zhao25_baseline_fidelity.md",
)
PRIMARY_OUTPUTS = (
    "tmc_aoi_calibration_seeds.csv",
    "tmc_aoi_calibration_summary.csv",
    "tmc_aoi_calibration_paired.csv",
    "measured_a2g_delayed_aoi_seeds.csv",
    "measured_a2g_delayed_aoi_summary.csv",
    "measured_a2g_delayed_aoi_paired.csv",
    "tmc_runtime_scaling.csv",
    "tmc_extended/tmc_extended_baselines_seeds.csv",
    "tmc_extended/tmc_extended_baselines_summary.csv",
    "tmc_extended/tmc_extended_baselines_paired.csv",
    "tmc_extended/tmc_factorial_ablation_seeds.csv",
    "tmc_extended/tmc_factorial_ablation_summary.csv",
    "tmc_extended/tmc_qcap_service_seeds.csv",
    "tmc_extended/tmc_qcap_service_summary.csv",
    "tmc_extended/tmc_delayed_shift_seeds.csv",
    "tmc_extended/tmc_delayed_shift_summary.csv",
)
DERIVED_OUTPUTS = tuple(
    f"tmc_extended/derived/{name}"
    for name in (
        "tmc_advanced_baseline_effects.csv",
        "tmc_advanced_baseline_pareto.csv",
        "tmc_advanced_factorial_contrasts_seeds.csv",
        "tmc_advanced_factorial_contrasts_summary.csv",
        "tmc_advanced_measured_effects_seeds.csv",
        "tmc_advanced_qcap_summary.csv",
        "tmc_advanced_rank_stability.csv",
        "tmc_advanced_runtime_focus.csv",
        "tmc_advanced_shield_tradeoff.csv",
        "tmc_advanced_shift_summary.csv",
        "tmc_extended_uav_long.csv",
        "tmc_v3_calibration_effects_seeds.csv",
        "tmc_v3_calibration_effects_summary.csv",
        "tmc_v3_factorial_multimetric_effects_seeds.csv",
        "tmc_v3_factorial_multimetric_effects_summary.csv",
        "tmc_v3_measured_multimetric_effects_seeds.csv",
        "tmc_v3_measured_multimetric_effects_summary.csv",
        "tmc_v3_measured_relation_seeds.csv",
        "tmc_v3_qcap_anchor_summary.csv",
        "tmc_v3_qcap_pressure_seeds.csv",
        "tmc_v3_qcap_pressure_summary.csv",
        "tmc_v3_qcap_tail_cells.csv",
        "tmc_v3_runtime_grid.csv",
        "tmc_v3_shield_tradeoff_seeds.csv",
        "tmc_v3_shift_differences.csv",
    )
)
OUTPUTS = PRIMARY_OUTPUTS + DERIVED_OUTPUTS
EXPECTED_ROWS = {
    "tmc_aoi_calibration_seeds.csv": 360,
    "tmc_aoi_calibration_summary.csv": 36,
    "tmc_aoi_calibration_paired.csv": 12,
    "measured_a2g_delayed_aoi_seeds.csv": 360,
    "measured_a2g_delayed_aoi_summary.csv": 18,
    "measured_a2g_delayed_aoi_paired.csv": 6,
    "tmc_runtime_scaling.csv": 54,
    "tmc_extended/tmc_extended_baselines_seeds.csv": 660,
    "tmc_extended/tmc_extended_baselines_summary.csv": 33,
    "tmc_extended/tmc_extended_baselines_paired.csv": 30,
    "tmc_extended/tmc_factorial_ablation_seeds.csv": 160,
    "tmc_extended/tmc_factorial_ablation_summary.csv": 8,
    "tmc_extended/tmc_qcap_service_seeds.csv": 180,
    "tmc_extended/tmc_qcap_service_summary.csv": 9,
    "tmc_extended/tmc_delayed_shift_seeds.csv": 80,
    "tmc_extended/tmc_delayed_shift_summary.csv": 4,
    "tmc_extended/derived/tmc_advanced_baseline_effects.csv": 30,
    "tmc_extended/derived/tmc_advanced_baseline_pareto.csv": 8,
    "tmc_extended/derived/tmc_advanced_factorial_contrasts_seeds.csv": 140,
    "tmc_extended/derived/tmc_advanced_factorial_contrasts_summary.csv": 7,
    "tmc_extended/derived/tmc_advanced_measured_effects_seeds.csv": 120,
    "tmc_extended/derived/tmc_advanced_qcap_summary.csv": 9,
    "tmc_extended/derived/tmc_advanced_rank_stability.csv": 200,
    "tmc_extended/derived/tmc_advanced_runtime_focus.csv": 6,
    "tmc_extended/derived/tmc_advanced_shield_tradeoff.csv": 4,
    "tmc_extended/derived/tmc_advanced_shift_summary.csv": 4,
    "tmc_extended/derived/tmc_extended_uav_long.csv": 7920,
    "tmc_extended/derived/tmc_v3_calibration_effects_seeds.csv": 120,
    "tmc_extended/derived/tmc_v3_calibration_effects_summary.csv": 12,
    "tmc_extended/derived/tmc_v3_factorial_multimetric_effects_seeds.csv": 700,
    "tmc_extended/derived/tmc_v3_factorial_multimetric_effects_summary.csv": 35,
    "tmc_extended/derived/tmc_v3_measured_multimetric_effects_seeds.csv": 480,
    "tmc_extended/derived/tmc_v3_measured_multimetric_effects_summary.csv": 24,
    "tmc_extended/derived/tmc_v3_measured_relation_seeds.csv": 120,
    "tmc_extended/derived/tmc_v3_qcap_anchor_summary.csv": 9,
    "tmc_extended/derived/tmc_v3_qcap_pressure_seeds.csv": 180,
    "tmc_extended/derived/tmc_v3_qcap_pressure_summary.csv": 9,
    "tmc_extended/derived/tmc_v3_qcap_tail_cells.csv": 90,
    "tmc_extended/derived/tmc_v3_runtime_grid.csv": 54,
    "tmc_extended/derived/tmc_v3_shield_tradeoff_seeds.csv": 80,
    "tmc_extended/derived/tmc_v3_shift_differences.csv": 6,
}
ADVANCED_FIGURE_OUTPUTS = tuple(
    f"fig_tmc_{stem}.{suffix}"
    for stem in (
        "baseline_evidence",
        "factorial_effects",
        "service_pareto",
        "robustness",
        "runtime",
    )
    for suffix in ("pdf", "svg", "png")
)
CONCEPT_FIGURE_OUTPUTS = tuple(
    f"fig_{stem}.{suffix}"
    for stem in ("system", "mechanism")
    for suffix in ("pdf", "png")
)
FIGURE_OUTPUTS = ADVANCED_FIGURE_OUTPUTS + CONCEPT_FIGURE_OUTPUTS


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_artifacts():
    problems = []
    row_counts = {}
    if not os.path.exists(MEASURED_INPUT):
        problems.append(
            "missing measured input: "
            "experiments/data/measured_a2g_urban_3p6ghz.xlsx"
        )
    for name in SCRIPTS:
        if not os.path.exists(os.path.join(HERE, name)):
            problems.append(f"missing source: experiments/{name}")
    for name, path in zip(FIGURE_MODULE_NAMES, FIGURE_MODULES):
        if not os.path.exists(path):
            problems.append(
                f"missing source: analysis/{name}"
            )
    if not os.path.exists(CONCEPT_FIGURE_SCRIPT):
        problems.append(
            "missing source: analysis/figure_generation.py"
        )
    for name, expected in EXPECTED_ROWS.items():
        path = os.path.join(RESULTS, name)
        if not os.path.exists(path):
            problems.append(f"missing output: {name}")
            continue
        actual = len(pd.read_csv(path))
        row_counts[name] = actual
        if actual != expected:
            problems.append(
                f"row-count mismatch: {name}: "
                f"expected {expected}, found {actual}"
            )
    for name in FIGURE_OUTPUTS:
        path = os.path.join(FIGURE_DIR, name)
        if not os.path.exists(path):
            problems.append(f"missing figure: {name}")
        elif os.path.getsize(path) <= 0:
            problems.append(f"empty figure: {name}")
    if problems:
        raise RuntimeError(
            "evidence validation failed:\n- " + "\n- ".join(problems)
        )
    return row_counts


def write_manifest():
    row_counts = validate_artifacts()
    os.makedirs(RESULTS, exist_ok=True)
    manifest = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": (
            platform.processor()
            or os.environ.get("PROCESSOR_IDENTIFIER", "unknown")
        ),
        "logical_processors": os.cpu_count(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "protocol": {
            "feedback_timing": {
                "parameter": "d",
                "meaning": "number of pending-feedback slots",
                "ack_generated_in_slot_j":
                    "available before decision in slot j+d+1",
            },
            "common_policy_semantics": {
                "individual_queue_update": "max(Q-pbar,0)+p",
                "priority_success_probability": {
                    "formula": "theta*phi_L+(1-theta)*phi_N",
                    "applies_to": [
                        "FreshSky-Safe",
                        "Static-Geometry Safe",
                        "Known-Transition Safe",
                        "Conservative-AoI Safe",
                        "Ji'24-style Greedy-Safe",
                        "Zhao'25 MMSE-MW-Safe",
                        "Tripathi'24 Whittle-Safe",
                        "Zhu'26-style Aggregate-DPP",
                        "factorial FreshSky adaptations",
                    ],
                    "exceptions": {
                        "Wang'26 PORMAB-Safe":
                            "transition/belief Whittle-like index",
                        "Max-Age-First Safe": "value-age score without q",
                        "Round-Robin Safe": "cyclic order without ranking",
                    },
                },
            },
            "families": {
                "calibration": {
                    "seeds": list(range(10)),
                    "slots": 4000,
                    "warmup_slots": 300,
                    "pending_feedback_slots": [2, 4, 8],
                    "mean_nlos_dwells": [2.0, 4.0, 8.0, 14.0],
                    "policies": [
                        "Expected-AoI Bayes-Safe",
                        "Conservative-AoI Bayes-Safe",
                        "Perfect-AoI Bayes-Safe",
                    ],
                    "individual_budget": 0.4,
                    "queue_cap": 24.0,
                },
                "measured": {
                    "seeds": list(range(20)),
                    "slots": 5000,
                    "warmup_slots": 500,
                    "pending_feedback_slots": [4, 8],
                    "power_quantiles": [0.25, 0.50, 0.75],
                    "policies": [
                        "Expected-AoI Bayes-Safe",
                        "Conservative-AoI Bayes-Safe",
                        "Perfect-AoI Bayes-Safe",
                    ],
                    "individual_budget": 0.4,
                    "queue_cap": 24.0,
                },
                "baseline": {
                    "seeds": list(range(20)),
                    "slots": 4000,
                    "warmup_slots": 300,
                    "pending_feedback_slots": [2, 4, 8],
                    "mean_nlos_dwell": 8.0,
                    "policies": 11,
                    "comparators": 10,
                    "zhao25_adaptation": {
                        "priority": "beta_i*q_i(t)*Ahat_i(t)",
                        "beta_calibration":
                            "Theorem-7 fixed beta from stationary geometry",
                        "source_system_time": 0,
                        "forward_delivery_delay": 0,
                        "event_weight_in_priority": False,
                        "common_additions": [
                            "posterior-predictive q_i(t)",
                            "individual Qmax shield",
                        ],
                        "guarantee_boundary":
                            "original asymptotic bound is not transferred",
                    },
                    "individual_budget": 0.4,
                    "queue_cap": 24.0,
                },
                "factorial": {
                    "seeds": list(range(20)),
                    "slots": 4000,
                    "warmup_slots": 300,
                    "pending_feedback_slots": [4],
                    "mean_nlos_dwell": 8.0,
                    "design": "2x2x2 belief x age-estimator x shield",
                    "individual_budget": 0.4,
                    "queue_cap": 24.0,
                },
                "qcap": {
                    "seeds": list(range(20)),
                    "slots": 4000,
                    "warmup_slots": 300,
                    "pending_feedback_slots": [4],
                    "mean_nlos_dwell": 8.0,
                    "queue_caps": [
                        4, 8, 12, 16, 24, 28, 32, 36, 48
                    ],
                    "individual_budget": 0.4,
                },
                "shift": {
                    "seeds": list(range(20)),
                    "slots": 4000,
                    "warmup_slots": 300,
                    "pending_feedback_slots": [4],
                    "shift_slot": 2000,
                    "mean_nlos_dwell_pre": 2.0,
                    "mean_nlos_dwell_post": 14.0,
                    "policies": 4,
                    "individual_budget": 0.4,
                    "queue_cap": 24.0,
                },
                "runtime": {
                    "seeds": None,
                    "deterministic_seed":
                        "880000+1000*N+10*grid_size+d",
                    "slots": None,
                    "warmup_iterations": 20,
                    "repeats": 1000,
                    "pending_feedback_slots": [0, 4, 8],
                    "fleet_sizes": [12, 25, 50, 100, 200, 500],
                    "transition_grid_sizes": [6, 12, 20],
                    "individual_budget": 0.4,
                    "queue_cap": 24.0,
                },
            },
            "strict_common_random_numbers": {
                "enabled": True,
                "external_streams": [
                    "geometry",
                    "initial_state",
                    "acknowledgement",
                    "channel_transition",
                    "event_transition",
                ],
            },
        },
        "source_sha256": {
            **{
                name: sha256(os.path.join(HERE, name))
                for name in SCRIPTS
            },
            **{
                f"analysis/{name}": sha256(path)
                for name, path in zip(
                    FIGURE_MODULE_NAMES, FIGURE_MODULES
                )
            },
            "analysis/figure_generation.py":
                sha256(CONCEPT_FIGURE_SCRIPT),
        },
        "input_sha256": {
            "experiments/data/measured_a2g_urban_3p6ghz.xlsx":
                sha256(MEASURED_INPUT),
        },
        "outputs": {
            name: sha256(os.path.join(RESULTS, name))
            for name in OUTPUTS
        },
        "row_counts": row_counts,
        "figure_outputs": {
            name: sha256(os.path.join(FIGURE_DIR, name))
            for name in FIGURE_OUTPUTS
        },
    }
    path = os.path.join(RESULTS, "tmc_evidence_manifest.json")
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
    print(f"saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="record versions and hashes without rerunning experiments",
    )
    args = parser.parse_args()
    if not args.manifest_only:
        subprocess.run(
            [
                sys.executable,
                "tmc_calibration_strict.py",
                "--project",
                PROJECT_ROOT,
                "--output",
                RESULTS,
            ],
            cwd=HERE,
            check=True,
        )
        subprocess.run(
            [sys.executable, CONCEPT_FIGURE_SCRIPT],
            cwd=os.path.dirname(CONCEPT_FIGURE_SCRIPT),
            check=True,
        )
        os.makedirs(FIGURE_DIR, exist_ok=True)
        for name in CONCEPT_FIGURE_OUTPUTS:
            shutil.copy2(
                os.path.join(CONCEPT_FIGURE_SOURCE_DIR, name),
                os.path.join(FIGURE_DIR, name),
            )
        subprocess.run(
            [
                sys.executable,
                "tmc_measured_strict.py",
                "--project",
                PROJECT_ROOT,
                "--output",
                RESULTS,
            ],
            cwd=HERE,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "tmc_runtime_scaling.py",
                "--output",
                RESULTS,
            ],
            cwd=HERE,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "tmc_extended_evidence.py",
                "--project",
                PROJECT_ROOT,
                "--output",
                EXTENDED_RESULTS,
            ],
            cwd=HERE,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "tmc_extended_evidence.py",
                "--project",
                PROJECT_ROOT,
                "--output",
                EXTENDED_RESULTS,
                "--shift-only",
            ],
            cwd=HERE,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                FIGURE_SCRIPT,
                "--project-root",
                PROJECT_ROOT,
                "--extended-results",
                EXTENDED_RESULTS,
                "--legacy-results",
                RESULTS,
                "--output-dir",
                FIGURE_DIR,
                "--derived-dir",
                DERIVED_RESULTS,
            ],
            cwd=HERE,
            check=True,
        )
    write_manifest()


if __name__ == "__main__":
    main()
