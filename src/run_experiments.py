"""Run the full experiment suite from configs/experiment_config.json.

Usage (from falsification-challenge/):
    python -m src.run_experiments
    python -m src.run_experiments --config configs/experiment_config.json --out results

Outputs:
    results/runs/<condition>_seed<s>.json   -- one file per training run
    results/reference_metrics.json          -- analytic oracles + logistic baselines
    results/summary.json                    -- aggregated per-condition summaries
    results/run_manifest.json               -- config, versions, timings (reproducibility)
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import sklearn
import torch

from .evaluate import evaluate_mlp, evaluate_reference_predictor, mlp_predict_labels
from .generate_data import (
    bayes_joint_accuracy,
    bayes_single_accuracy,
    generate_dataset,
    shuffled_labels,
    simple_feature,
    complex_feature,
)
from .models import build_mlp, fit_logistic_baseline, logistic_predict
from .train import train_mlp


def base_seed_for(seed_index: int, stream: str) -> int:
    """Deterministic base seeds. Data streams depend only on (seed_index) so
    that paired conditions share identical inputs/noise; the 'shuffle' control
    uses a separate stream namespace."""
    ss = np.random.SeedSequence([20260, seed_index, {"data": 1, "model": 2, "eval": 3, "shuffle": 4}[stream]])
    return int(ss.generate_state(1)[0])


def run_one(condition: dict, defaults: dict, data_cfg: dict, seed_index: int, out_dir: Path) -> dict:
    cfg = {**defaults, **condition}
    alpha = float(cfg["alpha"])
    beta = float(cfg["beta"])
    k = int(cfg["k"])
    label_mode = cfg.get("label_mode", "both")
    shuffle_labels_flag = bool(cfg.get("shuffle_labels", False))
    swap_columns_flag = bool(cfg.get("swap_columns", False))
    n_mult = int(cfg.get("n_train_mult", 1))
    epochs = int(cfg.get("epochs", defaults["epochs"]))
    lr = float(cfg.get("lr", defaults["lr"]))
    width = list(cfg.get("width", defaults["width"]))
    activation = str(cfg.get("activation", defaults["activation"]))
    ckpt = list(defaults["longrun_checkpoint_epochs"] if "longrun" in cfg["name"]
                else defaults["checkpoint_epochs"])
    ckpt = [e for e in ckpt if e <= epochs]

    data_seed = base_seed_for(seed_index, "data")
    splits = generate_dataset(
        base_seed=data_seed,
        alpha=alpha, beta=beta, k=k,
        n_train=int(data_cfg["n_train"] * n_mult),
        n_val=data_cfg["n_val"], n_test=data_cfg["n_test"],
        n_conflict_test=data_cfg["n_conflict_test"],
        label_mode=label_mode, noise_sigma=data_cfg["noise_sigma"],
    )
    if shuffle_labels_flag:
        splits["train"]["y"] = shuffled_labels(splits["train"]["y"], base_seed_for(seed_index, "shuffle"))
    if swap_columns_flag:
        # Positional control: move the simple feature to column 1 and the
        # complex feature to column 0. All semantic labels (u_s, u_c) are kept
        # tied to the FEATURE, so the preference metric still means
        # "follows the simple/complex feature", not "column 0/1".
        for name, s in splits.items():
            s["x"] = s["x"][:, ::-1].copy()
            s["u_s"], s["u_c"] = s["u_c"].copy(), s["u_s"].copy()

    model = build_mlp(width=width, activation=activation, seed=base_seed_for(seed_index, "model"))
    t0 = time.time()
    train_out = train_mlp(
        model, splits["train"]["x"], splits["train"]["y"],
        epochs=epochs, lr=lr, batch_size=defaults["batch_size"],
        seed=base_seed_for(seed_index, "model") + 1,
        checkpoint_epochs=ckpt,
        eval_test=splits["test"], eval_conflict=splits["conflict_test"],
        weight_decay=defaults["weight_decay"],
    )
    train_seconds = time.time() - t0

    final_eval = evaluate_mlp(
        model, splits["test"], splits["conflict_test"],
        perm_seed=base_seed_for(seed_index, "eval"),
    )
    # Logistic baseline on the SAME training data (condition-specific).
    logreg = fit_logistic_baseline(splits["train"]["x"], splits["train"]["y"])
    logreg_eval = evaluate_reference_predictor(
        lambda z: logistic_predict(logreg, z), splits["test"], splits["conflict_test"],
        perm_seed=base_seed_for(seed_index, "eval"),
    )

    result = {
        "condition": cfg["name"],
        "seed_index": seed_index,
        "settings": {
            "alpha": alpha, "beta": beta, "k": k, "label_mode": label_mode,
            "shuffle_labels": shuffle_labels_flag, "swap_columns": swap_columns_flag,
            "n_train": len(splits["train"]["y"]),
            "epochs": epochs, "lr": lr, "width": width, "activation": activation,
            "batch_size": defaults["batch_size"],
        },
        "oracle": {
            "bayes_single_simple": bayes_single_accuracy(alpha, beta, "simple"),
            "bayes_single_complex": bayes_single_accuracy(alpha, beta, "complex"),
            "bayes_joint": bayes_joint_accuracy(alpha, beta),
            "label_positive_rate_test": float(np.mean(splits["test"]["y"] > 0)),
            "conflict_rate_test": float(np.mean(
                splits["test"]["u_s"] != splits["test"]["u_c"])),
        },
        "final_eval": final_eval,
        "logistic_baseline": logreg_eval,
        "checkpoints": train_out["checkpoints"],
        "train_loss_final": train_out["history"]["train_loss"][-1],
        "train_seconds": round(train_seconds, 2),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{cfg['name']}_seed{seed_index}.json"
    path.write_text(json.dumps(result, indent=2))
    return result


def load_all_runs(runs_dir: Path) -> list[dict]:
    """Load every recorded run from disk so the summary always reflects the
    full set of runs (supports incremental `--conditions` re-runs)."""
    return [json.loads(p.read_text()) for p in sorted(runs_dir.glob("*.json"))]


def summarize(results: list[dict]) -> dict:
    """Aggregate seed-level values per condition (seeds are the unit of
    replication; checkpoints/epochs are NOT treated as independent)."""
    by_cond: dict[str, list[dict]] = {}
    for r in results:
        by_cond.setdefault(r["condition"], []).append(r)

    def agg(vals):
        a = np.array(vals, dtype=float)
        return {"values": [round(float(v), 4) for v in a],
                "mean": round(float(a.mean()), 4),
                "std": round(float(a.std(ddof=1)), 4) if len(a) > 1 else None}

    summary = {}
    for cond, rs in sorted(by_cond.items()):
        rs = sorted(rs, key=lambda r: r["seed_index"])
        summary[cond] = {
            "n_seeds": len(rs),
            "settings": rs[0]["settings"],
            "test_accuracy": agg([r["final_eval"]["test_accuracy"] for r in rs]),
            "preference_index": agg([
                r["final_eval"]["conflict_preference"]["preference_index"] for r in rs]),
            "ablation_drop_simple": agg([
                r["final_eval"]["ablation"]["drop_simple"] for r in rs]),
            "ablation_drop_complex": agg([
                r["final_eval"]["ablation"]["drop_complex"] for r in rs]),
            "logreg_test_accuracy": agg([
                r["logistic_baseline"]["test_accuracy"] for r in rs]),
            "logreg_preference_index": agg([
                r["logistic_baseline"]["conflict_preference"]["preference_index"] for r in rs]),
            "mean_train_seconds": round(float(np.mean([r["train_seconds"] for r in rs])), 1),
        }
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment_config.json")
    ap.add_argument("--out", default="results")
    ap.add_argument(
        "--conditions", default="all",
        help="Comma-separated condition names to run (default: all). "
             "Existing run files on disk are always included in the summary, "
             "so conditions can be added incrementally.",
    )
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    cfg_path = (root / args.config) if not Path(args.config).is_absolute() else Path(args.config)
    config = json.loads(cfg_path.read_text())
    out_dir = (root / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    runs_dir = out_dir / "runs"

    defaults = {
        **config["training_defaults"],
        "longrun_checkpoint_epochs": config["longrun_checkpoint_epochs"],
    }
    data_cfg = config["data"]

    if args.conditions == "all":
        conditions_to_run = list(config["conditions"])
    else:
        wanted = {c.strip() for c in args.conditions.split(",") if c.strip()}
        conditions_to_run = [c for c in config["conditions"] if c["name"] in wanted]
        missing = wanted - {c["name"] for c in conditions_to_run}
        if missing:
            raise SystemExit(f"Unknown conditions: {sorted(missing)}")

    manifest = {
        "experiment_name": config["experiment_name"],
        "config_file": str(cfg_path),
        "config": config,
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__,
                     "sklearn": sklearn.__version__, "torch": torch.__version__},
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    t0 = time.time()
    results = []
    for cond in conditions_to_run:
        for seed_index in config["seeds"]:
            r = run_one(cond, defaults, data_cfg, int(seed_index), runs_dir)
            results.append(r)
            print(f"[{r['condition']} seed {seed_index}] "
                  f"P={r['final_eval']['conflict_preference']['preference_index']:+.3f} "
                  f"acc={r['final_eval']['test_accuracy']:.3f} "
                  f"drop_s={r['final_eval']['ablation']['drop_simple']:.3f} "
                  f"drop_c={r['final_eval']['ablation']['drop_complex']:.3f} "
                  f"({r['train_seconds']:.1f}s)")

    # Reference oracles per unique (alpha, beta, k) condition, evaluated with
    # exact feature functions (no model involved).
    ref = {}
    for cond in config["conditions"]:
        alpha, beta, k = float(cond["alpha"]), float(cond["beta"]), int(cond["k"])
        key = f"a{alpha}_b{beta}_k{k}"
        if key in ref:
            continue
        splits = generate_dataset(
            base_seed=base_seed_for(99, "data"), alpha=alpha, beta=beta, k=k,
            n_train=data_cfg["n_train"], n_val=data_cfg["n_val"],
            n_test=data_cfg["n_test"], n_conflict_test=data_cfg["n_conflict_test"],
            noise_sigma=data_cfg["noise_sigma"],
        )
        oracles = {
            "oracle_simple": lambda z: simple_feature(z[:, 0]),
            "oracle_complex": lambda z: complex_feature(z[:, 1], k),
        }
        ref[key] = {
            "bayes_single_simple": bayes_single_accuracy(alpha, beta, "simple"),
            "bayes_single_complex": bayes_single_accuracy(alpha, beta, "complex"),
            "bayes_joint": bayes_joint_accuracy(alpha, beta),
            "feature_oracle_simple": evaluate_reference_predictor(
                oracles["oracle_simple"], splits["test"], splits["conflict_test"], 12345),
            "feature_oracle_complex": evaluate_reference_predictor(
                oracles["oracle_complex"], splits["test"], splits["conflict_test"], 12345),
        }

    summary = summarize(load_all_runs(runs_dir))
    manifest["conditions_run_this_invocation"] = [c["name"] for c in conditions_to_run]
    manifest["n_runs_total_on_disk"] = len(load_all_runs(runs_dir))
    manifest["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["total_seconds"] = round(time.time() - t0, 1)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "reference_metrics.json").write_text(json.dumps(ref, indent=2))
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nDone. {len(results)} runs in {manifest['total_seconds']}s. "
          f"Summary written to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
