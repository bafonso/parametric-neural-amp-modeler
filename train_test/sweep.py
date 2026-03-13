#!/usr/bin/env python3
"""
NAM WaveNet Architecture Comparison Sweep

Run from the repo root:
    python train_test/sweep.py [options]

Options:
    --named NAME1,NAME2,...   Comma-separated experiments to run (default: all named)
    --grid                    Also run grid sweep experiments
    --epochs N                Max epochs per run (default: 20)
    --output-dir DIR          Output dir relative to train_test/ (default: sweep_output)
    --eval-only               Skip training, re-evaluate existing checkpoints only

Examples:
    python train_test/sweep.py --named standard_base,slimmable --epochs 5
    python train_test/sweep.py --epochs 20
    python train_test/sweep.py --eval-only
"""

import argparse
import copy
import csv
import itertools
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch

# Resolve paths relative to this script
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from nam.data import Split, init_dataset
from nam.models.wavenet._slimmable import Slimmable
from nam.train.full import main as train_main
from nam.train.lightning_module import LightningModule

# ─────────────────────────────────────────────────────────────────────────────
# Experiment Registry
# Edit NAMED_EXPERIMENTS to add/remove experiments.
# Paths are relative to train_test/.
# ─────────────────────────────────────────────────────────────────────────────

NAMED_EXPERIMENTS: Dict[str, str] = {
    "standard_base":  "model.json",                              # 2-layer, 16/8ch, Tanh, pow2 dilations
    "slimmable":      "experiments/wavenet_slimmable.json",      # 1-layer, 12ch→3ch, LeakyReLU, slim dilations
    "1layer_tanh":    "experiments/wavenet_1layer_tanh.json",    # 1-layer, 16ch, Tanh, pow2 dilations
    "narrow":         "experiments/wavenet_narrow.json",         # 2-layer, 8/4ch, Tanh
    "wide":           "experiments/wavenet_wide.json",           # 2-layer, 32/16ch, Tanh
    "lrelu":          "experiments/wavenet_lrelu.json",          # 2-layer, 16/8ch, LeakyReLU
    "slim_dilations": "experiments/wavenet_slim_dilations.json", # 1-layer, 16ch, Tanh, [1,5,29,97,227] dilations
}

# Grid sweep: cartesian product of overrides applied to a base model config.
# Keys use dot notation with bracket indexing for lists.
# Leave param_grid empty ({}) to disable the grid sweep.
GRID_SWEEP: Dict[str, Any] = {
    "base_config": "model.json",
    "param_grid": {
        # "net.config.layers_configs[0].channels": [8, 16, 32],
        # "net.config.layers_configs[0].activation": ["Tanh", "LeakyReLU"],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Config utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def deep_set(config: dict, dotted_key: str, value: Any) -> dict:
    """Set a nested value using dot notation, e.g. 'net.config.layers_configs[0].channels'."""
    config = copy.deepcopy(config)
    # Split on dots that are not inside brackets
    keys = re.split(r"\.(?![^\[]*\])", dotted_key)
    node = config
    for key in keys[:-1]:
        m = re.match(r"^(\w+)\[(\d+)\]$", key)
        if m:
            node = node[m.group(1)][int(m.group(2))]
        else:
            node = node[key]
    last = keys[-1]
    m = re.match(r"^(\w+)\[(\d+)\]$", last)
    if m:
        node[m.group(1)][int(m.group(2))] = value
    else:
        node[last] = value
    return config


def generate_grid_configs(
    base_config: dict, param_grid: Dict[str, List[Any]]
) -> Iterator[Tuple[str, dict]]:
    """Yield (name, config) for each combination in the cartesian product."""
    if not param_grid:
        return
    keys = list(param_grid.keys())
    for combo in itertools.product(*param_grid.values()):
        parts = []
        cfg = copy.deepcopy(base_config)
        for k, v in zip(keys, combo):
            short_key = re.sub(r"\[\d+\]", "", k.split(".")[-1])
            parts.append(f"{short_key}={v}")
            cfg = deep_set(cfg, k, v)
        name = "grid_" + "_".join(parts)
        yield name, cfg


def resolve_data_paths(data_config: dict) -> dict:
    """Make x_path and y_path absolute relative to the repo root."""
    dc = copy.deepcopy(data_config)
    for section in ("common", "train", "validation"):
        for key in ("x_path", "y_path"):
            if key in dc.get(section, {}):
                p = Path(dc[section][key])
                if not p.is_absolute():
                    dc[section][key] = str(REPO_ROOT / p)
    return dc


# ─────────────────────────────────────────────────────────────────────────────
# Slimmable helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_slimmable(model_config: dict) -> bool:
    layers = model_config.get("net", {}).get("config", {}).get("layers_configs", [])
    return any("slimmable" in lc for lc in layers)


def get_allowed_channels(model_config: dict) -> List[int]:
    for lc in model_config.get("net", {}).get("config", {}).get("layers_configs", []):
        if "slimmable" in lc:
            return lc["slimmable"]["kwargs"]["allowed_channels"]
    return []


def get_max_channels(model_config: dict) -> int:
    for lc in model_config.get("net", {}).get("config", {}).get("layers_configs", []):
        if "slimmable" in lc:
            return lc["channels"]
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def find_best_checkpoint(exp_dir: Path) -> Optional[Path]:
    """Return the checkpoint with the lowest ESR value in its filename."""
    ckpts = list(exp_dir.glob("**/checkpoints/*.ckpt"))
    # Only checkpoints that encode ESR (not just epoch snapshots)
    scored = [c for c in ckpts if re.search(r"ESR", c.name, re.IGNORECASE)]
    if not scored:
        return None

    def extract_esr(path: Path) -> float:
        m = re.search(r"ESR[=_]([\d.e+\-]+)", path.name, re.IGNORECASE)
        return float(m.group(1)) if m else float("inf")

    return min(scored, key=extract_esr)


def compute_esr_mse(
    model: LightningModule, data_config: dict
) -> Tuple[float, float]:
    """Compute ESR and MSE on the full validation set."""
    dc = copy.deepcopy(data_config)
    dc.setdefault("common", {})["nx"] = model.net.receptive_field
    dc = resolve_data_paths(dc)

    ds = init_dataset(dc, Split.VALIDATION)
    with torch.no_grad():
        pred = model.net(ds.x).flatten()
        y = ds.y.flatten()
        esr_val = ((pred - y).pow(2).mean() / y.pow(2).mean()).item()
        mse_val = (pred - y).pow(2).mean().item()
    ds.teardown()
    return esr_val, mse_val


def get_first_layer_channels(model_config: dict) -> int:
    layers = model_config.get("net", {}).get("config", {}).get("layers_configs", [])
    return layers[0]["channels"] if layers else 0


def evaluate_experiment(
    name: str,
    exp_dir: Path,
    model_config: dict,
    data_config: dict,
) -> List[dict]:
    """
    Evaluate a trained experiment. Returns a list of result dicts:
    - One row per allowed width for slimmable models
    - One row for standard models
    """
    ckpt_path = find_best_checkpoint(exp_dir)
    if ckpt_path is None:
        print(f"  [!] No scored checkpoint found in {exp_dir}, skipping evaluation.")
        return []

    print(f"  Checkpoint: {ckpt_path.name}")
    parsed = LightningModule.parse_config(model_config)
    model = LightningModule.load_from_checkpoint(str(ckpt_path), **parsed)
    model.eval()
    model.cpu()

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    results = []

    if is_slimmable(model_config):
        allowed_channels = get_allowed_channels(model_config)
        max_channels = get_max_channels(model_config)
        for ch in allowed_channels:
            slimming_val = ch / max_channels
            Slimmable._set_slimming(model.net, value=slimming_val, recurse=True)
            esr_val, mse_val = compute_esr_mse(model, data_config)
            results.append({
                "name": name,
                "width_label": f"{ch}ch",
                "ESR": esr_val,
                "MSE": mse_val,
                "num_params": num_params,
                "effective_channels": ch,
            })
        Slimmable._set_slimming(model.net, value=1.0, recurse=True)
    else:
        esr_val, mse_val = compute_esr_mse(model, data_config)
        results.append({
            "name": name,
            "width_label": "full",
            "ESR": esr_val,
            "MSE": mse_val,
            "num_params": num_params,
            "effective_channels": get_first_layer_channels(model_config),
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    name: str,
    model_config: dict,
    data_config: dict,
    learning_config: dict,
    output_dir: Path,
    epochs: int,
) -> float:
    """Train one experiment. Returns wall-clock duration in seconds."""
    exp_dir = output_dir / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    lc = copy.deepcopy(learning_config)
    lc["trainer"]["max_epochs"] = epochs

    dc = resolve_data_paths(data_config)

    t0 = time.time()
    train_main(
        data_config=dc,
        model_config=copy.deepcopy(model_config),
        learning_config=lc,
        outdir=exp_dir,
        no_show=True,
        make_plots=True,
    )
    return time.time() - t0


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

_RESULT_FIELDS = ["name", "width_label", "ESR", "MSE", "num_params", "effective_channels", "epochs", "duration_s"]


def print_table(results: List[dict]) -> None:
    if not results:
        print("  (no results yet)")
        return

    def fmt(r, h):
        v = r.get(h, "")
        if h in ("ESR", "MSE") and isinstance(v, float):
            return f"{v:.4e}"
        return str(v)

    col_w = [max(len(h), max(len(fmt(r, h)) for r in results)) for h in _RESULT_FIELDS]

    def row_str(vals):
        return "  ".join(str(v).ljust(w) for v, w in zip(vals, col_w))

    sep = "-" * (sum(col_w) + 2 * (len(_RESULT_FIELDS) - 1))
    print("\n" + sep)
    print(row_str(_RESULT_FIELDS))
    print(sep)
    for r in sorted(results, key=lambda x: x.get("ESR", float("inf"))):
        print(row_str([fmt(r, h) for h in _RESULT_FIELDS]))
    print(sep + "\n")


def write_csv(results: List[dict], path: Path) -> None:
    if not results:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"Results written to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NAM WaveNet architecture comparison sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--named", type=str, default=None,
        help="Comma-separated named experiments to run (default: all)",
    )
    parser.add_argument(
        "--grid", action="store_true",
        help="Also run grid sweep experiments defined in GRID_SWEEP",
    )
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="Max epochs per training run (default: 20)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="sweep_output",
        help="Output directory relative to train_test/ (default: sweep_output)",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training; re-evaluate existing checkpoints only",
    )
    args = parser.parse_args()

    output_dir = SCRIPT_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load shared configs
    data_config = load_json(SCRIPT_DIR / "data.json")
    learning_config = load_json(SCRIPT_DIR / "learning.json")

    # Build experiment list
    experiments: Dict[str, dict] = {}

    names_to_run = (
        [n.strip() for n in args.named.split(",")]
        if args.named is not None
        else list(NAMED_EXPERIMENTS.keys())
    )
    for name in names_to_run:
        if name not in NAMED_EXPERIMENTS:
            print(f"[!] Unknown experiment '{name}'. Available: {list(NAMED_EXPERIMENTS.keys())}")
            continue
        experiments[name] = load_json(SCRIPT_DIR / NAMED_EXPERIMENTS[name])

    if args.grid and GRID_SWEEP.get("param_grid"):
        base_cfg = load_json(SCRIPT_DIR / GRID_SWEEP["base_config"])
        for name, cfg in generate_grid_configs(base_cfg, GRID_SWEEP["param_grid"]):
            experiments[name] = cfg

    if not experiments:
        print("No experiments to run.")
        return

    all_results: List[dict] = []

    for name, model_config in experiments.items():
        exp_dir = output_dir / name
        print(f"\n{'='*60}")
        print(f"  Experiment: {name}")
        print(f"{'='*60}")

        duration: Optional[float] = None
        already_trained = find_best_checkpoint(exp_dir) is not None
        if already_trained:
            print(f"  Checkpoint found — skipping training.")
        elif not args.eval_only:
            print(f"  Training for up to {args.epochs} epochs ...")
            try:
                duration = run_experiment(
                    name, model_config, data_config, learning_config,
                    output_dir, args.epochs,
                )
                print(f"  Training complete in {duration:.1f}s")
            except Exception as e:
                print(f"  [!] Training failed: {e}")
                continue

        print("  Evaluating ...")
        rows = evaluate_experiment(name, exp_dir, model_config, data_config)
        for r in rows:
            r["epochs"] = args.epochs
            r["duration_s"] = f"{duration:.1f}" if duration is not None else "N/A"
        all_results.extend(rows)

    print(f"\n{'='*60}")
    print("  Final results")
    print(f"{'='*60}")
    print_table(all_results)

    csv_path = output_dir / "results.csv"
    write_csv(all_results, csv_path)
    print(f"\nTensorBoard: tensorboard --logdir {output_dir}")


if __name__ == "__main__":
    main()
