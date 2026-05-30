"""Local hyperparameter sweep runner.

Reads a YAML describing a list of runs, executes `scripts/train.py` once per
run, and aggregates the best `val_loss` reported by each into a single CSV.
This is the dependency-free option — for proper Bayesian / parallel sweeps
on a cluster, use `configs/wandb_sweep.yaml` with `wandb agent` instead.

Sweep YAML format
-----------------

    base_config: configs/config.yaml          # path to the base config
    common_set:                               # applied to every run
      - "training.epochs=10"
      - "training.early_stopping_enabled=true"
    runs:
      - name: small
        set:
          - "model.d_model=256"
          - "model.n_layers=4"
      - name: medium
        set:
          - "model.d_model=512"
          - "model.n_layers=8"

Usage
-----

    python scripts/sweep.py --sweep configs/sweep_example.yaml --data_dir data/processed

Outputs (under `sweep_results/<timestamp>/`):

    summary.csv         — per-run name, best val_loss, runtime, return code.
    <run_name>.log      — captured stdout/stderr for inspection.
    <run_name>.json     — best metrics scraped from CSVLogger.

A failed run does not stop the sweep — it is recorded with the non-zero
return code so other configs still execute on the rented GPU.
"""
from __future__ import annotations

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import yaml

from jam_transformer.utils.logger import logger


def _scrape_best_val_loss(csv_logger_root: Path, run_name: str) -> Optional[float]:
    """Walk the CSVLogger output (`<root>/<name>/version_*/metrics.csv`) and
    return the minimum val_loss seen, or None if the file is missing."""
    candidates = sorted(csv_logger_root.glob(f"{run_name}/version_*/metrics.csv"))
    if not candidates:
        return None
    latest = candidates[-1]
    best: Optional[float] = None
    with latest.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            v = row.get("val_loss") or ""
            if not v:
                continue
            try:
                x = float(v)
            except ValueError:
                continue
            if best is None or x < best:
                best = x
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Local hyperparameter sweep.")
    parser.add_argument("--sweep", type=str, required=True,
                        help="Path to a sweep YAML (see docstring for format).")
    parser.add_argument("--data_dir", type=str, default="data/processed",
                        help="Forwarded to scripts/train.py.")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Where to put summary.csv + per-run artifacts. "
                             "Defaults to sweep_results/<timestamp>/.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print the train command for each run, do not execute.")
    args = parser.parse_args()

    sweep_path = Path(args.sweep)
    if not sweep_path.exists():
        raise SystemExit(f"Sweep YAML not found: {sweep_path}")
    spec = yaml.safe_load(sweep_path.read_text(encoding="utf-8")) or {}
    base_config = spec.get("base_config", "configs/config.yaml")
    common_set: list[str] = list(spec.get("common_set") or [])
    runs = spec.get("runs") or []
    if not runs:
        raise SystemExit("No `runs:` entries in sweep YAML.")

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path("sweep_results") / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Sweep output: {out_dir}")

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run_name", "return_code", "runtime_sec", "best_val_loss", "overrides"])

        for run in runs:
            name = run.get("name") or "run"
            overrides: list[str] = list(common_set) + list(run.get("set") or [])
            # Pin per-run output directories so runs don't clobber each other.
            ckpt_dir = (out_dir / name / "checkpoints").as_posix()
            log_dir  = (out_dir / name / "logs").as_posix()
            overrides.append(f"training.checkpoint_dir={ckpt_dir}")
            overrides.append(f"training.log_dir={log_dir}")

            cmd = [
                sys.executable, "scripts/train.py",
                "--config", base_config,
                "--data_dir", args.data_dir,
                "--run_name", name,
            ]
            for o in overrides:
                cmd.extend(["--set", o])

            logger.info(f"── run {name} ──")
            logger.info("  " + " ".join(cmd))
            if args.dry_run:
                writer.writerow([name, 0, 0.0, "", ";".join(overrides)])
                continue

            stdout_path = out_dir / f"{name}.log"
            env = os.environ.copy()
            env.setdefault("PYTHONIOENCODING", "utf-8")
            t0 = dt.datetime.now()
            with stdout_path.open("w", encoding="utf-8") as logf:
                proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                       env=env)
            runtime = (dt.datetime.now() - t0).total_seconds()
            best = _scrape_best_val_loss(Path(log_dir), name)
            best_str = f"{best:.6f}" if best is not None else ""
            writer.writerow([name, proc.returncode, f"{runtime:.1f}", best_str,
                             ";".join(overrides)])
            (out_dir / f"{name}.json").write_text(
                json.dumps({"run_name": name, "return_code": proc.returncode,
                            "runtime_sec": runtime, "best_val_loss": best,
                            "overrides": overrides}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info(f"  rc={proc.returncode}  {runtime:.1f}s  "
                        f"best_val_loss={best_str or 'n/a'}")

    logger.info(f"Summary written: {summary_path}")


if __name__ == "__main__":
    main()
