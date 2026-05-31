"""Report the melody-selection method breakdown of preprocessed shards.

Each shard records how its melody track was chosen (stored by _save_shard):
  - "miner"      : midi-miner classifier (Lakh, and Slakh instrument fallback)
  - "weight"     : mono-rate weight heuristic (fallback when miner finds nothing)
  - "instrument" : Slakh metadata.yaml inst_class (near-GT)
  - None         : legacy shard written before provenance logging, or POP909 GT

Usage:
    python scripts/analysis/melody_method_stats.py
    python scripts/analysis/melody_method_stats.py --prefix slakh
    python scripts/analysis/melody_method_stats.py --data_dir data/processed_redux
"""
from __future__ import annotations

import argparse
from collections import defaultdict, Counter
from pathlib import Path

import torch
from tqdm import tqdm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--prefix", default=None,
                    help="Only scan shards whose name starts with this "
                         "(e.g. 'slakh', 'lakh'). Default: all.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    pattern = f"{args.prefix}_*.pt" if args.prefix else "*.pt"
    shards = sorted(p for p in data_dir.glob(pattern) if not p.name.startswith("_"))
    if not shards:
        raise SystemExit(f"No shards matching {pattern} under {data_dir}")

    # source prefix (pop909/lakh/slakh) -> method -> count
    table: dict[str, Counter] = defaultdict(Counter)
    for p in tqdm(shards, desc="scanning"):
        try:
            d = torch.load(p, map_location="cpu", weights_only=False)
        except Exception:
            table["?"]["load_fail"] += 1
            continue
        src = p.name.split("_", 1)[0]
        method = d.get("method") or "none (legacy/GT)"
        table[src][method] += 1

    print(f"\n=== melody method breakdown - {data_dir} "
          f"({len(shards):,} shards) ===")
    for src in sorted(table):
        total = sum(table[src].values())
        print(f"\n[{src}]  {total:,} shards")
        for method, n in table[src].most_common():
            print(f"    {method:20s} {n:7,}  ({100*n/total:5.1f}%)")

    # overall miner-vs-weight among shards that recorded it
    overall = Counter()
    for src in table:
        overall.update(table[src])
    tracked = sum(overall[m] for m in ("miner", "weight"))
    if tracked:
        print(f"\n=== miner vs weight (tracked shards only: {tracked:,}) ===")
        print(f"    miner  {overall['miner']:7,}  ({100*overall['miner']/tracked:5.1f}%)")
        print(f"    weight {overall['weight']:7,}  ({100*overall['weight']/tracked:5.1f}%)")
        if overall["instrument"]:
            print(f"    (instrument GT: {overall['instrument']:,} - Slakh metadata labels)")


if __name__ == "__main__":
    main()
