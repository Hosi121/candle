#!/usr/bin/env python3
"""Compare complete CPU ConvTranspose1d calls in separate processes."""
import json
import os
from pathlib import Path
import statistics
import subprocess
import argparse

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--variants", nargs="+", default=["baseline", "candidate"])
parser.add_argument("--rounds", type=int, default=5)
parser.add_argument("--output", default="paired.json")
args = parser.parse_args()
records = []
for threads in (1, 4):
    for round_id in range(args.rounds):
        order = args.variants if round_id % 2 == 0 else args.variants[::-1]
        for variant in order:
            env = dict(os.environ, RAYON_NUM_THREADS=str(threads))
            command = ["taskset", "-c", "0" if threads == 1 else "0-3", str(root / (variant + "_probe"))]
            result = subprocess.run(command, env=env, check=True, text=True, capture_output=True)
            for line in result.stdout.splitlines():
                record = json.loads(line)
                record.update(threads=threads, round=round_id, variant=variant)
                records.append(record)
            (root / args.output).write_text(json.dumps(records, indent=2) + "\n")
            print(f"threads={threads} round={round_id} variant={variant}", flush=True)

summary = []
for threads in (1, 4):
    names = [r["case"] for r in records if r["threads"] == threads and r["round"] == 0 and r["variant"] == "baseline"]
    for name in names:
        group = [r for r in records if r["threads"] == threads and r["case"] == name]
        assert len({r["hash"] for r in group}) == 1, (threads, name)
        entry = dict(threads=threads, case=name)
        for variant in args.variants:
            entry[variant + "_ms"] = statistics.median(r["median_ms"] for r in group if r["variant"] == variant)
        for variant in args.variants[1:]:
            entry[variant + "_ratio_of_medians_percent"] = 100 * (1 - entry[variant + "_ms"] / entry["baseline_ms"])
            ratios = []
            for round_id in range(args.rounds):
                base = next(r["median_ms"] for r in group if r["variant"] == "baseline" and r["round"] == round_id)
                after = next(r["median_ms"] for r in group if r["variant"] == variant and r["round"] == round_id)
                ratios.append(100 * (1 - after / base))
            entry[variant + "_reduction_percent"] = statistics.median(ratios)
            entry[variant + "_range_percent"] = [min(ratios), max(ratios)]
        summary.append(entry)
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
