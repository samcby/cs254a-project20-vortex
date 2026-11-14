#!/usr/bin/env python3
"""Recompute archived reports or run and compare the timing SimX cost experiments."""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from scripts.project_paths import ROOT, SOURCE, TOOLS, WORK, result_path

COST = Path("experiments/cost_attribution")
ANALYSES = (
    "pc_pipeline/analyze.py",
    "gemm_relu/analyze_relu.py",
    "warp_capacity/analyze_sweep.py",
    "persistent_pipeline/analyze_persistent.py",
    "activation_balance/analyze_balance.py",
    "multi_cta/analyze_multi.py",
    "policy_generalization/analyze_policy.py",
    "policy_v2/analyze_v2.py",
    "cost_attribution/analyze_cost.py",
)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_script(output, script, *args):
    cmd = [sys.executable, str(script), *args]
    print("Running", *cmd, flush=True)
    env = dict(os.environ, PROJECT20_ARCHIVE_ROOT=str(ROOT))
    subprocess.run(cmd, cwd=output, env=env, check=True)


def compare_run(actual_path, expected_path):
    actual, expected = load(actual_path), load(expected_path)
    for key in ("status", "source_commit", "source_hashes", "binary_sha256", "effective_config"):
        if actual[key] != expected[key]:
            raise RuntimeError(f"{actual_path.name}: {key} differs from the archive")
    actual_cases, expected_cases = actual["cases"], expected["cases"]
    if len(actual_cases) != len(expected_cases):
        raise RuntimeError(f"{actual_path.name}: number of cases differs from the archive")
    fields = ("m", "n", "k", "ctas", "mode", "aw", "reserve", "seed", "passed",
              "output_hash", "chain", "core_counters", "smem", "cta_warps")
    for index, (a, e) in enumerate(zip(actual_cases, expected_cases), 1):
        for field in fields:
            if a[field] != e[field]:
                raise RuntimeError(f"{actual_path.name} case {index}: {field}: {a[field]} != {e[field]}")
    print(f"Matched {len(actual_cases)} runs in {actual_path.name}", flush=True)
    return len(actual_cases)


def compare_cost_report(output):
    relative = COST / "measurements.csv"
    with (output / relative).open(encoding="utf-8-sig", newline="") as actual:
        with (ROOT / relative).open(encoding="utf-8-sig", newline="") as expected:
            if list(csv.DictReader(actual)) != list(csv.DictReader(expected)):
                raise RuntimeError("The recomputed measurement rows differ from the archive")
    report = load(output / COST / "analysis.json")
    archived = load(ROOT / COST / "analysis.json")
    for name in ("runs", "formal_runs", "observer_counter_equality"):
        if report[name] != archived[name]:
            raise RuntimeError(f"Cost report differs for {name}")
    for group in ("main", "capacity"):
        a, e = (report, archived) if group == "main" else (report[group], archived[group])
        if [x["detail"] for x in a["traces"]] != [x["detail"] for x in e["traces"]]:
            raise RuntimeError(f"The {group} trace summaries differ from the archive")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("archive", "quick", "full"))
    parser.add_argument("--out", type=Path, help="New directory for generated files")
    args = parser.parse_args()
    if args.profile != "archive":
        if sys.platform != "linux" or any(c.isspace() for p in (WORK, TOOLS) for c in str(p)):
            parser.error("Use Linux/WSL with space-free Vortex and toolchain directories")
        if not (SOURCE / "configure").is_file() or not TOOLS.is_dir():
            parser.error("Run the setup instructions in README.md first")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.out or WORK / f"{args.profile}-{stamp}").expanduser().resolve()
    if output.is_relative_to(ROOT):
        parser.error("Choose an output directory outside the repository checkout")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(ROOT / "scripts", output / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "experiments", output / "experiments", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "results", output / "results")

    if args.profile == "archive":
        for script in ANALYSES:
            run_script(output, Path("experiments") / script)
        compare_cost_report(output)
        print(f"Archive analysis verified in {output}")
        return

    batches = [(16, "normal", False), (8, "normal", True)]
    if args.profile == "full":
        batches = [(8, "smoke", False)] + [
            (warps, phase, capacity)
            for warps in (8, 16)
            for phase in ("normal", "validate", "trace")
            for capacity in (False, True)
        ]
    passed = 0
    for warps, phase, capacity in batches:
        extra = ["--capacity"] if capacity else []
        run_script(output, COST / "run_cost.py", "--warps", str(warps), "--phase", phase, *extra)
        prefix = "cost_capacity" if capacity else "cost"
        name = f"{prefix}_w{warps}_{phase}_results.json"
        passed += compare_run(result_path(name, output), result_path(name))
    if args.profile == "full":
        run_script(output, COST / "analyze_cost.py")
        compare_cost_report(output)
    print(f"Reproduced {passed} timing SimX runs in {output}")


if __name__ == "__main__":
    main()
