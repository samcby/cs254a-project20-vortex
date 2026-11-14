#!/usr/bin/env python3
"""Run six bounded, unmodified-upstream GEMM probes for proposal scoping."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

from bootstrap_v3 import COMMIT, COURSE
from project_paths import SOURCE, TOOLS


def main():
    baseline = json.loads((COURSE / "baseline_state.json").read_text())
    source = SOURCE
    build = source / "build32-wgmma"
    build.mkdir(exist_ok=True)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if actual != COMMIT or subprocess.check_output(["git", "diff", "--name-only"], cwd=source, text=True):
        raise RuntimeError("The pinned, unmodified source is required")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logs = COURSE / "logs" / ("proposal_probe_" + stamp)
    logs.mkdir(parents=True, exist_ok=False)
    flags = next(c["configs"] for c in baseline["measurements"]
                 if c["variant"] == "single" and c["profile"] == 7)
    result = {"source_commit": actual, "purpose": "proposal feasibility exploration",
              "matrix_m": 128, "matrix_n": 128, "new_k_values": [32, 128, 256],
              "perf_class": 7, "logs": str(logs), "cases": [], "status": "running"}
    for c in baseline["measurements"]:
        if c["profile"] == 7:
            result["cases"].append({**c, "m": 128, "n": 128, "k": 64,
                                    "reused_prior_run": True})
    env = os.environ.copy()
    for name in ("CONFIGS", "DEBUG", "PERF", "SCOPE", "SIMX_FUNCTIONAL", "VORTEX_DRIVER", "VORTEX_PROFILING",
                 "VORTEX_HOME", "VORTEX_PATH", "VORTEX_BUILD_DIR", "VORTEX_RT_PATH",
                 "VORTEX_KN_PATH", "TOOLDIR", "TOOLCHAIN_REV", "DESTDIR", "MAKEFLAGS", "MFLAGS"):
        env.pop(name, None)
    env.update(CCACHE_DISABLE="1", MAKEFLAGS="-j4")

    def save():
        (COURSE / "proposal_probe_results.json").write_text(json.dumps(result, indent=2) + "\n")

    def run(name, command, case_env):
        print(name + ": " + shlex.join(command), flush=True)
        p = logs / (name + ".log")
        with p.open("w") as out:
            out.write("command=" + shlex.join(command) + "\ncwd=" + str(build) +
                      "\nCONFIGS=" + case_env.get("CONFIGS", "") + "\n")
            out.flush()
            proc = subprocess.run(command, cwd=build, env=case_env, stdout=out,
                                  stderr=subprocess.STDOUT, timeout=300)
        return proc.returncode, p.read_text(errors="replace"), str(p)

    save()
    try:
        code, text, _ = run("configure", ["../configure", "--xlen=32", f"--tooldir={TOOLS}"], env)
        if code:
            raise RuntimeError(text[-2000:])
        code, text, _ = run("software", ["make", "-j4", "software"], env)
        if code:
            raise RuntimeError(text[-2000:])
        for variant in ("single", "double"):
            config = flags + (" -DDXA_DOUBLE_BUFFER" if variant == "double" else "")
            case_env = {**env, "CONFIGS": config}
            code, text, _ = run("build_" + variant,
                               ["make", "-j4", "-C", "tests/regression/sgemm_tcu_wg_dxa"], case_env)
            if code:
                raise RuntimeError(text[-2000:])
            for k in result["new_k_values"]:
                code, text, log = run(variant + "_k" + str(k),
                    ["./ci/blackbox.sh", "--driver=simx", "--app=sgemm_tcu_wg_dxa",
                     "--args=-m 128 -n 128 -k " + str(k), "--perf=7"], case_env)
                summary = re.findall(r"PERF:\s+instrs=(\d+),\s*cycles=(\d+),\s*IPC=([0-9.]+)", text)
                passed = code == 0 and bool(re.search(r"^PASSED!?$", text, re.MULTILINE))
                case = {"variant": variant, "m": 128, "n": 128, "k": k,
                        "configs": config, "passed": passed, "exit_code": code,
                        "summary": list(summary[-1]) if summary else None,
                        "perf_lines": [l for l in text.splitlines() if l.startswith("PERF:")],
                        "verification_lines": [l for l in text.splitlines()
                            if l.startswith(("reference drift:", "smem:", "WGMMA tile:", "CTA tile:"))],
                        "log": log, "reused_prior_run": False}
                result["cases"].append(case)
                save()
                print(json.dumps({k: case[k] for k in ["variant", "k", "passed", "summary"]}), flush=True)
        result["status"] = "complete" if all(c["passed"] for c in result["cases"]) else "completed_with_failures"
        save()
    except Exception as exc:
        result.update(status="failed", error=str(exc))
        save()
        raise
    print(json.dumps({"status": result["status"], "logs": str(logs)}), flush=True)
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
