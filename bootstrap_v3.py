#!/usr/bin/env python3
"""Prepare an isolated, pinned Vortex v3.0 environment on Ubuntu or WSL."""

import argparse
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

from project_paths import WORK

COMMIT = "7fe19369587c087f902c3e2bc2add612aa910a95"
COURSE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=WORK)
    parser.add_argument("--phase", choices=("setup", "smoke", "measure", "trace"), default="setup")
    args = parser.parse_args()
    work = args.work_root.expanduser().resolve()
    if work in (Path("/"), Path.home()) or any(c.isspace() for c in str(work)):
        parser.error("Use a dedicated Linux directory without whitespace")
    source = work / "vortex"
    build = source / "build32"
    tools = work / "tools-v3.0.1"
    for target in (source, build, tools):
        if not target.resolve().is_relative_to(work):
            parser.error("All targets must remain inside the dedicated work root")
    work.mkdir(parents=True, exist_ok=True)
    marker = work / "project20_state.json"
    if marker.exists():
        state = json.loads(marker.read_text())
        if state.get("source_commit") != COMMIT:
            raise RuntimeError("Existing work root has a different source pin")
    else:
        if source.exists() or tools.exists():
            raise RuntimeError("Existing source/tools have no project marker; inspect before reuse")
        state = {"source_commit": COMMIT, "source_tag": "v3.0",
                 "toolchain_rev": "v3.0.1", "work_root": str(work),
                 "source": str(source), "build": str(build), "tools": str(tools),
                 "installed_components": [], "steps": []}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logs = COURSE / "logs" / (args.phase + "_" + stamp)
    logs.mkdir(parents=True, exist_ok=False)
    state.update(status="running", phase=args.phase, latest_logs=str(logs))
    env = os.environ.copy()
    for name in ("CONFIGS", "DEBUG", "PERF", "SCOPE", "TOOLDIR", "TOOLCHAIN_REV",
                 "VORTEX_DRIVER", "VORTEX_PROFILING", "VORTEX_HOME", "VORTEX_PATH",
                 "VORTEX_BUILD_DIR", "VORTEX_RT_PATH", "VORTEX_KN_PATH",
                 "MAKEFLAGS", "MFLAGS", "DESTDIR", "SIMX_FUNCTIONAL"):
        env.pop(name, None)
    env.update(CCACHE_DISABLE="1", DEBIAN_FRONTEND="noninteractive")

    def save():
        contents = json.dumps(state, indent=2) + "\n"
        marker.write_text(contents)
        (COURSE / "baseline_state.json").write_text(contents)

    def run(name, command, cwd=work, timeout=1800, extra_env=None):
        command = [str(c) for c in command]
        logfile = logs / (name + ".log")
        print(name + ": " + shlex.join(command), flush=True)
        started = datetime.now(timezone.utc).isoformat()
        save()
        with logfile.open("w") as output:
            output.write("command=" + shlex.join(command) + "\ncwd=" + str(cwd) + "\n")
            if extra_env:
                output.write("environment=" + json.dumps(extra_env) + "\n")
            output.flush()
            try:
                p = subprocess.run(command, cwd=cwd, env={**env, **(extra_env or {})}, stdout=output,
                                   stderr=subprocess.STDOUT, timeout=timeout)
                code = p.returncode
            except subprocess.TimeoutExpired:
                code = 124
                output.write("\nTimed out\n")
        state["steps"].append({"name": name, "command": command, "cwd": str(cwd),
                               "log": str(logfile), "started_utc": started,
                               "exit_code": code})
        save()
        content = logfile.read_text(errors="replace")
        if code:
            print(content[-6000:], flush=True)
            raise RuntimeError(name + " failed with exit " + str(code))
        print(name + ": completed", flush=True)
        return content

    try:
        save()
        if args.phase == "setup":
            if not source.exists():
                run("clone", ["git", "clone", "--depth=1", "--branch", "v3.0",
                              "https://github.com/vortexgpgpu/vortex.git", source], timeout=300)
            actual = run("source_commit", ["git", "rev-parse", "HEAD"], source).splitlines()[-1]
            if actual != COMMIT:
                raise RuntimeError("Tag no longer matches the agreed commit; inspect before continuing")
            run("submodules", ["git", "submodule", "update", "--init", "--recursive",
                               "--depth=1"], source, timeout=600)
            run("submodule_versions", ["git", "submodule", "status", "--recursive"], source)
            build.mkdir(exist_ok=True)
            run("configure", ["../configure", "--xlen=32", "--tooldir=" + str(tools)], build)
            for component in ("llvm", "riscv32", "libcrt32", "libc32", "verilator"):
                if component not in state["installed_components"]:
                    run("install_" + component, ["./ci/toolchain_install.sh", "--" + component], build, timeout=3600)
                    state["installed_components"].append(component)
                    save()
            run("third_party", ["make", "-j4", "third_party"], build, timeout=900)
            run("kernel_runtime", ["make", "-j4", "software"], build, timeout=600)
        elif args.phase == "smoke":
            run("configure", ["../configure"], build)
            run("demo", ["./ci/blackbox.sh", "--driver=simx", "--app=demo"], build,
                extra_env={"MAKEFLAGS": "-j4"})
            run("sgemm", ["./ci/blackbox.sh", "--driver=simx", "--app=sgemm", "--args=-n32"], build,
                extra_env={"MAKEFLAGS": "-j4"})
        elif args.phase == "measure":
            measured_build = source / "build32-wgmma"
            measured_build.mkdir(exist_ok=True)
            run("configure", ["../configure", "--xlen=32", "--tooldir=" + str(tools)], measured_build)
            run("kernel_runtime", ["make", "-j4", "software"], measured_build, timeout=600)
            base_flags = " ".join((
                "-DVX_CFG_NUM_THREADS=32", "-DVX_CFG_NUM_WARPS=8", "-DVX_CFG_ISSUE_WIDTH=4",
                "-DVX_CFG_EXT_TCU_ENABLE", "-DVX_CFG_TCU_WGMMA_ENABLE", "-DVX_CFG_EXT_DXA_ENABLE",
                "-DWGMMA_NRC=16", "-DVX_CFG_NUM_ALU_BLOCKS=1", "-DVX_CFG_NUM_LSU_BLOCKS=1",
                "-DVX_CFG_NUM_FPU_BLOCKS=1", "-DITYPE=fp16", "-DOTYPE=fp32", "-DPERF_ENABLE"))
            state["measurement_build"] = str(measured_build)
            state["measurements"] = []
            for variant in ("single", "double"):
                flags = base_flags + (" -DDXA_DOUBLE_BUFFER" if variant == "double" else "")
                case_env = {"CONFIGS": flags, "MAKEFLAGS": "-j4"}
                run("build_" + variant, ["make", "-j4", "-C", "tests/regression/sgemm_tcu_wg_dxa"],
                    measured_build, extra_env=case_env)
                for profile in (7, 16, 11):
                    name = variant + "_perf" + str(profile)
                    content = run(name, ["./ci/blackbox.sh", "--driver=simx", "--app=sgemm_tcu_wg_dxa",
                                        "--args=-m 128 -n 128 -k 64", "--perf=" + str(profile)],
                                  measured_build, extra_env=case_env)
                    if not re.search(r"^PASSED!?$", content, re.MULTILINE):
                        raise RuntimeError(name + " did not report a successful verification")
                    m = re.findall(r"PERF:\s+instrs=(\d+),\s*cycles=(\d+),\s*IPC=([0-9.]+)", content)
                    state["measurements"].append({"variant": variant, "profile": profile,
                        "configs": flags, "args": "-m 128 -n 128 -k 64", "passed": True,
                        "summary": list(m[-1]) if m else None,
                        "perf_lines": [l for l in content.splitlines() if l.startswith("PERF:")],
                        "log": str(logs / (name + ".log"))})
                    save()
        else:
            trace_build = source / "build32-dxa-trace"
            trace_build.mkdir(exist_ok=True)
            run("configure", ["../configure", "--xlen=32", "--tooldir=" + str(tools)], trace_build)
            run("kernel_runtime", ["make", "-j4", "software"], trace_build, timeout=600)
            trace_env = {"CONFIGS": "-DVX_CFG_EXT_DXA_ENABLE", "MAKEFLAGS": "-j4"}
            run("dxa_trace", ["./ci/blackbox.sh", "--driver=simx", "--app=dxa_copy",
                          "--args=-d1", "--debug=3", "--log=" + str(trace_build / "dxa_run.log")],
                          trace_build, extra_env=trace_env)
            raw = trace_build / "dxa_run.log"
            if not re.search(r"^PASSED!?$", raw.read_text(errors="replace"), re.MULTILINE):
                raise RuntimeError("DXA trace run did not report a successful verification")
            shutil.copyfile(raw, logs / "dxa_run.log")
            destination = logs / "dxa.perfetto.json.gz"
            run("perfetto", ["python3", "ci/perfetto.py", raw, "-t", "simx", "-c", "-o", destination], trace_build)
            with gzip.open(destination, "rt") as trace_file:
                trace_data = json.load(trace_file)
            events = trace_data["traceEvents"] if isinstance(trace_data, dict) else trace_data
            if not any(e.get("ph") in ("b", "e", "B", "E", "X") for e in events):
                raise RuntimeError("Trace contains no instruction slices")
            state["trace"] = {"raw_log": str(raw), "perfetto": str(destination),
                              "build": str(trace_build), "app": "dxa_copy", "passed": True,
                              "event_count": len(events)}
        state[args.phase + "_complete"] = True
        state["status"] = "complete"
        state.pop("error", None)
        save()
        print(json.dumps({"status": "complete", "phase": args.phase, "logs": str(logs)}), flush=True)
    except Exception as exc:
        state.update(status="failed", error=str(exc))
        save()
        print("FAILED: " + str(exc), file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
