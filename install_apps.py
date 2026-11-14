#!/usr/bin/env python3
"""Install the archived Project 20 application sources into the pinned Vortex tree."""

import argparse
from pathlib import Path
import shutil
import subprocess

from project_paths import SOURCE

COMMIT = "7fe19369587c087f902c3e2bc2add612aa910a95"
EXPERIMENTS = Path(__file__).resolve().parent / "experiments"


def main():
    apps = {path.parent.name: path.parent for path in EXPERIMENTS.glob("*/*/Makefile")}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("apps", nargs="*", help="Application directory names. Default: all archived apps.")
    args = parser.parse_args()
    names = args.apps or sorted(apps)
    if set(names) - apps.keys():
        parser.error(f"Available apps: {', '.join(sorted(apps))}")

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True).strip()
    if head != COMMIT:
        raise RuntimeError(f"Expected Vortex {COMMIT}, found {head}")
    for name in names:
        dest = SOURCE / "tests" / "regression" / name
        marker = dest / ".project20-owned"
        if dest.exists() and (not marker.is_file() or marker.read_text().strip() != COMMIT):
            raise RuntimeError(f"This application directory already exists: {dest}")
        dest.mkdir(exist_ok=True)
        for file in apps[name].iterdir():
            if file.is_file() and (file.name == "Makefile" or file.suffix in {".cpp", ".h"}):
                shutil.copyfile(file, dest / file.name)
        marker.write_text(COMMIT + "\n")
        print(f"Installed {name} in {dest}")


if __name__ == "__main__":
    main()
