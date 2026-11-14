"""Paths shared by the Project 20 runners and archived trace readers."""

import os
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("PROJECT20_WORK_ROOT", Path.home() / "cs254a/project20")).expanduser().resolve()
SOURCE = WORK / "vortex"
TOOLS = Path(os.environ.get("PROJECT20_TOOLCHAIN_DIR", WORK / "tools-v3.0.1")).expanduser().resolve()
STUDY_PREFIXES = {
    "activation_balance": ("activation_",),
    "baseline": ("baseline_", "environment_", "measure_", "proposal_probe_", "setup_", "smoke_", "trace_"),
    "cost_attribution": ("cost_",),
    "gemm_relu": ("relu_",),
    "multi_cta": ("multicta_",),
    "pc_pipeline": ("pc_",),
    "persistent_pipeline": ("persistent_",),
    "policy_generalization": ("policy_",),
    "policy_v2": ("policy2_",),
    "warp_capacity": ("warp",),
}


def study_for(name):
    for study, prefixes in STUDY_PREFIXES.items():
        if name.startswith(prefixes):
            return study
    raise ValueError(f"Unrecognized Project 20 record: {name}")


def result_path(name, root=ROOT):
    """Locate the summary JSON for a batch in this checkout or a rerun."""
    return root / "results" / study_for(name) / name


def run_log_dir(batch):
    """Place logs from new runs alongside the corresponding archived study."""
    return ROOT / "archive" / "logs" / study_for(batch) / batch


def log_path(value):
    """Find either a newly generated log or the same file in the bundled archive."""
    original = Path(value)
    parts = PurePosixPath(value.replace("\\", "/")).parts
    if "logs" not in parts:
        if original.is_file():
            return original
        raise FileNotFoundError(value)
    relative = parts[len(parts) - parts[::-1].index("logs"):]
    if not relative or ".." in relative:
        raise ValueError(f"Invalid archived log path: {value}")
    if relative[0] not in STUDY_PREFIXES:
        relative = study_for(relative[0]), *relative
    roots = [ROOT]
    archive = os.environ.get("PROJECT20_ARCHIVE_ROOT")
    if archive:
        roots.append(Path(archive))
    for root in roots:
        candidate = (root / "archive" / "logs").joinpath(*relative)
        if candidate.is_file():
            return candidate
    if not archive and original.is_file():
        return original
    raise FileNotFoundError(value)
