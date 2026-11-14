"""Paths shared by the Project 20 runners and archived trace readers."""

import os
from pathlib import Path, PurePosixPath

WORK = Path(os.environ.get("PROJECT20_WORK_ROOT", Path.home() / "cs254a/project20")).expanduser().resolve()
SOURCE = WORK / "vortex"
TOOLS = Path(os.environ.get("PROJECT20_TOOLCHAIN_DIR", WORK / "tools-v3.0.1")).expanduser().resolve()


def log_path(value):
    """Find either a newly generated log or the same file in the bundled archive."""
    original = Path(value)
    parts = PurePosixPath(value.replace("\\", "/")).parts
    if "logs" not in parts:
        if original.is_file():
            return original
        raise FileNotFoundError(value)
    relative = parts[len(parts) - 1 - parts[::-1].index("logs"):]
    if ".." in relative:
        raise ValueError(f"Invalid archived log path: {value}")
    roots = [Path(__file__).resolve().parent]
    archive = os.environ.get("PROJECT20_ARCHIVE_ROOT")
    if archive:
        roots.append(Path(archive))
    for root in roots:
        candidate = root.joinpath(*relative)
        if candidate.is_file():
            return candidate
    if not archive and original.is_file():
        return original
    raise FileNotFoundError(value)
