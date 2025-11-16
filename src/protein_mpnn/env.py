import os
import subprocess
from importlib.resources import as_file, files
from importlib.resources.abc import Traversable
from pathlib import Path

from protein_mpnn.utils import norm_path


def _detect_project_root_dir() -> Path:
    """Determine the project root directory with the following priority:

    1) Explicit environment variable PROJECT_ROOT_DIR
    2) Search upward from current file for directory containing pyproject.toml
    3) git rev-parse --show-toplevel
    4) Fallback: parent directory of the package src directory
    """

    project_root_dir_env = os.getenv("PROJECT_ROOT_DIR")
    if project_root_dir_env is not None:
        return norm_path(project_root_dir_env)

    here = norm_path(__file__)
    for parent_dir in (here, *here.parents):
        if (parent_dir / "pyproject.toml").is_file():
            return parent_dir

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if out:
            return norm_path(out)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass

    return here.parents[2]


def _detect_package_root_dir() -> Path:
    """Return a real filesystem path to the package root (zip-safe)."""
    root: Traversable = files(__package__)
    with as_file(root) as p:
        return Path(p)


PROJECT_ROOT_DIR = _detect_project_root_dir()
PACKAGE_ROOT_DIR = _detect_package_root_dir()


__all__ = [
    "PACKAGE_ROOT_DIR",
    "PROJECT_ROOT_DIR",
]
