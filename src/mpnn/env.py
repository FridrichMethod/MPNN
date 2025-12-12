import atexit
import os
import subprocess
from contextlib import ExitStack
from importlib.resources import as_file, files
from importlib.resources.abc import Traversable
from pathlib import Path

from mpnn.utils import norm_path

_PACKAGE_ROOT_DIR_EXIT_STACK = ExitStack()
atexit.register(_PACKAGE_ROOT_DIR_EXIT_STACK.close)


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

    assert len(here.parents) >= 2

    if here.parents[1].name == "src":
        return here.parents[2]

    return here.parents[1]


def _detect_package_root_dir() -> Path:
    """Return a real filesystem path to the package root (zip-safe)."""
    root: Traversable = files(__package__)
    if isinstance(root, Path):
        return root
    if isinstance(root, (str, os.PathLike)):
        return norm_path(os.fspath(root))

    # Falls back to extracting resources (zip-safe). Keep the context alive
    # for the lifetime of the interpreter so the returned path remains valid.
    extracted_path = _PACKAGE_ROOT_DIR_EXIT_STACK.enter_context(as_file(root))
    return Path(extracted_path)


PROJECT_ROOT_DIR = _detect_project_root_dir()
PACKAGE_ROOT_DIR = _detect_package_root_dir()

DEFAULT_TRAIN_DATA_PATH = PROJECT_ROOT_DIR / "datasets/pdb_2021aug02"
DEFAULT_TRAIN_OUTPUT_DIR = PROJECT_ROOT_DIR / "checkpoints/train"
DEFAULT_FINETUNE_OUTPUT_DIR = PROJECT_ROOT_DIR / "checkpoints/finetune"

MEGASCALE_SPLIT_PATH = PROJECT_ROOT_DIR / "datasets/megascale/mega_splits.pkl"
MEGASCALE_PDB_DIR = PROJECT_ROOT_DIR / "datasets/megascale/AlphaFold_model_PDBs"
MEGASCALE_CSV = PROJECT_ROOT_DIR / "datasets/megascale/Tsuboyama2023_Dataset2_Dataset3_20230416.csv"
MGNIFY_PDB_DIR = PROJECT_ROOT_DIR / "datasets/mgnify/wt_structures"
MGNIFY_CSV = PROJECT_ROOT_DIR / "datasets/mgnify/mgnify_processed_data.csv"
MGNIFY_CACHE_PATH = PROJECT_ROOT_DIR / "datasets/mgnify/mgnify.pkl"
FSD_THERMO_CSV = PROJECT_ROOT_DIR / "datasets/FSD/fsd_thermo.csv"
FSD_THERMO_PDB_DIR = PROJECT_ROOT_DIR / "datasets/FSD/PDBs"
FSD_THERMO_CACHE_PATH = PROJECT_ROOT_DIR / "datasets/FSD/fsd_thermo.pkl"


__all__ = [
    "DEFAULT_TRAIN_DATA_PATH",
    "DEFAULT_TRAIN_OUTPUT_DIR",
    "FSD_THERMO_CACHE_PATH",
    "FSD_THERMO_CSV",
    "FSD_THERMO_PDB_DIR",
    "MEGASCALE_CSV",
    "MEGASCALE_PDB_DIR",
    "MEGASCALE_SPLIT_PATH",
    "MGNIFY_CACHE_PATH",
    "MGNIFY_CSV",
    "MGNIFY_PDB_DIR",
    "PACKAGE_ROOT_DIR",
    "PROJECT_ROOT_DIR",
]
