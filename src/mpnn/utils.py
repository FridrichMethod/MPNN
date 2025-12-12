import functools
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

import torch
import yaml

from mpnn.typing_utils import StrPath

P = ParamSpec("P")
T = TypeVar("T")


_conda_not_installed_errmsg = "conda not installed"


def get_conda_prefix() -> str:
    """Attempts to find the root Conda folder. Works with miniforge3/miniconda3"""
    conda_root = os.getenv("CONDA_ROOT", None)
    if conda_root is None:
        # Attempt $CONDA_PREFIX_1 or $CONDA_PREFIX, depending
        # on whether the `base` environment is activated.
        default_env_name = os.getenv("CONDA_DEFAULT_ENV", None)
        assert default_env_name is not None, _conda_not_installed_errmsg
        conda_prefix_env_name = "CONDA_PREFIX" if default_env_name == "base" else "CONDA_PREFIX_1"
        conda_root = os.getenv(conda_prefix_env_name, None)
    assert conda_root is not None, _conda_not_installed_errmsg
    return conda_root


def clean_gpu_cache[**P, T](func: Callable[P, T]) -> Callable[P, T]:
    """Decorator to clean GPU memory cache after the decorated function is executed."""
    counter = 0

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal counter
        try:
            result = func(*args, **kwargs)
        finally:
            # gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                counter += 1
        return result

    return wrapper


def norm_path(
    path: StrPath,
    expandvars: bool = True,
    expanduser: bool = True,
    resolve: bool = True,
) -> Path:
    """Normalize a file path.

    Args:
        path (StrPath): The file path to normalize.
        expandvars (bool, optional): Whether to expand environment variables. Defaults to True.
        expanduser (bool, optional): Whether to expand the user directory. Defaults to True.
        resolve (bool, optional): Whether to resolve the path. Defaults to True.

    Returns:
        Path: The normalized file path.

    """
    p = Path(path)
    if expandvars:
        p = Path(os.path.expandvars(p))
    if expanduser:
        p = p.expanduser()
    if resolve:
        p = p.resolve()

    return p


def load_config(config_path: StrPath) -> dict[str, Any]:
    """Load a config file."""
    config_path = norm_path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file {config_path} not found")

    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def enable_tf32_if_available(precision: str = "high") -> bool:
    """Enable TF32 on Ampere+ CUDA GPUs for matmul (cuBLAS) and conv (cuDNN).

    On non-Ampere or non-CUDA setups it safely does nothing (and disables flags).

    Returns:
        bool: True if TF32 is enabled, False otherwise.

    """
    enabled = False

    # Default to strict FP32 unless we detect Ampere+
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Highest = strict FP32; High = allow TF32 on Ampere for matmul
    torch.set_float32_matmul_precision("highest")

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        major, _ = torch.cuda.get_device_capability(device)
        if major >= 8:  # Ampere(8.x) / Hopper(9.x)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision(precision)

            enabled = True

    return enabled


def parse_amp_dtype(amp_dtype: str | None) -> torch.dtype | None:
    """Convert a user string (or None) to a torch.dtype (or None) for autocast.

    Args:
        amp_dtype (str | None): The string to parse.

    Returns:
        torch.dtype | None: The parsed torch.dtype.

    Note:
        Accepted strings (case-insensitive):
        - 'fp16', 'float16', 'half'
        - 'bf16', 'bfloat16'
        - 'fp32', 'float32'

    """
    if amp_dtype is None:
        return None

    amp_dtype = amp_dtype.strip().lower()

    table = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "float": torch.float32,
    }

    try:
        return table[amp_dtype]
    except KeyError as e:
        raise ValueError(
            f"Unsupported amp dtype string: {amp_dtype!r}. "
            "Use one of: fp16/float16/half, bf16/bfloat16, fp32/float32/float."
        ) from e
