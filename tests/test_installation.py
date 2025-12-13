"""Test installation."""

from __future__ import annotations

import importlib
import tomllib

from mpnn.env import PROJECT_ROOT_DIR

PYPROJECT_PATH = PROJECT_ROOT_DIR / "pyproject.toml"


def _load_pyproject() -> dict:
    with PYPROJECT_PATH.open("rb") as pyproject_file:
        return tomllib.load(pyproject_file)


def test_project_metadata_matches_pyproject() -> None:
    """Test project metadata matches pyproject."""
    project = _load_pyproject()["project"]
    assert project["name"] == "mpnn"
    assert project["requires-python"] == ">=3.12"
    assert project["license"]["file"] == "LICENSE"
    assert project["readme"] == "README.md"
    assert project["description"] == "CS224W final project"
    assert project["dynamic"] == ["version"]


def test_core_dependencies_declared() -> None:
    """Test core dependencies declared."""
    project = _load_pyproject()["project"]
    dependencies = set(project["dependencies"])
    required = {
        "biopython",
        "fire",
        "hydra-core",
        "jaxtyping",
        "jupyter",
        "matplotlib",
        "numpy",
        "pandas",
        "prody",
        "python-dateutil",
        "scipy",
        "seaborn",
        "torch-cluster @ https://data.pyg.org/whl/torch-2.8.0+cu128/torch_cluster-1.6.3+pt28cu128-cp312-cp312-linux_x86_64.whl",
        "torch==2.8.0",
        "torch_geometric",
        "tqdm",
        "transformers",
        "typer",
        "uv",
        "wandb",
    }
    missing = sorted(required - dependencies)
    assert not missing, f"Missing mandatory dependencies: {missing}"


def test_dev_extras_cover_pytest_stack() -> None:
    """Test dev extras cover pytest stack."""
    dev_extras = _load_pyproject()["project"]["optional-dependencies"]["dev"]
    required = {
        "pre-commit",
        "pytest",
        "pytest-cov",
        "pytest-env",
        "pytest-html",
        "pytest-timeout",
        "pytest-xdist",
        "ruff",
    }
    missing = sorted(required - set(dev_extras))
    assert not missing, f"Missing dev dependencies: {missing}"


def test_dynamic_version_attr_points_to_package_version() -> None:
    """Test dynamic version attr points to package version."""
    config = _load_pyproject()
    attr_path = config["tool"]["setuptools"]["dynamic"]["version"]["attr"]
    module_path, _, attribute = attr_path.partition(".")
    module = importlib.import_module(module_path)
    version_value = getattr(module, attribute, "")
    assert version_value, "Package version attribute must not be empty"
