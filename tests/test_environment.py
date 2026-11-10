"""Stage 1: verify that the environment matches what the project expects."""

import platform
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
LOCK_FILE = ROOT / "requirements.txt"
DIRECT_DEPENDENCIES = [
    "numpy",
    "pandas",
    "xarray",
    "netcdf4",
    "scikit-learn",
    "matplotlib",
    "pytest",
]


def normalise(name: str) -> str:
    return name.strip().lower().replace("_", "-").replace(".", "-")


def read_pins() -> dict[str, str]:
    pins = {}
    for raw in LOCK_FILE.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, version = line.split("==", 1)
        pins[normalise(name)] = version.strip()
    return pins


def test_python_is_3_12():
    assert sys.version_info[:2] == (3, 12), sys.version


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS-only check")
def test_interpreter_is_native_arm64():
    assert platform.machine() == "arm64"


def test_package_is_importable():
    import xbt_uncertainty

    assert xbt_uncertainty.__version__ == "0.1.0"


def test_installed_versions_match_lock_file():
    assert LOCK_FILE.exists(), "requirements.txt is missing; run pip freeze"
    pins = read_pins()

    not_pinned = [name for name in DIRECT_DEPENDENCIES if name not in pins]
    assert not not_pinned, f"direct dependencies missing from lock file: {not_pinned}"

    mismatches = {}
    for name, pinned in pins.items():
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = None
        if installed != pinned:
            mismatches[name] = {"pinned": pinned, "installed": installed}
    assert not mismatches, mismatches


def test_netcdf_round_trip(tmp_path):
    import xarray as xr

    depth = np.array([0.0, 50.0, 100.0])
    temperature = np.array([20.5, 18.25, 10.0], dtype="float32")
    original = xr.Dataset(
        {"temperature": ("z", temperature, {"units": "degree_C"})},
        coords={"z": depth},
    )
    path = tmp_path / "round_trip.nc"
    original.to_netcdf(path, engine="netcdf4")

    with xr.open_dataset(path, engine="netcdf4") as restored:
        np.testing.assert_array_equal(restored["temperature"].values, temperature)
        np.testing.assert_array_equal(restored["z"].values, depth)
        assert restored["temperature"].attrs["units"] == "degree_C"


def test_decision_tree_fits_separable_data():
    from sklearn.tree import DecisionTreeClassifier

    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    model = DecisionTreeClassifier(random_state=0).fit(X, y)
    assert (model.predict(X) == y).all()


def test_matplotlib_saves_figure_headless(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    path = tmp_path / "figure.png"
    fig.savefig(path)
    plt.close(fig)
    assert path.stat().st_size > 0
