from __future__ import annotations

import os

# Must be set before importing nbconvert/nbformat, which pull in jupyter_core
# and warn about path migration at import time otherwise.
os.environ["JUPYTER_PLATFORM_DIRS"] = "1"

import runpy
from pathlib import Path

import matplotlib.pyplot as plt
import nbconvert
import nbformat
import pytest


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Collect the examples from the rootdir, so the cwd does not decide."""
    examples = metafunc.config.rootpath / "docs" / "examples"
    for name, suffix in (("notebook", "*.ipynb"), ("script", "*.py")):
        if name in metafunc.fixturenames:
            found = sorted(examples.glob(suffix))
            metafunc.parametrize(name, found, ids=[p.name for p in found])


@pytest.mark.filterwarnings("ignore:'datfix' made the change")
def test_example_notebook(notebook: Path, tmp_path: Path):
    """Run an example notebook and ensure it executes without errors."""
    script_path = tmp_path / notebook.with_suffix(".py").name
    exporter = nbconvert.ScriptExporter()
    with notebook.open("r", encoding="utf-8") as f:
        notebook_node = nbformat.read(f, as_version=4)
    script_content, _ = exporter.from_notebook_node(notebook_node)
    script_path.write_text(script_content, encoding="utf-8")

    try:
        runpy.run_path(str(script_path))
    finally:
        plt.close("all")


@pytest.mark.filterwarnings("ignore:'datfix' made the change")
def test_example_script(script: Path):
    """Run an example script and ensure it executes without errors."""
    runpy.run_path(str(script))
