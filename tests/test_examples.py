from __future__ import annotations

import os
import runpy
from pathlib import Path

import nbconvert
import nbformat
import pytest

# Make Jypyter happy
os.environ["JUPYTER_PLATFORM_DIRS"] = "1"

# Define the directory containing your example scripts
EXAMPLES_DIR = Path("docs/examples").resolve()


# Discover all Python files in the examples directory
example_scripts = list(EXAMPLES_DIR.glob("*.py"))

# Discover all Jupyter notebooks in the examples directory
example_notebooks = list(EXAMPLES_DIR.glob("*.ipynb"))


@pytest.mark.filterwarnings("ignore:'datfix' made the change")
@pytest.mark.parametrize("notebook", example_notebooks, ids=lambda nb: nb.name)
def test_example_notebook(notebook: Path, tmpdir):
    """Run Jupyter notebook and ensure it executes without errors."""

    # Convert notebook to a temporary Python script
    tmp_script_path = tmpdir / notebook.with_suffix(".py").name
    exporter = nbconvert.ScriptExporter()
    with notebook.open("r", encoding="utf-8") as f:
        notebook_node = nbformat.read(f, as_version=4)
    script_content, _ = exporter.from_notebook_node(notebook_node)

    with tmp_script_path.open("w", encoding="utf-8") as f:
        f.write(script_content)

    runpy.run_path(str(tmp_script_path))


@pytest.mark.filterwarnings("ignore:'datfix' made the change")
@pytest.mark.parametrize("script", example_scripts, ids=lambda s: s.name)
def test_example_script(script):
    """Run example script using runpy and ensure it executes without errors."""
    runpy.run_path(str(script))
