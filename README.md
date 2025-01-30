# RM-lite

[![Actions Status][actions-badge]][actions-link]
[![Codecov Status][codecov-badge]][codecov-link]
[![Documentation Status][rtd-badge]][rtd-link]

[![PyPI version][pypi-version]][pypi-link]

<!-- [![Conda-Forge][conda-badge]][conda-link] -->

[![PyPI platforms][pypi-platforms]][pypi-link]

<!-- [![GitHub Discussion][github-discussions-badge]][github-discussions-link] -->

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[codecov-link]:             https://codecov.io/gh/AlecThomson/rm-lite
[codecov-badge]:            https://codecov.io/gh/AlecThomson/rm-lite/graph/badge.svg?token=7EARBRN20D
[actions-badge]:            https://github.com/AlecThomson/rm-lite/workflows/CI/badge.svg
[actions-link]:             https://github.com/AlecThomson/rm-lite/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/rm-lite
[conda-link]:               https://github.com/conda-forge/rm-lite-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/AlecThomson/rm-lite/discussions
[pypi-link]:                https://pypi.org/project/rm-lite/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/rm-lite
[pypi-version]:             https://img.shields.io/pypi/v/rm-lite
[rtd-badge]:                https://readthedocs.org/projects/rm-lite/badge/?version=latest
[rtd-link]:                 https://rm-lite.readthedocs.io/en/latest/?badge=latest

<!-- prettier-ignore-end -->

A mini fork of RM-Tools - RM-synthesis, RM-clean and QU-fitting on polarised
radio spectra.

This just exposes a Python API. No plotting, I/O utilities, or CLI are provided.
See the main fork of [RM-Tools](https://github.com/CIRADA-Tools/RM-Tools) for
that functionality.

The goal of this project is to provide low code surface area with high
reliability, performance, and developer ergonomics.

_**Warning:** This is very much a work-in-progress. Do not expect stability for
a while._

## Installation

PyPI release:

```
pip install rm-lite
```

Current GitHub `main` commit:

```
pip install git+https://github.com/AlecThomson/rm-lite.git
```

## Citing

If you use this package in a publication, please cite main fork's
[ASCL entry](https://ui.adsabs.harvard.edu/abs/2020ascl.soft05003P/abstract) for
the time being.

## License

MIT

## Contributing

Contributions are welcome. Questions, bug reports, and feature requests can be
posted to the GitHub issues page.

The development dependencies can be installed via `pip` from PyPI:

```bash
pip install "rm-lite[dev]"
```

or for a local clone:

```bash
cd rm-lite
pip install ".[dev]"
```

Code formatting and style is handled by `ruff`, with tests run by `pytest`. A
`pre-commit` hook is available to handle the autoformatting. After installing
the `dev` dependencies, you can install the hooks by running:

```bash
cd rm-lite
pre-commit install
```
