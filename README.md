![Tests](https://github.com/CIRADA-Tools/RM-tools/actions/workflows/python-package.yml/badge.svg) [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/CIRADA-Tools/RM-Tools/master.svg)](https://results.pre-commit.ci/latest/github/CIRADA-Tools/RM-Tools/master)

# RM-lite

A mini fork of RM-Tools - RM-synthesis, RM-clean and QU-fitting on polarised radio spectra.

This just exposes a Python API. No plotting, I/O utilities, or CLI are provided. See the main fork of [RM-Tools](https://github.com/CIRADA-Tools/RM-Tools) for that functionality.

The goal of this project is to provide low code surface area with high reliability, performance, and developer ergonomics.

## Installtion

```
pip install rmtools-lite
```


## Citing
If you use this package in a publication, please cite main fork's [ASCL entry](https://ui.adsabs.harvard.edu/abs/2020ascl.soft05003P/abstract) for the time being. 

## License
MIT

## Contributing
Contributions are welcome. Questions, bug reports, and feature requests can be posted to the GitHub issues page.

The development dependencies can be installed via `pip` from PyPI:
```bash
pip install "rmtools-lite[dev]"
```
or for a local clone:
```bash
cd rmtools-lite
pip install ".[dev]"
```

Code formatting and style is handled by `ruff`, with tests run by `pytest`. A `pre-commit` hook is available to handle the autoformatting. After installing the `dev` dependencies, you can install the hooks by running:
```bash
cd RM-Tools
pre-commit install
```
