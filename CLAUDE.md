# rm-lite

Mini reimplementation of RM-Tools. Fully typed Python, mypy strict on
`rm_lite/*`.

## Data (this repo only — overrides global caution)

Root FITS cutouts and `scratch.ipynb` are dev/test fixtures, not SKA proprietary
data. OK to read/run directly. Same for anything under `tests/` or
`rm_lite/data/`.

## Environment

Use `uv run` for everything — never call `pytest`/`mypy`/`ruff` bare.

## Verify gate (run before calling any task done)

```
uv run pytest
uv run prek run --all-files
```

Full test suite every time, not just touched modules — dask/zarr tests are slow
but regressions there are easy to miss with partial runs.

## Docs

`docs/` uses sphinx-autoapi + nbsphinx. When a public function/param in
`rm_lite/` changes shape, update the relevant docstring/.rst/example in the same
change — don't leave it for a follow-up.

## Code style

- Docstrings and comments: terse, one-liners (`"""Does X"""`), same for
  NamedTuple field docstrings. No numpydoc/Google sections unless surrounding
  code already has them.
- Write like a human, not a manual. No over-explaining, no em/en dashes anywhere
  (docs, comments, commit messages, prose).
- `from __future__ import annotations` required in every module (ruff isort
  enforces).
- Python must be fully typed. mypy strict + `disallow_untyped_defs` on
  `rm_lite.*`.
- Ruff ruleset is wide (see `pyproject.toml`); don't add per-file ignores
  without asking.
- Prefer plain Options/Results containers (NamedTuple/dataclass) over full OOP
  with methods and inheritance. Functions take an Options object, return a
  Results object. Reach for a class-with-methods only when state genuinely needs
  to persist across calls.

## Documentation must be self-checking

Example notebooks and docs (`docs/`, `*.ipynb`) must actually run and assert
their own claims, not just show code with commentary. If a doc says "returns X",
it should execute and check X, not describe it and move on. Stale docs are worse
than no docs.
