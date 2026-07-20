"""Guard that public-tool keyword defaults stay in sync with the shared Options
dataclass field defaults they are packed into.

The public 1D/3D entry points keep flat keyword arguments for a friendly call
signature, then pack them into `FDFOptions`/`StokesIFitOptions`/`RMCleanOptions`
on the first line. That duplicates each default in two places; these tests fail
if the two ever drift apart.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any

import pytest
from rm_lite.tools_1d.rmsynth import run_rmsynth
from rm_lite.tools_3d.rmclean import rmclean_3d
from rm_lite.tools_3d.rmsynth import rmsynth_3d
from rm_lite.utils.clean import RMCleanOptions
from rm_lite.utils.fitting import StokesIFitOptions
from rm_lite.utils.synthesis import FDFOptions


def _field_defaults(cls: Any) -> dict[str, Any]:
    return {
        f.name: f.default
        for f in dataclasses.fields(cls)
        if f.default is not dataclasses.MISSING
    }


def _param_defaults(func: Any) -> dict[str, Any]:
    return {
        name: p.default
        for name, p in inspect.signature(func).parameters.items()
        if p.default is not inspect.Parameter.empty
    }


# (public function, options class, {param_name: field_name} for params that map).
# fit_function->fit_function, stokes_i_snr_cut->snr_cut, etc.
CASES = [
    (
        run_rmsynth,
        FDFOptions,
        {
            "phi_max_radm2": "phi_max_radm2",
            "d_phi_radm2": "d_phi_radm2",
            "n_samples": "n_samples",
            "weight_type": "weight_type",
            "do_fit_rmsf": "do_fit_rmsf",
            "do_fit_rmsf_real": "do_fit_rmsf_real",
        },
    ),
    (
        run_rmsynth,
        StokesIFitOptions,
        {"fit_order": "fit_order", "fit_function": "fit_function"},
    ),
    (
        rmsynth_3d,
        FDFOptions,
        {
            "phi_max_radm2": "phi_max_radm2",
            "d_phi_radm2": "d_phi_radm2",
            "n_samples": "n_samples",
            "weight_type": "weight_type",
        },
    ),
    (
        rmsynth_3d,
        StokesIFitOptions,
        {
            "fit_order": "fit_order",
            "fit_function": "fit_function",
            "stokes_i_snr_cut": "snr_cut",
            "compute_model_error": "compute_model_error",
            "n_error_samples": "n_error_samples",
        },
    ),
    (rmclean_3d, RMCleanOptions, {"max_iter": "max_iter", "gain": "gain"}),
]


@pytest.mark.parametrize(("func", "options_cls", "mapping"), CASES)
def test_public_defaults_match_options(
    func: Any, options_cls: Any, mapping: dict[str, str]
) -> None:
    params = _param_defaults(func)
    fields = _field_defaults(options_cls)
    for param_name, field_name in mapping.items():
        assert params[param_name] == fields[field_name], (
            f"{func.__name__}.{param_name} default {params[param_name]!r} != "
            f"{options_cls.__name__}.{field_name} default {fields[field_name]!r}"
        )
