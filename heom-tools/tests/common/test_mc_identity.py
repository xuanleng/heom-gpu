"""Cross-language disorder_identity formatting (must match mc_realizations.cpp)."""
import h5py
import numpy as np
import pytest

from heom_tools.common.mc_identity import (
    disorder_identity_from_attrs,
    disorder_identity_from_draws,
    format_sigma_cm_for_identity,
)
from heom_tools.mc_disorder import generate as gen


def test_format_sigma_matches_cpp_ostream_default():
    # libstdc++ std::ostringstream default precision is 6.
    assert format_sigma_cm_for_identity([30.0]) == "30"
    assert format_sigma_cm_for_identity([42.478723404255316]) == "42.4787"
    fwhm_sigma = 100.0 * gen.SIGMA_PER_FWHM
    assert format_sigma_cm_for_identity([fwhm_sigma]) == "42.4661"


def test_disorder_identity_from_attrs_rqmc():
    sig = 100.0 * gen.SIGMA_PER_FWHM
    ident = disorder_identity_from_attrs(
        estimator="mean",
        generator="sobol_gauss@v1|repl=2",
        master_seed=11,
        sigma_cm=np.full(3, sig),
    )
    assert ident == (
        f"sobol_gauss@v1|repl=2|seed=11|sigma={format_sigma_cm_for_identity([sig])},"
        f"{format_sigma_cm_for_identity([sig])},{format_sigma_cm_for_identity([sig])}"
    )


def test_disorder_identity_from_draws_file(tmp_path):
    path = tmp_path / "d.h5"
    sigma = np.full(3, 30.0)
    with h5py.File(path, "w") as h:
        h.create_dataset("delta_eps", data=np.zeros((4, 3)))
        h.create_dataset("weight", data=np.full(4, 0.25))
        h.create_dataset("group", data=np.zeros(4, dtype=np.int32))
        h.attrs["estimator"] = "mean"
        h.attrs["generator"] = "sobol_gauss@v1|repl=2"
        h.attrs["master_seed"] = np.uint64(11)
        h.attrs["fwhm_cm"] = np.full(3, 100.0)
        h.attrs["sigma_cm"] = sigma
    assert disorder_identity_from_draws(str(path)) == (
        "sobol_gauss@v1|repl=2|seed=11|sigma=30,30,30"
    )


def test_disorder_identity_quadrature_requires_level():
    with pytest.raises(ValueError):
        disorder_identity_from_attrs(
            estimator="quadrature",
            generator="sparse_gh@v1",
            master_seed=0,
            sigma_cm=[1.0],
            sg_level=None,
        )
