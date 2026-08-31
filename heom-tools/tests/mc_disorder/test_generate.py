"""Pure generator tests (numpy + h5py only; no solver / GPU needed).

The solver-dependent MC tests (sigma>0 reproducibility, error shrink) live on
the solver side in heom-gpu/tests/test_mc_disorder.py,
since they require the heom_solver binary.
"""
import os
import subprocess
import sys

import h5py
import numpy as np
import pytest

from heom_tools.mc_disorder import generate as gen

GEN = [sys.executable, "-m", "heom_tools.mc_disorder"]


def _gen(tmp, **kw):
    os.makedirs(tmp, exist_ok=True)
    out = os.path.join(tmp, "d.h5")
    cmd = GEN + ["--out", out, "--n-node", str(kw.get("n", 8)),
                 "--n-site", str(kw.get("s", 7)), "--fwhm", str(kw.get("fwhm", 100.0)),
                 "--seed", str(kw.get("seed", 42)), "--method", "iid"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"generator failed:\n{r.stderr}"
    return out


def test_schema_and_shapes(tmp_path):
    with h5py.File(_gen(str(tmp_path), n=8, s=7), "r") as h:
        assert h["delta_eps"].shape == (8, 7)
        assert h["weight"].shape == (8,)
        assert h["group"].shape == (8,)
        assert abs(h["weight"][:].sum() - 1.0) < 1e-12   # sums to 1
        assert h.attrs["estimator"] == "mean"
        assert h.attrs["generator"] == "iid_gauss@v1"
        np.testing.assert_allclose(h.attrs["sigma_cm"], 100.0 / 2.354820045031, rtol=1e-9)


def test_reproducible_same_seed(tmp_path):
    a = _gen(str(tmp_path / "a"), seed=7)
    b = _gen(str(tmp_path / "b"), seed=7)
    with h5py.File(a) as ha, h5py.File(b) as hb:
        np.testing.assert_array_equal(ha["delta_eps"][:], hb["delta_eps"][:])


def test_append_preserves_old_rows(tmp_path):
    # The first 8 rows are a prefix of the 16-row file.
    s8 = _gen(str(tmp_path / "a"), n=8, seed=9)
    s16 = _gen(str(tmp_path / "b"), n=16, seed=9)
    with h5py.File(s8) as h8, h5py.File(s16) as h16:
        np.testing.assert_array_equal(h8["delta_eps"][:], h16["delta_eps"][:8])


def test_sigma_zero(tmp_path):
    out = _gen(str(tmp_path), fwhm=0.0)
    with h5py.File(out) as h:
        assert np.all(h["delta_eps"][:] == 0.0)


def test_extend_preserves_prefix_and_params(tmp_path):
    base = os.path.join(tmp_path, "d.h5")
    subprocess.run(GEN + ["--out", base, "--n-node", "8",
                    "--n-site", "7", "--fwhm", "120.0", "--per-site", "3:140",
                    "--seed", "55", "--method", "iid"], check=True)
    ext = os.path.join(tmp_path, "d16.h5")
    # Only --n-node needed; seed/fwhm/n_site inherited from base.
    subprocess.run(GEN + ["--extend", base, "--n-node", "16",
                    "--out", ext], check=True)
    with h5py.File(base) as b, h5py.File(ext) as e:
        assert e["delta_eps"].shape == (16, 7)
        np.testing.assert_array_equal(b["delta_eps"][:], e["delta_eps"][:8])
        assert int(e.attrs["master_seed"]) == 55
        np.testing.assert_array_equal(np.array(b.attrs["fwhm_cm"]),
                                      np.array(e.attrs["fwhm_cm"]))


def test_extend_default_out_overwrites_source(tmp_path):
    base = os.path.join(tmp_path, "d.h5")
    subprocess.run(GEN + ["--out", base, "--n-node", "4",
                    "--n-site", "3", "--fwhm", "80.0", "--seed", "9",
                    "--method", "iid"], check=True)
    with h5py.File(base) as b:
        first4 = b["delta_eps"][:].copy()
    # No --out → overwrite the extended file in place; prefix must survive.
    subprocess.run(GEN + ["--extend", base, "--n-node", "8"], check=True)
    with h5py.File(base) as b:
        assert b["delta_eps"].shape == (8, 3)
        np.testing.assert_array_equal(b["delta_eps"][:4], first4)


def test_print_params(tmp_path):
    base = os.path.join(tmp_path, "d.h5")
    subprocess.run(GEN + ["--out", base, "--n-node", "5",
                    "--n-site", "7", "--fwhm", "100.0", "--seed", "42",
                    "--method", "iid"], check=True)
    r = subprocess.run(GEN + ["--print-params", base],
                       capture_output=True, text=True, check=True)
    assert "master_seed" in r.stdout and "42" in r.stdout
    assert "n_site" in r.stdout and "7" in r.stdout


# --------------------------------------------------------------------------
# Antithetic pairs. Pure functions.
# --------------------------------------------------------------------------

def test_antithetic_pairs_are_negatives():
    """KILLER: row 2i+1 == -(row 2i) exactly, for every pair."""
    de = gen.generate_antithetic(8, 7, np.full(7, 42.0), master_seed=3)
    assert de.shape == (8, 7)
    for i in range(4):
        np.testing.assert_array_equal(de[2 * i + 1], -de[2 * i])


def test_antithetic_odd_moments_cancel_exactly():
    """KILLER: the sum of every column is 0 to machine precision (each ±ξ pair
    cancels), so any odd-in-ξ response averages out regardless of N."""
    de = gen.generate_antithetic(16, 5, np.full(5, 100.0), master_seed=11)
    col_sum = de.sum(axis=0)
    np.testing.assert_allclose(col_sum, 0.0, atol=1e-9)


def test_antithetic_first_component_matches_iid_pairing():
    """The ξ side of pair i uses SeedSequence([seed, i]); regeneration is
    bit-identical (pure function of (seed, pair))."""
    a = gen.generate_antithetic(8, 4, np.full(4, 90.0), master_seed=7)
    b = gen.generate_antithetic(8, 4, np.full(4, 90.0), master_seed=7)
    np.testing.assert_array_equal(a, b)


def test_antithetic_append_preserves_pairs():
    """Pair i = f(seed, i): the first N/2 pairs are a prefix of a larger file."""
    small = gen.generate_antithetic(8, 4, np.full(4, 90.0), master_seed=7)
    big = gen.generate_antithetic(16, 4, np.full(4, 90.0), master_seed=7)
    np.testing.assert_array_equal(small, big[:8])


def test_antithetic_odd_n_rejected():
    import pytest
    with pytest.raises(ValueError):
        gen.generate_antithetic(7, 4, np.full(4, 90.0), master_seed=1)


def test_antithetic_cli_schema(tmp_path):
    """CLI writes a mean-estimator file (solver-transparent) with the new
    generator tag and w=1/N."""
    out = os.path.join(tmp_path, "a.h5")
    r = subprocess.run(GEN + ["--out", out, "--n-node", "8",
                        "--n-site", "5", "--fwhm", "100.0", "--seed", "13",
                        "--method", "antithetic"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    with h5py.File(out) as h:
        assert h.attrs["estimator"] == "mean"
        assert h.attrs["generator"] == "antithetic_gauss@v1"
        assert h["delta_eps"].shape == (8, 5)
        assert abs(h["weight"][:].sum() - 1.0) < 1e-12
        np.testing.assert_allclose(h["delta_eps"][:].sum(axis=0), 0.0, atol=1e-9)


def test_antithetic_cli_odd_rejected(tmp_path):
    out = os.path.join(tmp_path, "a.h5")
    r = subprocess.run(GEN + ["--out", out, "--n-node", "7",
                        "--n-site", "5", "--fwhm", "100.0", "--method", "antithetic"],
                       capture_output=True, text=True)
    assert r.returncode != 0
    assert "even" in r.stderr.lower()


def test_antithetic_cli_extend_rejected(tmp_path):
    out = os.path.join(tmp_path, "a.h5")
    subprocess.run(GEN + ["--out", out, "--n-node", "8",
                    "--n-site", "5", "--fwhm", "100.0", "--method", "antithetic"],
                   check=True)
    r = subprocess.run(GEN + ["--extend", out, "--n-node", "16"],
                       capture_output=True, text=True)
    assert r.returncode != 0


# --------------------------------------------------------------------------
# sparse_gh (Smolyak Gauss–Hermite) quadrature sampler
# --------------------------------------------------------------------------

def test_hermegauss_1d_normalised():
    """1D rule integrates the N(0,1) measure exactly: Σw=1, mean 0, var 1."""
    for n in (1, 2, 3, 5, 8):
        x, w = gen.hermegauss_1d(n)
        assert abs(w.sum() - 1.0) < 1e-13
        assert abs((w * x).sum()) < 1e-12
        if n >= 2:  # var exact once the rule handles degree 2 (2n-1 >= 2)
            assert abs((w * x**2).sum() - 1.0) < 1e-12


def test_sparse_gh_level0_is_centre():
    """ℓ=0 collapses to a single centre node with weight 1."""
    de, w = gen.sparse_gh(5, 0, np.full(5, 100.0))
    assert de.shape == (1, 5)
    assert np.allclose(de, 0.0)
    assert abs(w[0] - 1.0) < 1e-15


def test_sparse_gh_weights_sum_to_one():
    """Σw=1 at every level (Smolyak integrates the constant exactly)."""
    for level in (0, 1, 2, 3):
        _, w = gen.sparse_gh(4, level, np.full(4, 100.0))
        assert abs(w.sum() - 1.0) < 1e-12


def test_sparse_gh_moments_exact():
    """KILLER TEST: ∫ξ dN=0, ∫ξ² dN=σ², ∫ξ⁴ dN=3σ⁴ per site, to machine
    precision. ℓ=2 → exact to total polynomial degree 2ℓ+1 = 5 (≥ 4)."""
    sigma = np.array([80.0, 120.0, 100.0])
    de, w = gen.sparse_gh(3, 2, sigma)
    assert abs(w.sum() - 1.0) < 1e-12
    for j in range(3):
        xj = de[:, j]
        assert abs((w * xj).sum()) < 1e-8 * sigma[j]
        assert abs((w * xj**2).sum() - sigma[j] ** 2) < 1e-9 * sigma[j] ** 2
        assert abs((w * xj**4).sum() - 3 * sigma[j] ** 4) < 1e-9 * sigma[j] ** 4


def test_sparse_gh_cross_moments_zero():
    """Independent dims: ∫ξ_iξ_j dN = 0 (i≠j), exact for ℓ≥1."""
    sigma = np.array([100.0, 90.0, 110.0])
    de, w = gen.sparse_gh(3, 2, sigma)
    for i in range(3):
        for j in range(i + 1, 3):
            assert abs((w * de[:, i] * de[:, j]).sum()) < 1e-6 * sigma[i] * sigma[j]


def test_sparse_gh_reproducible():
    """Pure function of (n_site, level, sigma): re-generation is bit-identical."""
    sig = np.array([100.0, 90.0])
    a_de, a_w = gen.sparse_gh(2, 3, sig)
    b_de, b_w = gen.sparse_gh(2, 3, sig)
    np.testing.assert_array_equal(a_de, b_de)
    np.testing.assert_array_equal(a_w, b_w)


def test_sparse_gh_cli_schema(tmp_path):
    """CLI writes the quadrature subset. The group column is 0."""
    out = os.path.join(tmp_path, "q.h5")
    r = subprocess.run(GEN + ["--out", out, "--n-site", "4",
                        "--fwhm", "100.0", "--method", "sparse_gh", "--level", "2"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    with h5py.File(out) as h:
        assert h.attrs["estimator"] == "quadrature"
        assert h.attrs["generator"] == "sparse_gh@v1"
        assert int(h.attrs["sg_level"]) == 2
        assert int(h.attrs["master_seed"]) == 0
        assert np.all(h["group"][:] == 0)
        assert h["delta_eps"].shape[1] == 4
        assert h["weight"].shape[0] == h["delta_eps"].shape[0]
        assert abs(h["weight"][:].sum() - 1.0) < 1e-12


def test_sparse_gh_extend_rejected(tmp_path):
    """Refinement = new file; --extend must refuse quadrature."""
    out = os.path.join(tmp_path, "q.h5")
    subprocess.run(GEN + ["--out", out, "--n-site", "3",
                    "--fwhm", "100.0", "--method", "sparse_gh", "--level", "1"],
                   check=True)
    r = subprocess.run(GEN + ["--extend", out, "--n-node", "10"],
                       capture_output=True, text=True)
    assert r.returncode != 0
    assert "iid" in r.stderr.lower() or "extend" in r.stderr.lower()


# --------------------------------------------------------------------------
# RQMC (scrambled Sobol) generator — Method B / Slice 4 de-risk probe
# --------------------------------------------------------------------------

def test_rqmc_shape_and_scale():
    sig = np.full(5, 40.0)
    d = gen.generate_rqmc(256, 5, sig, master_seed=7, replica=0)
    assert d.shape == (256, 5)
    # Scrambled Sobol → each column ~ N(0, σ²): mean≈0, std≈σ.
    assert np.all(np.abs(d.mean(axis=0)) < 6.0)          # ≪ σ=40
    assert np.allclose(d.std(axis=0), 40.0, rtol=0.15)


def test_rqmc_reproducible():
    sig = np.full(4, 30.0)
    a = gen.generate_rqmc(128, 4, sig, master_seed=11, replica=2)
    b = gen.generate_rqmc(128, 4, sig, master_seed=11, replica=2)
    np.testing.assert_array_equal(a, b)


def test_rqmc_replicas_differ():
    sig = np.full(4, 30.0)
    a = gen.generate_rqmc(128, 4, sig, master_seed=11, replica=0)
    b = gen.generate_rqmc(128, 4, sig, master_seed=11, replica=1)
    assert not np.allclose(a, b)          # independent scrambles


def test_rqmc_prefix_is_append_safe():
    # The probe reads N-point estimates as PREFIXES of one Sobol pool, so the
    # first N rows of an M-point set MUST equal a fresh N-point set (same seed/
    # replica). This is what makes the convergence sweep a single pool.
    sig = np.full(6, 25.0)
    big = gen.generate_rqmc(256, 6, sig, master_seed=3, replica=5)
    small = gen.generate_rqmc(64, 6, sig, master_seed=3, replica=5)
    np.testing.assert_array_equal(big[:64], small)


def test_rqmc_lower_discrepancy_than_iid():
    # The whole point: Sobol covers the cube more evenly than iid. Compare the
    # 2D star-discrepancy proxy — max gap of the sorted 1D projections — Sobol
    # must beat iid on average across dimensions at the same N.
    sig = np.ones(4)
    N = 256
    rq = gen.generate_rqmc(N, 4, sig, master_seed=1, replica=0)
    iid = gen.generate_iid(N, 4, sig, master_seed=1)
    from scipy.stats import norm
    def max_gap(x):                       # in uniform space
        u = np.sort(norm.cdf(x))
        return np.max(np.diff(np.concatenate([[0.0], u, [1.0]])))
    rq_gap = np.mean([max_gap(rq[:, k]) for k in range(4)])
    iid_gap = np.mean([max_gap(iid[:, k]) for k in range(4)])
    assert rq_gap < iid_gap               # Sobol fills gaps better


def test_rqmc_cli_schema(tmp_path):
    out = os.path.join(tmp_path, "s.h5")
    r = subprocess.run(GEN + ["--out", out, "--n-node", "64",
                        "--n-site", "4", "--fwhm", "100.0", "--seed", "9",
                        "--method", "rqmc", "--replica", "3"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    with h5py.File(out) as h:
        assert h.attrs["estimator"] == "mean"
        # repl rides IN the generator tag → distinct disorder_identity per
        # replica (C++ builds "<generator>|seed=..|sigma=.."), so two replicas
        # never share a merge key and can't be silently pooled into a bogus √N.
        assert h.attrs["generator"] == "sobol_gauss@v1|repl=3"
        assert np.all(h["group"][:] == 3)          # replica id also in group
        assert h["delta_eps"].shape == (64, 4)
        assert abs(h["weight"][:].sum() - 1.0) < 1e-12


def test_rqmc_distinct_replicas_get_distinct_identity(tmp_path):
    # The anti-footgun: two replicas MUST carry different generator tags so their
    # merge keys differ and mc-disorder-merge/las-post can never silently pool
    # them into one bogus √N mean (they must be combined via --combine-replicas).
    tags = []
    for rep in (0, 1, 2):
        out = os.path.join(tmp_path, f"r{rep}.h5")
        subprocess.run(GEN + ["--out", out, "--n-node", "32",
                        "--n-site", "3", "--fwhm", "100.0", "--seed", "9",
                        "--method", "rqmc", "--replica", str(rep)], check=True)
        with h5py.File(out) as h:
            tags.append(str(h.attrs["generator"]))
    assert len(set(tags)) == 3          # all distinct
    assert tags == ["sobol_gauss@v1|repl=0", "sobol_gauss@v1|repl=1",
                    "sobol_gauss@v1|repl=2"]


def test_rqmc_cli_extend_rejected(tmp_path):
    out = os.path.join(tmp_path, "s.h5")
    subprocess.run(GEN + ["--out", out, "--n-node", "32",
                    "--n-site", "3", "--fwhm", "100.0", "--method", "rqmc"],
                   check=True)
    r = subprocess.run(GEN + ["--extend", out, "--n-node", "64"],
                       capture_output=True, text=True)
    assert r.returncode != 0


def test_rqmc_n_replicas_writes_files_and_manifest(tmp_path):
    import yaml
    from heom_tools.mc_disorder.manifest import read_rqmc_manifest

    stem = os.path.join(tmp_path, "nodes_rqmc")
    r = subprocess.run(
        GEN + ["--out", stem, "--n-node", "32",
         "--n-site", "3", "--fwhm", "100.0", "--seed", "9",
         "--method", "rqmc", "--n-replicas", "4"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    tags = []
    for repl in range(4):
        path = f"{stem}_r{repl}.h5"
        assert os.path.isfile(path)
        with h5py.File(path) as h:
            tags.append(str(h.attrs["generator"]))
            assert h.attrs["estimator"] == "mean"
            assert h["delta_eps"].shape == (32, 3)
    assert tags == [f"sobol_gauss@v1|repl={i}" for i in range(4)]
    manifest = f"{stem}.manifest.yaml"
    assert os.path.isfile(manifest)
    with open(manifest, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    assert doc["method"] == "rqmc"
    assert doc["n_replicas"] == 4
    assert doc["master_seed"] == 9
    assert len(doc["draws_files"]) == 4
    assert doc["draws_files"] == [f"nodes_rqmc_r{i}.h5" for i in range(4)]
    got = read_rqmc_manifest(manifest)
    assert len(got["draws_files"]) == 4


def test_rqmc_n_replicas_no_manifest(tmp_path):
    stem = os.path.join(tmp_path, "nodes_rqmc")
    r = subprocess.run(
        GEN + ["--out", stem, "--n-node", "16",
         "--n-site", "2", "--fwhm", "80.0", "--seed", "1",
         "--method", "rqmc", "--n-replicas", "2", "--no-manifest"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert os.path.isfile(f"{stem}_r0.h5")
    assert os.path.isfile(f"{stem}_r1.h5")
    assert not os.path.isfile(f"{stem}.manifest.yaml")


def test_rqmc_n_replicas_mutually_exclusive_with_replica(tmp_path):
    stem = os.path.join(tmp_path, "nodes_rqmc")
    r = subprocess.run(
        GEN + ["--out", stem, "--n-node", "16",
         "--n-site", "2", "--fwhm", "80.0", "--method", "rqmc",
         "--n-replicas", "2", "--replica", "1"],
        capture_output=True, text=True,
    )
    assert r.returncode != 0
    assert "mutually exclusive" in r.stderr.lower()


def test_rqmc_n_replicas_only_rqmc(tmp_path):
    out = os.path.join(tmp_path, "d.h5")
    r = subprocess.run(
        GEN + ["--out", out, "--n-node", "8",
         "--n-site", "3", "--fwhm", "100.0", "--method", "iid",
         "--n-replicas", "2"],
        capture_output=True, text=True,
    )
    assert r.returncode != 0
    assert "rqmc" in r.stderr.lower()


def test_rqmc_single_file_backward_compat_default_replica(tmp_path):
    out = os.path.join(tmp_path, "s.h5")
    r = subprocess.run(
        GEN + ["--out", out, "--n-node", "32",
         "--n-site", "3", "--fwhm", "100.0", "--seed", "9",
         "--method", "rqmc"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    with h5py.File(out) as h:
        assert h.attrs["generator"] == "sobol_gauss@v1|repl=0"


def test_iid_default_emits_rqmc_production_note(tmp_path):
    out = os.path.join(tmp_path, "d.h5")
    r = subprocess.run(
        GEN + ["--out", out, "--n-node", "8",
         "--n-site", "3", "--fwhm", "100.0", "--seed", "1", "--method", "iid"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "rqmc" in r.stderr.lower()


def test_rqmc_n_replicas_rollback_on_failure(tmp_path, monkeypatch):
    stem = os.path.join(tmp_path, "nodes_rqmc")

    def fail_manifest(*_a, **_k):
        raise RuntimeError("manifest write failed")

    monkeypatch.setattr(gen, "_write_rqmc_manifest", fail_manifest)
    with pytest.raises(RuntimeError, match="manifest write failed"):
        gen.main([
            "--out", stem, "--n-node", "16", "--n-site", "3",
            "--fwhm", "100.0", "--seed", "1", "--method", "rqmc",
            "--n-replicas", "3",
        ])
    for r in range(3):
        assert not os.path.isfile(f"{stem}_r{r}.h5")
    assert not os.path.isfile(f"{stem}.manifest.yaml")
