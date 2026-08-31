"""Shared mc-probe orchestration helpers (solver + GPU path)."""
import glob
import os

import numpy as np

from . import config_io as cfg_io
from .probe_core import l2_relative_diff

def _write_nodes_sparse_gh(path, n_site, level, sigma_cm):
    from . import generate as gen
    delta, weight = gen.sparse_gh(n_site, level, sigma_cm)
    group = np.zeros(len(weight), dtype=np.int32)
    fwhm = sigma_cm / gen.SIGMA_PER_FWHM
    gen.write_file(path, delta, weight, group, estimator="quadrature",
                   generator="sparse_gh@v1", master_seed=0, fwhm_cm=fwhm,
                   sigma_cm=sigma_cm, sg_level=level)
    return len(weight)


def _write_nodes_iid(path, n_site, n_node, sigma_cm, seed):
    from . import generate as gen
    delta = gen.generate_iid(n_node, n_site, sigma_cm, seed)
    weight = np.full(n_node, 1.0 / n_node, dtype=np.float64)
    group = np.zeros(n_node, dtype=np.int32)
    fwhm = sigma_cm / gen.SIGMA_PER_FWHM
    gen.write_file(path, delta, weight, group, estimator="mean",
                   generator="iid_gauss@v1", master_seed=seed, fwhm_cm=fwhm,
                   sigma_cm=sigma_cm)
    return n_node


def _write_nodes_antithetic(path, n_site, n_node, sigma_cm, seed):
    from . import generate as gen
    delta = gen.generate_antithetic(n_node, n_site, sigma_cm, seed)
    weight = np.full(n_node, 1.0 / n_node, dtype=np.float64)
    group = np.zeros(n_node, dtype=np.int32)
    fwhm = sigma_cm / gen.SIGMA_PER_FWHM
    gen.write_file(path, delta, weight, group, estimator="mean",
                   generator="antithetic_gauss@v1", master_seed=seed,
                   fwhm_cm=fwhm, sigma_cm=sigma_cm)
    return n_node


def _write_single_node(path, delta_row, sigma_cm, seed):
    """One-row mean file (weight 1) — the unit of a per-node arena run."""
    from . import generate as gen
    delta = np.asarray(delta_row, dtype=np.float64)[None, :]
    fwhm = sigma_cm / gen.SIGMA_PER_FWHM
    gen.write_file(path, delta, np.array([1.0]), np.zeros(1, dtype=np.int32),
                   estimator="mean", generator="iid_gauss@v1", master_seed=seed,
                   fwhm_cm=fwhm, sigma_cm=sigma_cm)


def _load_h0_mu(base_cfg):
    """Base excited-state Hamiltonian H0 [n,n] and per-site dipoles μ [n,3]
    (= dipoles × strengths) — the surrogate's only physics inputs."""
    return cfg_io.load_h0_mu(cfg_io.load_doc(base_cfg))


def _cfg_for_nodes(base_cfg, draws_path, save_path, work_dir, task="absorption"):
    """Write a temp config copy pointing at this node set's draws + save_path."""
    return cfg_io.write_probe_cfg(
        base_cfg,
        draws_path=draws_path,
        save_path=save_path,
        work_dir=work_dir,
        task=task,
    )


def _n_site_of(base_cfg):
    return cfg_io.read_n_states(base_cfg)


def _sigma_cm(base_cfg, n_site, fwhm, per_site=""):
    """Per-site σ (cm⁻¹) from a global FWHM + optional 1-based overrides
    (e.g. "3:50" for FMO's site-3 = 50 cm⁻¹, rest = fwhm). Single source is
    the generator's parse_per_site — the disorder model MUST match the reference
    (the production draws file is site3=50/oth100)."""
    from . import generate as gen
    fwhm_cm = gen.parse_per_site(per_site, n_site, fwhm)
    return fwhm_cm * gen.SIGMA_PER_FWHM


def _merge_and_spectrum(save_path, wmin, wmax, w0, normalized=True):
    """Merge the shard group in save_path and return (ω, intensity).

    `normalized=True` returns the once-normalised spectrum (smoothness probe, shape
    comparison). `normalized=False` returns the RAW intensity — required by the
    control-variate arena: per-node normalisation is nonlinear and would break
    mean(O_i) linearity, so CV runs on raw intensity throughout."""
    from heom_tools.common.mc_merge import merge_mc_shards, write_merged_spectrum
    from heom_tools.common.spectrum import load_spectrum_h5
    from heom_tools.absorption.post.core import compute_absorption_post
    from heom_tools.absorption.post.io import resolve_frequency_window
    shards = sorted(glob.glob(os.path.join(save_path, "*_rank*.h5")))
    if not shards:
        raise RuntimeError(f"no shards in {save_path} (did the solver run?)")
    merged = merge_mc_shards(shards, "rho_xyz")
    merged_path = os.path.join(save_path, "merged.h5")
    write_merged_spectrum(shards[0], merged_path, merged, "rho_xyz")
    spec = load_spectrum_h5(merged_path)
    if w0 is not None:
        from dataclasses import replace
        spec = replace(spec, w0_cm=float(w0))
    omega_min, omega_max = resolve_frequency_window(wmin, wmax, spec)
    res = compute_absorption_post(spec, wmin_cm=omega_min, wmax_cm=omega_max,
                                  strict_ground_w0=(w0 is None))
    inten = res.intensity_total_norm if normalized else res.intensity_total
    return res.omega_cm, inten


def _merge_and_2des_map(save_path, w0, dw, pol, tw_index):
    """Merge the 2des shard group in save_path and return (grid_index, flat_real):
    the REAL 2DES total spectrum (rephasing+nonrephasing) at one waiting time,
    flattened to a 1-D feature vector.

    Good taste: the 'grid' is a plain integer index so the SAME l2_relative_diff
    written for 1-D absorption works unchanged — every node lands on the identical
    FFT grid, so an element-wise compare equals np.interp on identical x. That is
    the whole 1-D↔2-D unification: one metric, flattened features, zero special
    case. w0/dw only shift/scale the (discarded) physical axis; what matters is
    that ALL nodes share them, so the convergence metric is consistent."""
    from heom_tools.common.mc_merge import merge_mc_shards, write_merged_spectrum
    from heom_tools.twodes.post.core import (
        build_tw_slices, fft_grid, pathway_plan, polarization_weights,
    )
    from heom_tools.twodes.post.h5_spectra import resolve_spectra_compute_needs
    from heom_tools.twodes.post.io import load_axes
    from heom_tools.twodes.post.types import OutputPlan
    shards = sorted(glob.glob(os.path.join(save_path, "*_rank*.h5")))
    if not shards:
        raise RuntimeError(f"no 2des shards in {save_path} (did the solver run?)")
    merged = merge_mc_shards(shards, "resp")
    merged_path = os.path.join(save_path, "merged.h5")
    write_merged_spectrum(shards[0], merged_path, merged, "resp")

    tau, _tw, t, signals, pathways = load_axes(merged_path)
    weights = polarization_weights(pol)
    pplan = pathway_plan(pathways)
    needs = resolve_spectra_compute_needs(do_plot=False, h5_spectra=["2des"],
                                          pathways=pathways)
    plan = OutputPlan(write_h5=False, do_plot=False, h5_spectra=None,
                      spectra_needs=needs, need_time=False, need_time_path=False)
    n_tau, n_t, w_tau, w_t, dtau, dt, w_tau_nat, w_t_nat = fft_grid(
        tau, t, dw=dw, w0=w0,
    )
    slices = build_tw_slices(merged_path, weights, signals, pathways,
                             pplan.pidx, [tw_index], n_tau, n_t, plan, "axis",
                             dtau=dtau, dt=dt,
                             w_tau_src=w_tau_nat, w_t_src=w_t_nat,
                             w_tau_dst=w_tau, w_t_dst=w_t)
    spec = np.asarray(slices[tw_index].spec_2des, dtype=np.complex128).real
    flat = spec.reshape(-1)
    grid = np.arange(flat.shape[0], dtype=np.float64)
    return grid, flat


def _extract_observable(save_path, task, wmin, wmax, w0, dw, pol, tw_index):
    """Task-dispatched observable from a finished node run's shard dir:
    (grid, real feature vector). absorption -> (ω, raw intensity); 2des ->
    (index, flattened real 2DES map). RAW/real so mean(O_i) stays linear."""
    if task == "2des":
        return _merge_and_2des_map(save_path, 0.0 if w0 is None else float(w0),
                                   dw, pol, tw_index)
    return _merge_and_spectrum(save_path, wmin, wmax, w0, normalized=False)


def _run_one_set(base_cfg, draws_path, tag, workers, precision, work_root,
                 task="absorption"):
    """Generate-run one node set for `task`. Returns the shard dir save_path.

    ``workers`` is the number of GPUs (one process each). Default 1 is unpinned.
    """
    from . import run as mc_run
    save_path = os.path.join(work_root, tag)
    os.makedirs(save_path, exist_ok=True)
    cfg = _cfg_for_nodes(base_cfg, draws_path, save_path, save_path, task)
    argv = [task, cfg, "--precision", precision]
    if workers > 1:
        argv.extend(["--gpus", ",".join(str(i) for i in range(workers))])
    rc = mc_run.main(argv)
    if rc != 0:
        raise RuntimeError(f"mc-disorder-run failed for set '{tag}' (rc={rc})")
    return save_path


def _per_node_spectra(base_cfg, delta, sigma, seed, tag, precision, work_root,
                      ref_omega, wmin, wmax, w0, task="absorption", dw=10.0,
                      pol="ap", tw_index=0):
    """N single-node solver runs -> O[N, n_feat] (raw/real observable) on the
    shared ref_omega grid. 1 node -> 1 worker (extra workers would idle). For
    2des the 'grid' is an integer index shared by all nodes, so the interp is
    an identity (see _merge_and_2des_map)."""
    N = delta.shape[0]
    O = np.empty((N, ref_omega.shape[0]), dtype=np.float64)
    for r in range(N):
        nd = os.path.join(work_root, f"node_{tag}_{r}.h5")
        _write_single_node(nd, delta[r], sigma, seed)
        sp = _run_one_set(base_cfg, nd, f"{tag}node_{r}", 1, precision,
                          work_root, task)
        g_r, v_r = _extract_observable(sp, task, wmin, wmax, w0, dw, pol, tw_index)
        O[r] = np.interp(ref_omega, g_r, v_r)
    return O


def _surrogate_nodes(delta, H0, mu, ref_omega, cv_width):
    from .surrogate import stick_spectrum
    c = np.empty((delta.shape[0], ref_omega.shape[0]), dtype=np.float64)
    for r in range(delta.shape[0]):
        c[r] = stick_spectrum(delta[r], H0, mu, ref_omega, cv_width)
    return c


def _load_ref_spectrum(path):
    """Load an external gold-reference spectrum (las-post .h5): (ω, peak-norm I).
    Used to compare against a trusted independent solution (e.g. disorder-in-HEOM
    deep-hierarchy L20) instead of a noisy self-iid reference."""
    import h5py
    with h5py.File(path, "r") as h:
        return h["omega_cm"][:], h["intensity_total_norm"][:]


def _peak_norm(y):
    m = float(np.max(np.abs(y)))
    return y / m if m > 0 else y


def _make_cmp(ref_omega, ref_i, normalize):
    """Return cmp(omega, inten) -> L² rel error vs the reference. When
    `normalize` (cross-scheme gold ref), peak-normalise both sides so absolute
    scale differences (MC vs disorder-in-HEOM) don't pollute the shape metric;
    CV linearity is untouched (normalisation is applied only at the metric)."""
    ref = _peak_norm(ref_i) if normalize else ref_i

    def cmp(omega, inten):
        y = _peak_norm(inten) if normalize else inten
        return l2_relative_diff(ref_omega, ref, omega, y)
    return cmp


def _mean_cv_rows(fam, O, c, c_mean, sweep_ns, ref_omega, cmp):
    """From per-node spectra: (fam) plain mean curve + (fam+"_cv") CV curve.
    Both from the SAME per-node data (fairest comparison). `cmp` is the
    reference comparator from _make_cmp (raw or peak-normalised)."""
    from .surrogate import control_variate_apply
    rows = []
    for Np in sweep_ns:
        est = O[:Np].mean(axis=0)
        rows.append((fam, Np, Np, cmp(ref_omega, est)))
        cv_est, _, _ = control_variate_apply(O[:Np], c[:Np], c_mean)
        rows.append((fam + "_cv", Np, Np, cmp(ref_omega, cv_est)))
    return rows


def _save_pool(path, precision="fp64", peak_norm=False, **arrays):
    """Persist a per-node pool, stamping the build precision as provenance so a
    reloaded pool self-documents (and load can refuse to silently mix fp64 with
    the mixed-FP32 build, whose ~1e-4 floor would pollute the sampling metric).
    `peak_norm` records whether the stored ref_i is used under peak-normalised
    shape comparison (gold cross-scheme) vs raw intensity (self-iid ref)."""
    import h5py
    with h5py.File(path, "w") as h:
        h.attrs["precision_mode"] = precision
        h.attrs["peak_norm"] = bool(peak_norm)
        for k, v in arrays.items():
            h.create_dataset(k, data=np.asarray(v))


def _load_pool(path):
    import h5py
    with h5py.File(path, "r") as h:
        d = {k: h[k][:] for k in h.keys()}
        pm = h.attrs.get("precision_mode")
        d["_precision_mode"] = pm.decode() if isinstance(pm, bytes) else pm
        pn = h.attrs.get("peak_norm")
        d["_peak_norm"] = bool(pn) if pn is not None else False
    return d


def _warn_pool_precision(P, precision, tag):
    """Guard against silently comparing pools built at different precisions."""
    pm = P.get("_precision_mode")
    if pm is not None and pm != precision:
        print(f"[mc-probe {tag}] WARNING: pool precision_mode={pm!r} != requested "
              f"{precision!r} — different builds; the FP32 ~1e-4 floor pollutes "
              "the sampling-error metric")


def _grid_probe(base_cfg, delta_row, sigma, seed, precision, work_root,
                wmin, wmax, w0, task="absorption", dw=10.0, pol="ap", tw_index=0):
    """Establish the feature grid from one throwaway single-node run (no-gold
    path). For 2des this returns the integer feature index shared by all nodes."""
    nd = os.path.join(work_root, "grid_node.h5")
    _write_single_node(nd, delta_row, sigma, seed)
    sp = _run_one_set(base_cfg, nd, "grid_probe", 1, precision, work_root, task)
    g, _ = _extract_observable(sp, task, wmin, wmax, w0, dw, pol, tw_index)
    return g


# --------------------------------------------------------------------------
# CLI: one subcommand per task. Each subparser exposes ONLY its own knobs (plus
# the shared physics/grid/precision parent), so `mc-probe rqmc --help` no longer
# lists 25 irrelevant flags. Small _add_* helpers keep the few options shared by
# 2-3 subcommands in one place instead of copy-pasted per parser.
# --------------------------------------------------------------------------

