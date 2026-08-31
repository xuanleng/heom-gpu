"""mc-probe arena: variance reduction on absorption."""
import os

import numpy as np

from .probe_core import l2_relative_diff
from .surrogate import control_variate_apply, mean_stick_spectrum
from . import probe_orch as orch

def run_arena(base_cfg, sweep_ns, fwhm, workers, precision, wmin, wmax, w0,
              cv_width, ref_n, ref_seed, out_csv, work_root, seed,
              do_antithetic, stack=False, ref_file=None, per_site="",
              pool_file=None):
    """Variance-reduction comparison on absorption. Runs per-node sweeps
    (N single-node runs) and DERIVES each family's plain-mean AND control-variate
    (CV) curves from the SAME per-node spectra (fairest possible). Prints the
    killer ρ headline first.

    Reference (what "error" is measured against):
      - `ref_file`: a trusted INDEPENDENT gold spectrum (e.g. disorder-in-HEOM
        deep-hierarchy L20). Cross-scheme → peak-normalised shape comparison.
        Best: no self-reference sampling-noise floor.
      - else: an independent large-N iid run (raw comparison; has its own SEM).

    Families: iid, iid_cv always. With do_antithetic: antithetic (accumulated,
    cheap) — OR, with stack=True, antithetic + antithetic_cv (per-node), to test
    whether antithetic (kills odd/linear variance) and CV (absorbs residual
    correlation with the stick surrogate) STACK multiplicatively.

    Per-node spectra stay RAW (mean(O_i) must be linear in ρ for CV); any
    normalisation happens only inside the reference comparator.

    `pool_file`: like converge/rqmc — load if present (free re-analysis), else
    run once and save the per-node iid (+ stacked antithetic) arrays."""
    from .surrogate import mean_stick_spectrum, control_variate_apply
    from . import generate as gen
    n_site = orch._n_site_of(base_cfg)
    sigma = orch._sigma_cm(base_cfg, n_site, fwhm, per_site)
    H0, mu = orch._load_h0_mu(base_cfg)
    os.makedirs(work_root, exist_ok=True)
    N = max(sweep_ns)

    O_anti = c_anti = None
    if pool_file and os.path.exists(pool_file):
        P = orch._load_pool(pool_file)
        orch._warn_pool_precision(P, precision, "arena")
        O_iid, c_iid, c_mean = P["O_iid"], P["c_iid"], P["c_mean"]
        ref_omega, ref_i = P["ref_omega"], P["ref_i"]
        peak_norm = bool(P.get("orch._peak_norm", False))
        O_anti, c_anti = P.get("O_anti"), P.get("c_anti")
        if ref_file:
            ref_omega, ref_i = orch._load_ref_spectrum(ref_file)
            peak_norm = True
            ref_desc = f"gold file {os.path.basename(ref_file)} (peak-norm shape)"
        else:
            ref_desc = (f"pool ref ({'peak-norm' if peak_norm else 'raw'}; "
                        f"M={O_iid.shape[0]})")
        print(f"[mc-probe arena] loaded pool {pool_file} "
              f"(M={O_iid.shape[0]}, {O_iid.shape[1]} ω)")
    else:
        if ref_file:
            ref_omega, ref_i = orch._load_ref_spectrum(ref_file)
            peak_norm = True
            ref_desc = f"gold file {os.path.basename(ref_file)} (peak-norm shape)"
        else:
            # Independent reference: large iid, accumulated, RAW intensity.
            ref_path = os.path.join(work_root, f"nodes_ref_N{ref_n}.h5")
            orch._write_nodes_iid(ref_path, n_site, ref_n, sigma, ref_seed)
            ref_sp = orch._run_one_set(base_cfg, ref_path, f"ref_N{ref_n}", workers,
                                  precision, work_root)
            ref_omega, ref_i = orch._merge_and_spectrum(ref_sp, wmin, wmax, w0,
                                                   normalized=False)
            peak_norm = False
            ref_desc = f"independent iid N={ref_n} (seed={ref_seed})"

        # Exact ⟨c⟩ (shared by all families; same N(0,diag σ²) distribution).
        c_mean = mean_stick_spectrum(H0, mu, sigma, ref_omega, cv_width,
                                     n_samples=200_000, seed=ref_seed)

        # Per-node iid sweep -> iid + iid_cv curves.
        delta_iid = gen.generate_iid(N, n_site, sigma, seed)
        O_iid = orch._per_node_spectra(base_cfg, delta_iid, sigma, seed, "iid",
                                  precision, work_root, ref_omega, wmin, wmax, w0)
        c_iid = orch._surrogate_nodes(delta_iid, H0, mu, ref_omega, cv_width)

        if do_antithetic and stack:
            even = [n for n in sweep_ns if n % 2 == 0]
            Na = max(even)
            delta_anti = gen.generate_antithetic(Na, n_site, sigma, seed)
            O_anti = orch._per_node_spectra(base_cfg, delta_anti, sigma, seed, "anti",
                                       precision, work_root, ref_omega,
                                       wmin, wmax, w0)
            c_anti = orch._surrogate_nodes(delta_anti, H0, mu, ref_omega, cv_width)

        if pool_file:
            kw = dict(precision=precision, peak_norm=peak_norm,
                      O_iid=O_iid, c_iid=c_iid, c_mean=c_mean,
                      ref_omega=ref_omega, ref_i=ref_i)
            if O_anti is not None:
                kw["O_anti"] = O_anti
                kw["c_anti"] = c_anti
            orch._save_pool(pool_file, **kw)
            print(f"[mc-probe arena] saved pool {pool_file}")

    cmp = orch._make_cmp(ref_omega, ref_i, normalize=peak_norm)

    # Killer metric: intensity-weighted mean ρ over the full iid sweep.
    _, _, rho_full = control_variate_apply(O_iid, c_iid, c_mean)
    wts = ref_i ** 2
    rho_bar = float(np.sum(rho_full * wts) / np.sum(wts))

    rows = orch._mean_cv_rows("iid", O_iid, c_iid, c_mean, sweep_ns, ref_omega, cmp)

    if do_antithetic and stack:
        if O_anti is None or c_anti is None:
            # Pool was built without --stack; finish the missing antithetic nodes.
            even = [n for n in sweep_ns if n % 2 == 0]
            Na = max(even)
            delta_anti = gen.generate_antithetic(Na, n_site, sigma, seed)
            O_anti = orch._per_node_spectra(base_cfg, delta_anti, sigma, seed, "anti",
                                       precision, work_root, ref_omega,
                                       wmin, wmax, w0)
            c_anti = orch._surrogate_nodes(delta_anti, H0, mu, ref_omega, cv_width)
            if pool_file:
                orch._save_pool(pool_file, precision=precision, peak_norm=peak_norm,
                           O_iid=O_iid, c_iid=c_iid, c_mean=c_mean,
                           ref_omega=ref_omega, ref_i=ref_i,
                           O_anti=O_anti, c_anti=c_anti)
                print(f"[mc-probe arena] updated pool {pool_file} (+antithetic)")
        even = [n for n in sweep_ns if n % 2 == 0]
        rows += orch._mean_cv_rows("antithetic", O_anti, c_anti, c_mean, even,
                              ref_omega, cmp)
    elif do_antithetic:
        for Np in sweep_ns:
            if Np % 2:
                continue
            ap = os.path.join(work_root, f"nodes_anti_N{Np}.h5")
            orch._write_nodes_antithetic(ap, n_site, Np, sigma, seed)
            sp = orch._run_one_set(base_cfg, ap, f"anti_N{Np}", workers, precision, work_root)
            w_a, i_a = orch._merge_and_spectrum(sp, wmin, wmax, w0, normalized=False)
            rows.append(("antithetic", Np, Np, cmp(w_a, i_a)))

    rows.sort(key=lambda r: (r[0], r[2]))
    with open(out_csv, "w") as fh:
        fh.write("family,param,n_eval,l2_rel_error\n")
        for fam, param, n, err in rows:
            fh.write(f"{fam},{param},{n},{err:.6e}\n")

    print(f"[mc-probe arena] reference: {ref_desc}")
    print(f"[mc-probe arena] KILLER METRIC  intensity-weighted ρ(O,stick) = {rho_bar:.4f}"
          f"   → CV variance-reduction ceiling 1/(1−ρ²) = {1.0/max(1e-12, 1-rho_bar**2):.2f}×")
    print(f"[mc-probe arena] wrote {out_csv} ({len(rows)} rows)")
    for fam, param, n, err in rows:
        print(f"  {fam:11s} n_eval={n:<5} L2_err={err:.3e}")
    return rows, rho_bar


