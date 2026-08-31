"""mc-probe converge subcommand (closed-form N*)."""
import os

import numpy as np

from .probe_core import analyze_pool, l2_relative_diff
from . import probe_orch as orch

PROJECTION_NS = (8, 16, 32, 64, 128, 256, 512, 1024, 2048)

def run_converge(base_cfg, pool_size, tols, fwhm, per_site, cv_width,
                 precision, wmin, wmax, w0, ref_file, seed, out_csv,
                 work_root, pool_file=None):
    """Minimum-evaluations-to-converge, the core deliverable. Run a per-node
    pool ONCE (iid + antithetic, persisted), then answer analytically for
    iid / antithetic / antithetic+CV. A gold --ref-file additionally reports the
    IRREDUCIBLE scheme floor ‖μ_pool − gold‖/‖gold‖ (below it, more sampling is
    pointless — the two schemes simply differ)."""
    from .surrogate import mean_stick_spectrum
    from . import generate as gen
    n_site = orch._n_site_of(base_cfg)
    sigma = orch._sigma_cm(base_cfg, n_site, fwhm, per_site)
    H0, mu_phys = orch._load_h0_mu(base_cfg)
    os.makedirs(work_root, exist_ok=True)

    if ref_file:
        ref_omega, gold_i = orch._load_ref_spectrum(ref_file)
    else:
        ref_omega, gold_i = None, None

    if pool_file and os.path.exists(pool_file):
        P = orch._load_pool(pool_file)
        orch._warn_pool_precision(P, precision, "converge")
        O_iid, O_anti = P["O_iid"], P["O_anti"]
        c_iid, c_anti, c_mean = P["c_iid"], P["c_anti"], P["c_mean"]
        ref_omega = P["ref_omega"]
        print(f"[mc-probe converge] loaded pool {pool_file} "
              f"(M={O_iid.shape[0]}, {O_iid.shape[1]} ω)")
    else:
        if pool_size % 2:
            raise ValueError("--pool-size must be even (antithetic pairs)")
        delta_iid = gen.generate_iid(pool_size, n_site, sigma, seed)
        delta_anti = gen.generate_antithetic(pool_size, n_site, sigma, seed)
        if ref_omega is None:
            # No gold: establish the ω grid from one throwaway single-node run.
            ref_omega = orch._grid_probe(base_cfg, delta_iid[0], sigma, seed,
                                    precision, work_root, wmin, wmax, w0)
        O_iid = orch._per_node_spectra(base_cfg, delta_iid, sigma, seed, "cvgiid",
                                  precision, work_root, ref_omega, wmin, wmax, w0)
        O_anti = orch._per_node_spectra(base_cfg, delta_anti, sigma, seed, "cvganti",
                                   precision, work_root, ref_omega, wmin, wmax, w0)
        c_iid = orch._surrogate_nodes(delta_iid, H0, mu_phys, ref_omega, cv_width)
        c_anti = orch._surrogate_nodes(delta_anti, H0, mu_phys, ref_omega, cv_width)
        c_mean = mean_stick_spectrum(H0, mu_phys, sigma, ref_omega, cv_width,
                                     n_samples=200_000, seed=seed + 1)
        if pool_file:
            orch._save_pool(pool_file, precision=precision, O_iid=O_iid,
                       O_anti=O_anti, c_iid=c_iid, c_anti=c_anti,
                       c_mean=c_mean, ref_omega=ref_omega)
            print(f"[mc-probe converge] saved pool {pool_file}")

    res = analyze_pool(O_iid, O_anti, c_iid, c_anti, c_mean, tols)

    floor = None
    M = O_iid.shape[0]
    if gold_i is not None:
        cmp = orch._make_cmp(ref_omega, gold_i, normalize=True)
        floor = cmp(ref_omega, res["_mu"])
        pool_noise = res["iid"]["A"] / np.sqrt(M)

    fams = ["iid", "iid_cv", "antithetic", "antithetic_cv"]
    with open(out_csv, "w") as fh:
        fh.write("family,A,n_eval,err_self,err_vs_gold\n")
        for fam in fams:
            A = res[fam]["A"]
            for N in PROJECTION_NS:
                es = A / np.sqrt(N)
                eg = (np.sqrt(floor ** 2 + es ** 2) if floor is not None
                      else float("nan"))
                fh.write(f"{fam},{A:.6e},{N},{es:.6e},{eg:.6e}\n")

    print(f"\n[mc-probe converge] pool M={O_iid.shape[0]} "
          f"({O_anti.shape[0]//2} antithetic pairs), fp64, per-site={per_site!r}")
    if floor is not None:
        print(f"[mc-probe converge] ‖μ_pool(M={M}) − gold‖/‖gold‖ = {floor:.4f} "
              f"— WARNING: sampling-inflated; the pool mean's own noise is "
              f"~A/√M = {pool_noise:.4f}, so this is an UPPER bound on the true "
              f"scheme floor (estimate that from a large-N iid mean, not M=256).")
    print(f"[mc-probe converge] per-eval variance reduction vs iid: "
          + ", ".join(f"{f}={res['_var_ratio'][f]:.2f}×" for f in fams))
    hdr = "  method         A        " + "".join(f"N*({t:g})".rjust(11) for t in tols)
    print(hdr)
    for fam in fams:
        row = f"  {fam:13s} {res[fam]['A']:.4f}  "
        row += "".join(str(res[fam]['nstar'][t]).rjust(11) for t in tols)
        print(row)
    print(f"[mc-probe converge] wrote {out_csv}")
    return res, floor


