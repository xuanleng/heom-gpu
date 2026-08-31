"""mc-probe rqmc subcommand (Sobol rate de-risk)."""
import os

import numpy as np

from .probe_core import l2_relative_diff
from . import probe_orch as orch

def _fit_loglog_slope(ns, errs):
    """Least-squares slope of log(err) vs log(N). iid/antithetic → −0.5;
    RQMC breaking the barrier → steeper (more negative). Non-positive errors
    (e.g. a self-reference's exact-zero tail point) are dropped — log is
    undefined there; with < 2 usable points the slope is nan."""
    ns = np.asarray(ns, dtype=np.float64)
    errs = np.asarray(errs, dtype=np.float64)
    ok = np.isfinite(errs) & (errs > 0)
    if ok.sum() < 2:
        return float("nan"), float("nan")
    x = np.log(ns[ok])
    y = np.log(errs[ok])
    A = np.vstack([x, np.ones_like(x)]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(m), float(b)


def run_rqmc(base_cfg, sweep_ns, n_replica, fwhm, per_site, precision,
             wmin, wmax, w0, ref_file, seed, out_csv, work_root,
             iid_pool=None, pool_file=None, task="absorption", dw=10.0,
             pol="ap", tw_index=0, ref_n=256, ref_seed=99999):
    """RQMC (scrambled Sobol) rate DE-RISK probe. The one question: does the
    empirical convergence exponent beat iid's −0.5? Two curves come out:
      * rqmc_1rep — RMS over R scrambles of one replica's N-point prefix mean;
        measures the sequence EXPONENT (cost N, but NO error bar).
      * rqmc_comb — the DEPLOYED estimator: mean over ALL R replicas, whose real
        cost is R·N TOTAL evaluations. This is the ONLY fair axis vs iid.
    Comparing rqmc_1rep(x=N) to iid(x=N) is apples-to-oranges by R — always read
    the CSV's total_eval column. iid baseline is read FREE from the converge
    pool. Only the Sobol points cost GPU.

    `task` picks the observable (absorption 1-D or 2des 2-D map). The reference
    (what "error" is measured against) has THREE modes:
      * ref_file  — external gold (absorption's disorder-in-HEOM L20); cross-
                    scheme peak-normalised shape compare. REQUIRED for absorption.
      * ref_n > 0 — 2des: an INDEPENDENT large-iid reference (seed ref_seed).
                    Keep ref_n well above the deepest total R·M so its own √N
                    noise doesn't floor the metric. Honest but expensive.
      * ref_n ≤ 0 — 2des: SELF reference = the pool's own combined grand mean
                    (all R·M). Zero extra GPU (cheapest de-risk); the deepest
                    points bias toward it, so read the SLOPE from the cheap→mid
                    range, not the tail."""
    from . import generate as gen
    n_site = orch._n_site_of(base_cfg)
    sigma = orch._sigma_cm(base_cfg, n_site, fwhm, per_site)
    os.makedirs(work_root, exist_ok=True)
    M = max(sweep_ns)
    self_ref = (not ref_file) and task == "2des" and ref_n <= 0

    # Load an existing pool first (the expensive replica nodes + any stored ref).
    P = orch._load_pool(pool_file) if (pool_file and os.path.exists(pool_file)) else None
    O = ref_omega = ref_vec = gold_i = None
    if P is not None:
        orch._warn_pool_precision(P, precision, "rqmc")
        O = P["O_rqmc"]
        ref_omega = P["ref_omega"]
        ref_vec = P["ref_vec"] if "ref_vec" in P else None
        print(f"[mc-probe rqmc] loaded pool {pool_file} "
              f"(R={O.shape[0]}, M={O.shape[1]})")

    # Establish the feature grid (needed to size the pool) and the reference,
    # except the self-ref which is derived from the pool after it is built.
    if ref_file:
        ref_omega, gold_i = orch._load_ref_spectrum(ref_file)
    elif task != "2des":
        raise ValueError("rqmc probe needs --ref-file (gold) for absorption")
    elif ref_omega is None:
        seed0 = gen.generate_iid(1, n_site, sigma, ref_seed)[0]
        ref_omega = orch._grid_probe(base_cfg, seed0, sigma, ref_seed, precision,
                                work_root, wmin, wmax, w0, task, dw, pol, tw_index)

    if (not ref_file) and (not self_ref) and ref_vec is None:
        if ref_seed == seed:
            raise ValueError("--ref-seed must differ from --seed (independent ref)")
        ref_delta = gen.generate_iid(ref_n, n_site, sigma, ref_seed)
        Oref = orch._per_node_spectra(base_cfg, ref_delta, sigma, ref_seed, "rqmcref",
                                 precision, work_root, ref_omega, wmin, wmax, w0,
                                 task, dw, pol, tw_index)
        ref_vec = Oref.mean(axis=0)
        print(f"[mc-probe rqmc] 2des reference: independent iid N={ref_n} "
              f"(seed={ref_seed}), {ref_omega.shape[0]} features")

    # Build the replica pool if not loaded.
    if O is None:
        O = np.empty((n_replica, M, ref_omega.shape[0]), dtype=np.float64)
        for r in range(n_replica):
            d = gen.generate_rqmc(M, n_site, sigma, seed, replica=r)
            O[r] = orch._per_node_spectra(base_cfg, d, sigma, seed, f"rqmc{r}",
                                     precision, work_root, ref_omega,
                                     wmin, wmax, w0, task, dw, pol, tw_index)
        if pool_file:
            kw = dict(precision=precision, O_rqmc=O, ref_omega=ref_omega)
            if ref_vec is not None:      # store an iid reference (self-ref is free)
                kw["ref_vec"] = ref_vec
            orch._save_pool(pool_file, **kw)
            print(f"[mc-probe rqmc] saved pool {pool_file}")

    if self_ref:
        ref_vec = O.mean(axis=(0, 1))    # combined grand mean = the target
        print(f"[mc-probe rqmc] 2des reference: SELF combined grand mean "
              f"(R·M={O.shape[0] * O.shape[1]} total), {ref_omega.shape[0]} feat")

    cmp = (orch._make_cmp(ref_omega, gold_i, normalize=True) if ref_file
           else orch._make_cmp(ref_omega, ref_vec, normalize=False))
    R = O.shape[0]

    # (a) SINGLE-REPLICA rate: RMS over R scrambles of one replica's N-point
    #     mean. This measures the sequence's convergence EXPONENT. Cost of ONE
    #     such estimate is N, but it carries NO error bar (a lone randomised set
    #     has no error bar. Optimistic point-estimate bound.
    rq = np.array([[cmp(ref_omega, O[r, :N].mean(axis=0)) for N in sweep_ns]
                   for r in range(R)])
    rq_rms = np.sqrt((rq ** 2).mean(axis=0))
    rq_std = rq.std(axis=0)

    # (b) HONEST production curve: the DEPLOYED estimator averages ALL R
    #     replicas, so its real cost is R·N TOTAL evaluations. This is the only
    #     fair x-axis against iid (which spends its full N on one set). Plotting
    #     the single-replica error at x=N vs iid at x=N is apples-to-oranges by
    #     the factor R — always compare on total_eval.
    comb = np.array([cmp(ref_omega, O[:, :N].mean(axis=(0, 1)))
                     for N in sweep_ns])
    comb_total = np.array([R * N for N in sweep_ns])

    iid_err = None
    if iid_pool and os.path.exists(iid_pool):
        O_iid = orch._load_pool(iid_pool)["O_iid"]
        iid_err = np.array([cmp(ref_omega, O_iid[:N].mean(axis=0))
                            for N in sweep_ns])

    m_rq, _ = _fit_loglog_slope(sweep_ns, rq_rms)     # per-replica exponent
    m_cb, _ = _fit_loglog_slope(comb_total, comb)     # honest total-eval exponent
    m_iid = (_fit_loglog_slope(sweep_ns, iid_err)[0]
             if iid_err is not None else float("nan"))

    # explicit total_eval column kills the per-replica/total ambiguity forever
    with open(out_csv, "w") as fh:
        fh.write("family,n_eval,total_eval,err,err_std\n")
        for j, N in enumerate(sweep_ns):
            fh.write(f"rqmc_1rep,{N},{N},{rq_rms[j]:.6e},{rq_std[j]:.6e}\n")
        for j, N in enumerate(sweep_ns):
            fh.write(f"rqmc_comb,{N},{comb_total[j]},{comb[j]:.6e},0\n")
        if iid_err is not None:
            for j, N in enumerate(sweep_ns):
                fh.write(f"iid,{N},{N},{iid_err[j]:.6e},0\n")

    ref_desc = (f"gold={os.path.basename(ref_file)}" if ref_file
                else f"ref=iid_N{ref_n}(seed{ref_seed})")
    print(f"\n[mc-probe rqmc] task={task}, pool R={R} × M={M}, fp64, "
          f"per-site={per_site!r}, {ref_desc}")
    print("[mc-probe rqmc] " + "N".ljust(7) + "rqmc_1rep(RMS±std)".ljust(26)
          + "total".ljust(8) + "rqmc_comb".ljust(13) + "iid@N")
    for j, N in enumerate(sweep_ns):
        ib = f"{iid_err[j]:.3e}" if iid_err is not None else "n/a"
        print(f"  {N:<6} {rq_rms[j]:.3e} ± {rq_std[j]:.0e}     "
              f"{comb_total[j]:<6} {comb[j]:.3e}    {ib}")
    print(f"[mc-probe rqmc] LOG-LOG SLOPE  1rep={m_rq:+.3f}  "
          f"comb(total)={m_cb:+.3f}  iid={m_iid:+.3f} (iid≈−0.5 theory)")
    verdict = ("RQMC BREAKS the 1/√N barrier → worth the full replica-variance "
               "machinery" if m_cb < -0.60 else
               "RQMC ≈ 1/√N (just a constant factor at best) → NOT worth the "
               "extra machinery; antithetic+CV stays the answer")
    print(f"[mc-probe rqmc] VERDICT: {verdict}")
    print("[mc-probe rqmc] NOTE: compare methods on TOTAL evaluations. The "
          "honest deployed win is rqmc_comb-vs-iid; rqmc_1rep is the "
          "no-error-bar optimistic bound.")
    print(f"[mc-probe rqmc] wrote {out_csv}")
    return {"sweep": sweep_ns, "rq_rms": rq_rms, "rq_std": rq_std,
            "comb_total": comb_total, "comb": comb, "iid": iid_err,
            "slope_rqmc": m_rq, "slope_comb": m_cb, "slope_iid": m_iid}


