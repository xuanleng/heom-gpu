"""mc-probe absorption subcommand (smoothness probe)."""
import os

from .probe_core import l2_relative_diff
from . import probe_orch as orch

def run_probe(base_cfg, levels, iid_ns, fwhm, workers, precision, wmin, wmax,
              w0, out_csv, work_root, seed):
    """Full sweep. Reference = the largest node set (highest level or largest N,
    whichever has more nodes). Writes a CSV of (family, param, n_eval, l2_err)
    and returns the rows."""
    n_site = orch._n_site_of(base_cfg)
    sigma = orch._sigma_cm(base_cfg, n_site, fwhm)
    os.makedirs(work_root, exist_ok=True)

    sets = []  # (family, param, n_node, save_path)
    for lv in levels:
        p = os.path.join(work_root, f"nodes_sgh_L{lv}.h5")
        n = orch._write_nodes_sparse_gh(p, n_site, lv, sigma)
        sp = orch._run_one_set(base_cfg, p, f"sgh_L{lv}", workers, precision, work_root)
        sets.append(("sparse_gh", lv, n, sp))
    for nn in iid_ns:
        p = os.path.join(work_root, f"nodes_iid_N{nn}.h5")
        n = orch._write_nodes_iid(p, n_site, nn, sigma, seed)
        sp = orch._run_one_set(base_cfg, p, f"iid_N{nn}", workers, precision, work_root)
        sets.append(("iid", nn, n, sp))

    spectra = {}
    for fam, param, n, sp in sets:
        spectra[(fam, param)] = orch._merge_and_spectrum(sp, wmin, wmax, w0)

    # Reference = the set with the most nodes (best-resolved). Everything is
    # measured against it; its own error is 0 by construction.
    ref_key = max(sets, key=lambda s: s[2])[:2]
    ref_omega, ref_i = spectra[ref_key]

    rows = []
    for fam, param, n, _sp in sets:
        omega, inten = spectra[(fam, param)]
        err = l2_relative_diff(ref_omega, ref_i, omega, inten)
        rows.append((fam, param, n, err))

    rows.sort(key=lambda r: (r[0], r[2]))
    with open(out_csv, "w") as fh:
        fh.write("family,param,n_eval,l2_rel_error\n")
        for fam, param, n, err in rows:
            fh.write(f"{fam},{param},{n},{err:.6e}\n")
    print(f"[mc-probe] reference set: {ref_key[0]} param={ref_key[1]} "
          f"(n_eval={max(s[2] for s in sets)})")
    print(f"[mc-probe] wrote {out_csv} ({len(rows)} rows)")
    for fam, param, n, err in rows:
        print(f"  {fam:10s} param={param:<4} n_eval={n:<5} L2_err={err:.3e}")
    return rows


