#!/usr/bin/env python3
"""FMO-7 time-domain rephasing at short waiting time.

Compares this engine's carry-through rephasing raw against the published
ADO-reset reference data/2des_refs_data/fmo_case1_ref/time_domain/rpT0fs.dat at
the matching (τ,t) grid points. At T=0 reset≈carry-through, so the *shape*
(after normalising out the unknown overall scale/sign) must agree.

  python tests/compare_fmo7_2des.py [H5] [--ref ...rpT0fs.dat] [--rho 0.75]

Gate:
  * COARSE feasibility (fmo7_2des.yaml as committed: 10 fs, τ_max=300, L=3, K=1)
    → ρ_Re ≥ 0.75. The coarse grid (Δω≈111 cm⁻¹) and tight-but-shallow bath
    expansion smear the exciton fine structure, so this is a qualitative /
    pipeline gate, not a peak-by-peak one. Measured ρ_Re ≈ 0.80.
  * PUBLICATION convergence (separate long run: 5 fs, τ_max≈595, L≥4, K≥2)
    targets ρ_Re ≥ 0.95 with peak-by-peak agreement at short T.

Exit 0 if Pearson correlation ≥ --rho, else 1.
"""
import argparse
import glob
import os
import sys

import numpy as np

try:
    import h5py
except ImportError:
    sys.exit("compare_fmo7_2des: h5py required")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REF_DEFAULT = os.path.join(
    ROOT, "data/2des_refs_data/fmo_case1_ref/time_domain/rpT0fs.dat")


def load_mine(path, tw_index=0):
    """AP-contracted total (SE+GSB+ESA) rephasing S(τ,t) at one T."""
    with h5py.File(path, "r") as f:
        r = f["/results/resp"][...]
        resp = (r["r"] + 1j * r["i"]) if r.dtype.names else np.asarray(r)
        tau = f["/results/tau_fs"][...]
        t = f["/results/t_fs"][...]
        sig = f["/parameters"].attrs["signals"]
    sig = sig.decode() if isinstance(sig, bytes) else sig
    signals = sig.split(",")
    if "rephasing" not in signals:
        sys.exit("h5 has no rephasing signal")
    si = signals.index("rephasing")
    # resp[3, nsig, npath, n_tau, n_tw, n_t]; AP = 1/15(I1+I2+I3), sum pathways.
    ap = (resp[0] + resp[1] + resp[2]) / 15.0
    S = ap[si].sum(axis=0)[:, tw_index, :]  # [n_tau, n_t]
    return np.asarray(tau, float), np.asarray(t, float), S


def load_ref(path):
    d = np.loadtxt(path, delimiter=",")
    tau = np.unique(d[:, 0]); t = np.unique(d[:, 1])
    S = np.zeros((len(tau), len(t)), dtype=complex)
    ti = {v: i for i, v in enumerate(tau)}
    tj = {v: i for i, v in enumerate(t)}
    for row in d:
        S[ti[row[0]], tj[row[1]]] = row[2] + 1j * row[3]
    return tau, t, S


def match(tau_a, t_a, S_a, tau_b, t_b, S_b, atol=1e-6):
    """Restrict both grids to common (τ,t) points (coarse grid ⊂ fine grid)."""
    def idx(axis_a, axis_b):
        ia, ib = [], []
        for i, va in enumerate(axis_a):
            j = np.where(np.abs(axis_b - va) < atol)[0]
            if j.size:
                ia.append(i); ib.append(int(j[0]))
        return np.array(ia), np.array(ib)
    iat, ibt = idx(tau_a, tau_b)
    iax, ibx = idx(t_a, t_b)
    if iat.size < 3 or iax.size < 3:
        sys.exit(f"too few common grid points (τ:{iat.size}, t:{iax.size})")
    A = S_a[np.ix_(iat, iax)]
    B = S_b[np.ix_(ibt, ibx)]
    return A, B, iat.size, iax.size


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * np.conj(b)).sum().real /
                 (np.sqrt((np.abs(a) ** 2).sum() * (np.abs(b) ** 2).sum()) + 1e-300))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("h5", nargs="?", default=None)
    ap.add_argument("--ref", default=REF_DEFAULT)
    ap.add_argument("--rho", type=float, default=0.75)
    args = ap.parse_args(argv)

    h5 = args.h5
    if h5 is None:
        cand = sorted(glob.glob(os.path.join(ROOT, "data/fmo7_2des/*.h5")))
        if not cand:
            sys.exit("no FMO-7 h5 found in data/fmo7_2des/ — run the solver first")
        h5 = cand[-1]
    print(f"mine: {h5}\nref : {args.ref}")

    tau_a, t_a, A = load_mine(h5)
    tau_b, t_b, B = load_ref(args.ref)
    Am, Bm, n_tau, n_t = match(tau_a, t_a, A, tau_b, t_b, B)
    print(f"common grid: {n_tau} τ × {n_t} t")

    # normalise out unknown overall complex scale: align by least-squares scalar.
    scale = (np.conj(Bm) * Am).sum() / ((np.abs(Bm) ** 2).sum() + 1e-300)
    Bs = Bm * scale
    rel = np.linalg.norm(Am - Bs) / (np.linalg.norm(Am) + 1e-300)

    rho_re = pearson(Am.real.ravel(), Bm.real.ravel())
    rho_abs = pearson(np.abs(Am).ravel(), np.abs(Bm).ravel())
    print(f"Pearson(Re)   = {rho_re:.4f}")
    print(f"Pearson(|S|)  = {rho_abs:.4f}")
    print(f"scaled relL2  = {rel:.4f}  (complex scale {scale:.3g})")

    ok = rho_re >= args.rho
    print("RESULT:", "PASS" if ok else "FAIL", f"(gate ρ_Re ≥ {args.rho})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
