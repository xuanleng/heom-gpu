"""h5_spectra token table — single source for compute and HDF5 write."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from .constants import H5_PATHWAY_SUFFIX, SpecMap

SpecTotalStack = Dict[str, np.ndarray]


@dataclass(frozen=True)
class H5TokenSpec:
    """One config token: FFT needs and HDF5 destination."""

    h5_group: str
    h5_dataset: str
    total_signals: FrozenSet[str] = frozenset()
    pathway: Optional[str] = None
    pathway_signal: Optional[str] = None


def _build_token_specs() -> Dict[str, H5TokenSpec]:
    specs: Dict[str, H5TokenSpec] = {
        "2des": H5TokenSpec(
            "total", "2des", total_signals=frozenset({"rephasing", "nonrephasing"}),
        ),
        "rp": H5TokenSpec("total", "rephasing", total_signals=frozenset({"rephasing"})),
        "nr": H5TokenSpec("total", "nonrephasing", total_signals=frozenset({"nonrephasing"})),
    }
    for suffix, pname in H5_PATHWAY_SUFFIX.items():
        specs[f"rp-{suffix}"] = H5TokenSpec(
            pname, "rephasing", pathway=pname, pathway_signal="rephasing",
        )
        specs[f"nr-{suffix}"] = H5TokenSpec(
            pname, "nonrephasing", pathway=pname, pathway_signal="nonrephasing",
        )
    return specs


H5_TOKEN_SPECS: Dict[str, H5TokenSpec] = _build_token_specs()
H5_SPECTRA_TOKENS: FrozenSet[str] = frozenset(H5_TOKEN_SPECS)


@dataclass(frozen=True)
class SpectraComputeNeeds:
    """Merged FFT work for one Tw slice (plot ∪ h5_spectra)."""

    pathway_fft: FrozenSet[Tuple[int, str]]
    total_fft: FrozenSet[str]
    compute_2des: bool


def token_spec(token: str) -> H5TokenSpec:
    try:
        return H5_TOKEN_SPECS[token]
    except KeyError as e:
        raise ValueError(f"unknown h5_spectra token {token!r}") from e


def normalize_h5_spectra(raw, *, write_h5: bool) -> Optional[List[str]]:
    """Parse config list; ignored (None) unless write_h5 is true."""
    if not write_h5:
        return None
    if raw is None:
        raise ValueError("write_h5 is true but h5_spectra is missing")
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"h5_spectra must be a list, got {type(raw).__name__}")
    if write_h5 and not raw:
        raise ValueError("h5_spectra must not be empty when write_h5 is true")
    out: List[str] = []
    for item in raw:
        tok = str(item).strip().lower()
        if tok not in H5_SPECTRA_TOKENS:
            known = ", ".join(sorted(H5_SPECTRA_TOKENS))
            raise ValueError(f"unknown h5_spectra token {item!r} (known: {known})")
        if tok not in out:
            out.append(tok)
    return out if out else None


def validate_h5_spectra_pathways(tokens: List[str], pathways: List[str]) -> None:
    """Hard error when a pathway token is absent from the solver file."""
    present = set(pathways)
    for tok in tokens:
        spec = token_spec(tok)
        if spec.pathway is not None and spec.pathway not in present:
            raise ValueError(
                f"h5_spectra token {tok!r} requires pathway {spec.pathway!r} in solver file; "
                f"have {pathways}"
            )


def h5_spectra_fingerprint(
    tokens: Optional[List[str]], *, write_h5: bool,
) -> Optional[List[str]]:
    """Sidecar / post H5 attrs; omitted when write_h5 is false."""
    if not write_h5 or tokens is None:
        return None
    return list(tokens)


def resolve_spectra_compute_needs(
    *,
    do_plot: bool,
    h5_spectra: Optional[List[str]],
    pathways: List[str],
) -> SpectraComputeNeeds:
    pathway_fft: Set[Tuple[int, str]] = set()
    total_fft: Set[str] = set()

    if do_plot:
        total_fft.update(("rephasing", "nonrephasing"))
        for pi in range(len(pathways)):
            pathway_fft.add((pi, "rephasing"))
            pathway_fft.add((pi, "nonrephasing"))

    write_2des = False
    if h5_spectra:
        validate_h5_spectra_pathways(h5_spectra, pathways)
        for tok in h5_spectra:
            spec = token_spec(tok)
            total_fft.update(spec.total_signals)
            if tok == "2des":
                write_2des = True
            if spec.pathway is not None and spec.pathway_signal is not None:
                pi = pathways.index(spec.pathway)
                pathway_fft.add((pi, spec.pathway_signal))

    compute_2des = do_plot or write_2des
    return SpectraComputeNeeds(
        pathway_fft=frozenset(pathway_fft),
        total_fft=frozenset(total_fft),
        compute_2des=compute_2des,
    )


def array_for_token(
    token: str,
    *,
    spec_by_pathway: SpecMap,
    spec_total: SpecTotalStack,
) -> np.ndarray:
    """Resolve stacked spectrum array for one write token.

    spec_total is flat: {rephasing, nonrephasing, 2des, ...} — every total
    signal (including the 2des sum) lives at the same level, so total tokens
    and pathway tokens share one lookup path each.
    """
    spec = token_spec(token)
    if spec.pathway is None:
        if spec.h5_dataset not in spec_total:
            raise ValueError(
                f"h5_spectra includes {token!r} but {spec.h5_dataset} was not computed"
            )
        return spec_total[spec.h5_dataset]
    pname, sname = spec.pathway, spec.pathway_signal
    if pname not in spec_by_pathway or sname not in spec_by_pathway[pname]:
        raise ValueError(
            f"h5_spectra includes {token!r} but {pname}/{sname} was not computed"
        )
    return spec_by_pathway[pname][sname]
