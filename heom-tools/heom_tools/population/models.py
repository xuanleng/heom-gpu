"""Data structures for population dynamics post-processing."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class PopulationData:
    """Site populations are canonical; exciton populations are optional."""

    filename: str
    time: np.ndarray
    populations_site: np.ndarray
    site_labels: list[str] = field(default_factory=list)
    populations_exciton: Optional[np.ndarray] = None
    exciton_labels: Optional[list[str]] = None
    primary_basis: str = "site"  # "site" or "exciton"
    temperature: str = ""
    total_time: float = 0.0

    sink_enabled: bool = False
    sink_site: int = -1
    gamma_sink_fs: float = 0.0
    N_captured: Optional[np.ndarray] = None
    Phi: float = 0.0
    t50: float = 0.0

    hamiltonian_cm: Optional[np.ndarray] = None
    w0_cm: float = 0.0
    # Columns = exciton eigenvectors in site basis (HDF5 /derived/eigenvectors).
    eigenvectors: Optional[np.ndarray] = None
    params: dict = field(default_factory=dict)

    @property
    def is_dual_basis(self) -> bool:
        return self.populations_exciton is not None

    @property
    def has_sink(self) -> bool:
        return self.sink_enabled and self.N_captured is not None

    @property
    def n_sites(self) -> int:
        return self.populations_site.shape[1]

    @property
    def n_times(self) -> int:
        return len(self.time)

    def ensure_labels(self) -> None:
        """Fill missing site / exciton label lists with defaults."""
        n = self.n_sites
        if len(self.site_labels) != n:
            self.site_labels = [f"Site{i + 1}" for i in range(n)]
        if self.is_dual_basis and (
            self.exciton_labels is None or len(self.exciton_labels) != n
        ):
            self.exciton_labels = [f"Exciton{i + 1}" for i in range(n)]


@dataclass(frozen=True)
class BasisSeries:
    """Interpolated pops on a common time grid: Δ = pops2 − pops1."""

    pops1: np.ndarray
    pops2: np.ndarray
    diff: np.ndarray


@dataclass
class ComparisonResult:
    """Site comparison always present; exciton only when H and U match."""

    data1: PopulationData
    data2: PopulationData
    time_common: np.ndarray
    site: BasisSeries
    exciton: Optional[BasisSeries] = None

    @property
    def has_exciton(self) -> bool:
        return self.exciton is not None
