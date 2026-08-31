"""Unit tests for population I/O and comparison helpers."""
from __future__ import annotations

import io
import tempfile
import unittest
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from heom_tools.population.analysis import (
    create_comparison_result,
    exciton_skip_reason,
    eigenvectors_compatible,
    format_compare_l2_label,
    hamiltonians_compatible,
    population_l2_rel_percent,
    print_analysis,
    windowed_series,
)
from heom_tools.population.config_diff import format_params_diff_overlay
from heom_tools.population.draw import add_sink_curves, plot_comparison_panel, refresh_legend
from heom_tools.population.io import parse_population_file
from heom_tools.population.models import PopulationData
from heom_tools.population.plot_compare import compare_dual_basis
from heom_tools.population.util import interp_populations


def _site_data(
    name: str,
    pops: np.ndarray,
    *,
    time: np.ndarray | None = None,
    params: dict | None = None,
) -> PopulationData:
    t = time if time is not None else np.array([0.0, 10.0])
    d = PopulationData(
        filename=name,
        time=t,
        populations_site=pops,
        site_labels=[f"Site{i + 1}" for i in range(pops.shape[1])],
        params=params or {},
    )
    d.ensure_labels()
    return d


def _dual_data(
    name: str,
    pops_site: np.ndarray,
    pops_exc: np.ndarray,
    H: np.ndarray,
    U: np.ndarray,
    *,
    w0_cm: float = 12000.0,
) -> PopulationData:
    t = np.array([0.0, 10.0])
    d = PopulationData(
        filename=name,
        time=t,
        populations_site=pops_site,
        populations_exciton=pops_exc,
        site_labels=["Site1", "Site2"],
        exciton_labels=["Exciton1", "Exciton2"],
        hamiltonian_cm=H,
        w0_cm=w0_cm,
        eigenvectors=U,
    )
    d.ensure_labels()
    return d


class TestPopulationIO(unittest.TestCase):
    def test_parse_single_basis_text(self) -> None:
        content = """# Temperature: 77 K
# Total time: 100.0 fs
Time(fs),Site1,Site2
0.0,1.0,0.0
10.0,0.9,0.1
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            path = f.name

        try:
            data = parse_population_file(path)
            self.assertEqual(data.n_sites, 2)
            self.assertEqual(data.site_labels, ["Site1", "Site2"])
            np.testing.assert_allclose(data.populations_site[0], [1.0, 0.0])
        finally:
            Path(path).unlink(missing_ok=True)

    def test_comparison_uses_site_basis(self) -> None:
        t = np.array([0.0, 10.0, 20.0])
        pops_site = np.array([[1.0, 0.0], [0.8, 0.2], [0.5, 0.5]])
        pops_exciton = np.array([[0.0, 1.0], [0.1, 0.9], [0.3, 0.7]])

        d1 = PopulationData(
            filename="a.h5",
            time=t,
            populations_site=pops_site,
            site_labels=["Site1", "Site2"],
            primary_basis="site",
        )
        d2 = PopulationData(
            filename="b.h5",
            time=t,
            populations_site=pops_site * 0.9,
            populations_exciton=pops_exciton,
            site_labels=["Site1", "Site2"],
            exciton_labels=["Exciton1", "Exciton2"],
            primary_basis="exciton",
        )
        d1.ensure_labels()
        d2.ensure_labels()

        result = create_comparison_result(d1, d2)
        self.assertFalse(result.has_exciton)
        np.testing.assert_allclose(result.site.pops1, pops_site)
        np.testing.assert_allclose(result.site.pops2, pops_site * 0.9)

    def test_interp_populations_shape(self) -> None:
        time = np.linspace(0, 10, 5)
        pops = np.column_stack([time, time * 0.1])
        grid = np.array([0.0, 5.0, 10.0])
        out = interp_populations(grid, time, pops, 2)
        self.assertEqual(out.shape, (3, 2))
        np.testing.assert_allclose(out[:, 0], grid)

    def test_params_diff_overlay(self) -> None:
        text = format_params_diff_overlay(
            {"dt": 1.0, "L_cutoff": 3},
            {"dt": 2.0, "L_cutoff": 5},
        )
        self.assertIsNotNone(text)
        assert text is not None
        self.assertTrue(text.startswith("1 vs 2:"))
        self.assertRegex(text, r"L_cutoff\s+3\s+->\s+5")
        self.assertRegex(text, r"dt\s+1\s+->\s+2")

        same = format_params_diff_overlay({"dt": 1.0}, {"dt": 1.0})
        self.assertIsNone(same)

    def test_comparison_diff_is_file2_minus_file1(self) -> None:
        pops1 = np.array([[1.0, 0.0], [0.5, 0.5]])
        pops2 = np.array([[1.0, 0.0], [0.4, 0.6]])
        d1 = _site_data("a.h5", pops1, params={"L_cutoff": 3})
        d2 = _site_data("b.h5", pops2, params={"L_cutoff": 5})
        result = create_comparison_result(d1, d2)
        np.testing.assert_allclose(result.site.diff, pops2 - pops1)
        overlay = format_params_diff_overlay(d1.params, d2.params)
        self.assertIsNotNone(overlay)
        assert overlay is not None
        self.assertTrue(overlay.startswith("1 vs 2:"))
        self.assertRegex(overlay, r"L_cutoff\s+3\s+->\s+5")

    def test_population_l2_is_global_not_per_site(self) -> None:
        pops_ref = np.ones((3, 2))
        pops_other = np.ones((3, 2))
        pops_other[1, 0] = 1.1
        pct = population_l2_rel_percent(pops_ref, pops_other)
        expected = 100.0 * 0.1 / np.sqrt(6.0)
        self.assertAlmostEqual(pct, expected, places=10)
        self.assertEqual(format_compare_l2_label(pct), f"1 vs 2 (L² {pct:.1f}%)")
        self.assertGreater(pct, 0.0)
        self.assertEqual(
            population_l2_rel_percent(pops_ref[:, 1:2], pops_other[:, 1:2]),
            0.0,
        )

    def test_population_l2_matches_constant_scale(self) -> None:
        pops_ref = np.ones((4, 3))
        pops_other = pops_ref * 0.5
        self.assertAlmostEqual(population_l2_rel_percent(pops_ref, pops_other), 50.0)

    def test_site_count_mismatch_raises(self) -> None:
        d1 = _site_data("a.h5", np.ones((2, 2)))
        d2 = _site_data("b.h5", np.ones((2, 3)))
        with self.assertRaises(ValueError) as ctx:
            create_comparison_result(d1, d2)
        self.assertIn("site count mismatch", str(ctx.exception))

    def test_eigenvectors_compatible_allows_column_phase(self) -> None:
        U1 = np.eye(2, dtype=np.complex128)
        U2 = U1.copy()
        U2[:, 1] *= -1.0
        self.assertTrue(eigenvectors_compatible(U1, U2))

    def test_eigenvectors_compatible_rejects_permutation(self) -> None:
        U1 = np.eye(2, dtype=np.complex128)
        U2 = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        self.assertFalse(eigenvectors_compatible(U1, U2))

    def test_exciton_compared_when_h_and_u_match(self) -> None:
        pops_site = np.array([[1.0, 0.0], [0.5, 0.5]])
        pops_exc = np.array([[0.0, 1.0], [0.2, 0.8]])
        H = np.array([[100.0, 10.0], [10.0, 200.0]])
        U = np.eye(2, dtype=np.complex128)

        d1 = _dual_data("a.h5", pops_site, pops_exc, H, U)
        d2 = _dual_data("b.h5", pops_site * 0.9, pops_exc * 0.8, H.copy(), U.copy())
        self.assertTrue(hamiltonians_compatible(d1, d2))

        result = create_comparison_result(d1, d2)
        self.assertTrue(result.has_exciton)
        assert result.exciton is not None
        np.testing.assert_allclose(result.exciton.pops1, pops_exc)
        np.testing.assert_allclose(result.exciton.diff, pops_exc * 0.8 - pops_exc)

    def test_exciton_skipped_when_hamiltonians_differ(self) -> None:
        pops_site = np.array([[1.0, 0.0], [0.5, 0.5]])
        pops_exc = np.array([[0.0, 1.0], [0.2, 0.8]])
        U = np.eye(2, dtype=np.complex128)
        d1 = _dual_data(
            "a.h5",
            pops_site,
            pops_exc,
            np.array([[100.0, 10.0], [10.0, 200.0]]),
            U,
        )
        d2 = _dual_data(
            "b.h5",
            pops_site,
            pops_exc,
            np.array([[100.0, 20.0], [20.0, 200.0]]),
            U,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = create_comparison_result(d1, d2)
        self.assertFalse(result.has_exciton)
        self.assertTrue(
            any("exciton comparison skipped" in str(w.message) for w in caught)
        )
        self.assertIn("Hamiltonians", exciton_skip_reason(d1, d2) or "")

    def test_exciton_skipped_when_u_permuted(self) -> None:
        pops_site = np.array([[1.0, 0.0], [0.5, 0.5]])
        pops_exc = np.array([[0.0, 1.0], [0.2, 0.8]])
        H = np.array([[100.0, 10.0], [10.0, 200.0]])
        U1 = np.eye(2, dtype=np.complex128)
        U2 = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        d1 = _dual_data("a.h5", pops_site, pops_exc, H, U1)
        d2 = _dual_data("b.h5", pops_site, pops_exc, H.copy(), U2)

        result = create_comparison_result(d1, d2)
        self.assertFalse(result.has_exciton)
        reason = exciton_skip_reason(d1, d2) or ""
        self.assertIn("eigenvectors differ", reason)

    def test_exciton_skipped_without_u_even_if_h_matches(self) -> None:
        """Text dual-basis exports have pops but no U — cannot compare exciton."""
        pops_site = np.array([[1.0, 0.0], [0.5, 0.5]])
        pops_exc = np.array([[0.0, 1.0], [0.2, 0.8]])
        H = np.array([[100.0, 10.0], [10.0, 200.0]])
        d1 = _dual_data("a.h5", pops_site, pops_exc, H, np.eye(2, dtype=np.complex128))
        d2 = PopulationData(
            filename="b.txt",
            time=np.array([0.0, 10.0]),
            populations_site=pops_site,
            populations_exciton=pops_exc,
            site_labels=["Site1", "Site2"],
            exciton_labels=["Exciton1", "Exciton2"],
            hamiltonian_cm=H.copy(),
            w0_cm=12000.0,
            eigenvectors=None,
        )
        d2.ensure_labels()
        result = create_comparison_result(d1, d2)
        self.assertFalse(result.has_exciton)
        self.assertIn("eigenvectors missing", exciton_skip_reason(d1, d2) or "")

    def test_print_analysis_reports_exciton_skip_reason(self) -> None:
        pops_site = np.array([[1.0, 0.0], [0.5, 0.5]])
        pops_exc = np.array([[0.0, 1.0], [0.2, 0.8]])
        U = np.eye(2, dtype=np.complex128)
        d1 = _dual_data(
            "a.h5",
            pops_site,
            pops_exc,
            np.array([[100.0, 10.0], [10.0, 200.0]]),
            U,
        )
        d2 = _dual_data(
            "b.h5",
            pops_site,
            pops_exc,
            np.array([[100.0, 20.0], [20.0, 200.0]]),
            U,
        )
        result = create_comparison_result(d1, d2)
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_analysis(result, verbose=False)
        out = buf.getvalue()
        self.assertIn("exciton basis: skipped", out)
        self.assertIn("Hamiltonians", out)

    def test_print_analysis_l2_uses_time_window(self) -> None:
        t = np.array([0.0, 10.0, 20.0, 30.0])
        pops1 = np.ones((4, 2))
        pops2 = np.ones((4, 2))
        pops2[2:, :] = 2.0

        d1 = _site_data("a.h5", pops1, time=t)
        d2 = _site_data("b.h5", pops2, time=t)
        result = create_comparison_result(d1, d2)

        _, site_full, _ = windowed_series(result, None)
        _, site_win, _ = windowed_series(result, 10.0)
        l2_full = population_l2_rel_percent(site_full.pops1, site_full.pops2)
        l2_win = population_l2_rel_percent(site_win.pops1, site_win.pops2)
        self.assertGreater(l2_full, 0.0)
        self.assertEqual(l2_win, 0.0)
        print_analysis(result, time_limit=10.0, verbose=False)

    def test_empty_time_window_raises(self) -> None:
        d1 = _site_data("a.h5", np.ones((2, 2)), time=np.array([5.0, 10.0]))
        d2 = _site_data("b.h5", np.ones((2, 2)), time=np.array([5.0, 10.0]))
        result = create_comparison_result(d1, d2)
        with self.assertRaises(ValueError) as ctx:
            windowed_series(result, time_limit=1.0)
        self.assertIn("time window is empty", str(ctx.exception))

    def test_nonpositive_time_limit_raises(self) -> None:
        d1 = _site_data("a.h5", np.ones((2, 2)))
        d2 = _site_data("b.h5", np.ones((2, 2)))
        result = create_comparison_result(d1, d2)
        for bad in (0.0, -1.0):
            with self.assertRaises(ValueError) as ctx:
                windowed_series(result, time_limit=bad)
            self.assertIn("time_limit must be positive", str(ctx.exception))

    def test_exciton_skip_reason_none_when_compatible(self) -> None:
        pops_site = np.array([[1.0, 0.0], [0.5, 0.5]])
        pops_exc = np.array([[0.0, 1.0], [0.2, 0.8]])
        H = np.array([[100.0, 10.0], [10.0, 200.0]])
        U = np.eye(2, dtype=np.complex128)
        d1 = _dual_data("a.h5", pops_site, pops_exc, H, U)
        d2 = _dual_data("b.h5", pops_site, pops_exc, H.copy(), U.copy())
        self.assertIsNone(exciton_skip_reason(d1, d2))

    def test_dual_compare_panel_titles(self) -> None:
        pops_site = np.array([[1.0, 0.0], [0.5, 0.5]])
        pops_exc = np.array([[0.0, 1.0], [0.2, 0.8]])
        H = np.array([[100.0, 10.0], [10.0, 200.0]])
        U = np.eye(2, dtype=np.complex128)
        d1 = _dual_data("a.h5", pops_site, pops_exc, H, U)
        d2 = _dual_data("b.h5", pops_site * 0.9, pops_exc * 0.8, H.copy(), U.copy())
        result = create_comparison_result(d1, d2)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "dual.png"
            with mock.patch(
                "heom_tools.population.plot_compare.load_compare_diff_overlay",
                return_value=None,
            ):
                fig, axes = compare_dual_basis(
                    result, output_filename=str(out), show=False
                )
            self.assertEqual(axes[0, 0].get_title(), "Site Basis")
            self.assertEqual(axes[0, 1].get_title(), "Exciton Basis")
            self.assertEqual(axes[1, 0].get_title(), "Difference (2 − 1) · Site")
            self.assertEqual(axes[1, 1].get_title(), "Difference (2 − 1) · Exciton")
            plt.close(fig)

    def test_sink_curves_appear_in_legend_after_refresh(self) -> None:
        t = np.array([0.0, 10.0])
        pops = np.array([[1.0, 0.0], [0.5, 0.5]])
        d1 = PopulationData(
            filename="a.h5",
            time=t,
            populations_site=pops,
            site_labels=["Site1", "Site2"],
            sink_enabled=True,
            sink_site=0,
            N_captured=np.array([0.0, 0.2]),
            Phi=0.2,
        )
        d2 = _site_data("b.h5", pops * 0.9)
        d1.ensure_labels()

        fig, ax = plt.subplots()
        colors = plt.cm.tab10(np.linspace(0, 1, 2))
        plot_comparison_panel(ax, t, pops, pops * 0.9, d1.site_labels, colors)
        self.assertTrue(add_sink_curves(ax, d1, d2))
        refresh_legend(ax, 2)
        labels = [txt.get_text() for txt in ax.get_legend().get_texts()]
        self.assertTrue(any("N_cap" in lab for lab in labels))
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
