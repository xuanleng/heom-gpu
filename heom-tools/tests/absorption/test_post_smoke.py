"""``las-post`` end-to-end smoke test (synthetic solver HDF5 → ``*_spectrum.h5``)."""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
import warnings

import h5py
import numpy as np

from heom_tools.absorption.post.cli import main as absorption_main


def _write_minimal_absorption_input(path: str) -> None:
    """Write a synthetic raw absorption HDF5 in the C++ writer's group layout.

    The post-processor only consumes ``/parameters``, ``/derived``, and
    ``/results``; optional ``parameters/config_yaml`` is echoed by the CLI.
    """
    n_outputs = 16
    n_pigments = 2
    n_hilbert = n_pigments + 1

    time_fs = np.arange(n_outputs, dtype=np.float64) * 2.0
    rho = np.zeros((3, n_outputs, n_hilbert, n_hilbert), dtype=np.complex128)
    for alpha in range(3):
        for t in range(n_outputs):
            decay = np.exp(-0.05 * t)
            rho[alpha, t, 1, 0] = (alpha + 1) * decay
            rho[alpha, t, 2, 0] = 0.5 * (alpha + 1) * decay * np.exp(0.1j * t)

    evec = np.eye(n_hilbert, dtype=np.complex128)

    with h5py.File(path, "w") as f:
        # Root identity: enough for heom_h5.task() to claim "absorption".
        f.attrs["task"] = "absorption"
        f.attrs["config_hash"] = "test_absorption_post_synthetic"
        f.attrs["simulation_time_seconds"] = 2.25

        params = f.create_group("/parameters")
        params.attrs["spectrum_type"] = "absorption"
        params.attrs["basis"] = "with_ground"
        params.attrs["n_steps"] = n_outputs - 1
        params.attrs["output_interval"] = 2
        params.attrs["dt"] = 1.0
        params.attrs["temperature"] = 77.0
        params.attrs["w0_cm"] = 12000.0
        params.attrs["eta_cm"] = 35.0
        params.attrs["gamma_cm"] = 53.0
        params.create_dataset("dipoles", data=np.array([[1.0, 0.0, 0.0],
                                                        [0.0, 1.0, 0.0]]))
        params.create_dataset("strengths", data=np.array([1.0, 0.8]))
        params.create_dataset("hamiltonian_cm", data=np.diag([0.0, 100.0, 200.0]))
        params.create_dataset(
            "config_yaml",
            data="# comment line\n\nsystem:\n  n_states: 2\n  w0: 12000.0  # inline\n\n",
        )

        derived = f.create_group("/derived")
        derived.attrs["M_total"] = 42
        derived.create_dataset("eigenvalues_relative_cm",
                               data=np.array([0.0, 100.0, 200.0]))
        # /derived/eigenvectors uses h5py's native complex compound; matches
        # the C++ ``make_complex_compound_type()`` field names {r, i}.
        derived.create_dataset("eigenvectors", data=evec)

        results = f.create_group("/results")
        results.create_dataset("time", data=time_fs)
        results.create_dataset("rho_xyz", data=rho)

        f.create_group("/provenance").attrs.update(
            cpu_model="Test CPU",
            gpu_model="Test GPU",
            os="Test OS",
            cuda_version="13.0",
            compiler="GCC 11.4.0",
            config_path="./config/fmo.yaml",
        )


def test_las_post_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        source = os.path.join(tmp, "abs_source.h5")
        expected_out = os.path.join(tmp, "abs_source_spectrum.h5")
        _write_minimal_absorption_input(source)
        with h5py.File(source, "a") as f:
            # Post-processing may see hand-written / legacy files. Match the
            # C++ parser: normalize directions, warn, and keep magnitudes in
            # strengths instead of silently scaling twice.
            f["/parameters/dipoles"][...] = np.array([[2.0, 0.0, 0.0],
                                                       [0.0, 3.0, 0.0]])

        try:
            import matplotlib  # noqa: F401
            plots_available = True
        except Exception:
            plots_available = False

        cli = [
            source,
            "--wmin", "11800",
            "--wmax", "12500",
        ]
        if not plots_available:
            cli.append("--no-plots")

        stdout = io.StringIO()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with contextlib.redirect_stdout(stdout):
                rc = absorption_main(cli)
        if rc != 0:
            raise AssertionError(f"las-post returned {rc}")
        out_text = stdout.getvalue()
        assert "[run parameters]" not in out_text
        assert "[source] abs_source.h5" in out_text
        assert "  input = " in out_text
        assert "  output = " in out_text
        assert "_spectrum.h5" in out_text
        assert "[run_info]" in out_text
        assert "  M_total = 42" in out_text
        assert "  cpu_model = Test CPU" in out_text
        assert "  simulation_time = 2.250s" in out_text
        assert "config_path" not in out_text
        assert "[config_yaml]" in out_text
        assert "# comment line" not in out_text
        assert "# inline" not in out_text
        assert "  system:" in out_text
        assert "    n_states: 2" in out_text
        assert "    w0: 12000.0" in out_text
        assert "\n\n\n" not in out_text
        assert "[source HDF5 metadata]" not in out_text
        assert "dataset results/rho_xyz" not in out_text
        assert "dephasing gamma" not in out_text
        warning_text = "\n".join(str(w.message) for w in caught)
        assert "dipoles[0] is not unit length" in warning_text
        assert "dipoles[1] is not unit length" in warning_text

        with h5py.File(expected_out, "r") as f:
            assert f.attrs["spectrum_type"] == "absorption"
            assert f.attrs["source_h5"].endswith("abs_source.h5")
            assert "dephasing_enabled" not in f.attrs
            assert "gamma_dephasing_cm" not in f.attrs
            assert "pad_factor" not in f.attrs
            assert f["omega_cm"].ndim == 1
            assert f["intensity_total"].shape == f["omega_cm"].shape
            assert f["intensity_per_exciton"].shape[0] == f["omega_cm"].shape[0]
            assert f["intensity_per_exciton"].shape[1] == 2
            assert f["correlation/ct_iso_real"].shape == (16,)
            assert "correlation/ct_iso_windowed_real" not in f
            assert "correlation/ct_iso_windowed_imag" not in f
            assert f["correlation/ct_iso_per_exciton_real"].shape == (16, 2)
            assert f["sticks/table"].shape == (2, 4)
            assert "peaks/total" in f
            assert "source/dipoles" in f
            np.testing.assert_allclose(
                f["source/dipoles"][...],
                np.array([[1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0]]),
            )

        plot_stem = os.path.join(tmp, "abs_source")
        if plots_available:
            for suf in ("_spectrum.png", "_corr.png"):
                png = plot_stem + suf
                if not os.path.isfile(png):
                    raise AssertionError(f"missing PNG: {png}")
                if os.path.getsize(png) <= 0:
                    raise AssertionError(f"empty PNG: {png}")
        else:
            assert "plots skipped by --no-plots" in out_text


def test_las_post_auto_no_args() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        source = os.path.join(tmp, "abs_source.h5")
        _write_minimal_absorption_input(source)
        old = os.getcwd()
        try:
            os.chdir(tmp)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = absorption_main(["--no-plots"])
        finally:
            os.chdir(old)
        assert rc == 0
        assert "[las-post auto]" in stdout.getvalue()
        assert os.path.isfile(os.path.join(tmp, "abs_source_spectrum.h5"))


def test_las_post_bare_invocation_runs_auto() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        source = os.path.join(tmp, "abs_source.h5")
        _write_minimal_absorption_input(source)
        old = os.getcwd()
        try:
            os.chdir(tmp)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = absorption_main([])
        finally:
            os.chdir(old)
        assert rc == 0
        assert "[las-post auto]" in stdout.getvalue()
        assert os.path.isfile(os.path.join(tmp, "abs_source_spectrum.h5"))
