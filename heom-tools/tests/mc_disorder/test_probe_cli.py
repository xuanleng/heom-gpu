"""CLI contract tests for mc-probe argparse subparsers (no GPU / no solver).

Locks the post-refactor shape: each subcommand only accepts its own knobs,
documented invocations still parse, and foreign flags are rejected.
"""
import pytest

from heom_tools.mc_disorder.probe import _build_parser


@pytest.fixture
def parser():
    return _build_parser()


def test_top_level_lists_four_subcommands(parser):
    with pytest.raises(SystemExit):
        parser.parse_args([])  # required subcommand
    # help text is the user-facing contract
    help_txt = parser.format_help()
    for name in ("absorption", "arena", "converge", "rqmc"):
        assert name in help_txt


def test_absorption_accepts_levels_rejects_pool_file(parser):
    a = parser.parse_args(
        ["absorption", "c.yaml", "--fwhm", "100", "--levels", "0", "1",
         "--iid", "1", "8"])
    assert a.task == "absorption"
    assert a.levels == [0, 1]
    assert a.iid == [1, 8]
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["absorption", "c.yaml", "--fwhm", "100", "--pool-file", "x.h5"])


def test_arena_accepts_pool_file_and_stack(parser):
    a = parser.parse_args(
        ["arena", "c.yaml", "--fwhm", "100", "--per-site", "3:50",
         "--sweep", "8", "16", "--stack", "--pool-file", "arena.h5",
         "--ref-file", "gold.h5"])
    assert a.task == "arena"
    assert a.stack is True
    assert a.pool_file == "arena.h5"
    assert a.ref_file == "gold.h5"
    assert a.per_site == "3:50"
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["arena", "c.yaml", "--fwhm", "100", "--levels", "0"])


def test_converge_documented_invocation(parser):
    # Expected output shape.
    a = parser.parse_args(
        ["converge", "config/fmo.yaml", "--fwhm", "100", "--per-site", "3:50",
         "--workers", "4", "--pool-size", "4096", "--pool-file", "x.h5",
         "--ref-file", "g.h5", "--tol", "0.01", "0.02", "--out", "c.csv"])
    assert a.task == "converge"
    assert a.pool_size == 4096
    assert a.tol == [0.01, 0.02]
    assert a.workers == 4
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["converge", "c.yaml", "--fwhm", "100", "--n-replica", "4"])


def test_rqmc_documented_invocation_and_2des_knobs(parser):
    a = parser.parse_args(
        ["rqmc", "config/fmo.yaml", "--fwhm", "100", "--per-site", "3:50",
         "--sweep", "8", "16", "32", "--n-replica", "8",
         "--pool-file", "r.h5", "--iid-pool", "c.h5", "--ref-file", "g.h5"])
    assert a.task == "rqmc"
    assert a.n_replica == 8
    assert a.obs == "absorption"
    d2 = parser.parse_args(
        ["rqmc", "c.yaml", "--fwhm", "100", "--obs", "2des", "--ref-n", "0",
         "--dw", "16", "--pol", "ap", "--tw-index", "0"])
    assert d2.obs == "2des" and d2.ref_n == 0 and d2.dw == 16.0
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["rqmc", "c.yaml", "--fwhm", "100", "--levels", "0"])


def test_rqmc_rejects_cv_width(parser):
    # cv-width is arena/converge only — isolation must hold.
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["rqmc", "c.yaml", "--fwhm", "100", "--cv-width", "60"])
