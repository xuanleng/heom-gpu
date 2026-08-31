"""CLI entry smoke tests for las-plot / las-compare."""
from __future__ import annotations

from heom_tools.absorption.plot.cli import main as las_plot_main
from heom_tools.absorption.plot.compare_cli import main as las_compare_main


def test_las_plot_rejects_two_files(capsys) -> None:
    code = las_plot_main(["a_spectrum.h5", "b_spectrum.h5"])
    assert code == 1
    err = capsys.readouterr().err
    assert "las-compare" in err


def test_las_compare_rejects_one_file(capsys) -> None:
    code = las_compare_main(["a_spectrum.h5"])
    assert code == 1
    err = capsys.readouterr().err
    assert "las-plot" in err
    assert "a_spectrum.h5" in err


def test_las_compare_rejects_zero_files(capsys) -> None:
    code = las_compare_main([])
    assert code == 1
    assert "至少两个" in capsys.readouterr().err


def test_las_plot_rejects_zero_files(capsys) -> None:
    code = las_plot_main([])
    assert code == 1
    assert "恰好一个" in capsys.readouterr().err
