"""CLI entry smoke tests for pop-plot / pop-compare."""
from __future__ import annotations

from heom_tools.population.cli import main as pop_plot_main
from heom_tools.population.compare_cli import main as pop_compare_main


def test_pop_plot_rejects_two_files(capsys) -> None:
    code = pop_plot_main(["a.h5", "b.h5"])
    assert code == 1
    err = capsys.readouterr().err
    assert "pop-compare" in err
    assert "a.h5" in err and "b.h5" in err


def test_pop_compare_rejects_one_file(capsys) -> None:
    code = pop_compare_main(["a.h5"])
    assert code == 1
    err = capsys.readouterr().err
    assert "pop-plot" in err
    assert "a.h5" in err


def test_pop_compare_rejects_zero_files(capsys) -> None:
    code = pop_compare_main([])
    assert code == 1
    assert "恰好两个文件" in capsys.readouterr().err


def test_pop_plot_rejects_zero_files(capsys) -> None:
    code = pop_plot_main([])
    assert code == 1
    assert "恰好一个文件" in capsys.readouterr().err
