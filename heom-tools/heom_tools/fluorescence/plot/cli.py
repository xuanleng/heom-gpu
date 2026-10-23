"""flu-plot: redraw the emission figure from a flu-post ``*_spectrum.h5``."""
from __future__ import annotations

import argparse
import os
from typing import Optional

from .plot import EmissionView, show_emission_plot, write_emission_plot


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("h5", help="flu-post output (<stem>_spectrum.h5)")
    p.add_argument("--out", default=None, help="PNG path (default <stem>_fl.png next to H5)")
    p.add_argument("--show", action="store_true", help="open a Tk window instead of saving")
    args = p.parse_args(argv)

    view = EmissionView.from_processed_h5(args.h5)
    if args.show:
        show_emission_plot(view)
        return 0
    stem = os.path.splitext(os.path.abspath(args.h5))[0].removesuffix("_spectrum")
    out = args.out or f"{stem}_fl.png"
    print(f"[flu-plot] wrote {write_emission_plot(view, out)}")
    return 0
