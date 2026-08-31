"""``python -m heom_tools.absorption.plot`` entry point."""
from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
