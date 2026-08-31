"""Shared pytest hooks for heom_tools.mc_disorder."""
import importlib.util

import pytest


def pytest_collection_modifyitems(config, items):
    if importlib.util.find_spec("scipy") is not None:
        return
    skip = pytest.mark.skip(
        reason="scipy required for RQMC tests "
               "(pip install 'heom-tools[rqmc]')")
    for item in items:
        if "rqmc" in item.nodeid.lower():
            item.add_marker(skip)
