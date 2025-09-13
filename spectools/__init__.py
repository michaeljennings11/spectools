"""
spectools: A set of tools for handling astronomical spectroscopic data in Python.
================================================

Documentation is available online at https://github.com/michaeljennings11/spectools.

"""

from spectools.version import version as __version__

submodules = [
    "get_Elevels",
    "level_diagram",
    "load_elementDataFrame",
    "load_ionDataFrame",
    "load_lineDataFrame",
]

__all__ = submodules + [
    "LineData",
    "__version__",
]


def __dir__():
    return __all__
