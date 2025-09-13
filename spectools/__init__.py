"""
spectools: A set of tools for handling astronomical spectroscopic data in Python.
================================================

Documentation is available online at https://github.com/michaeljennings11/spectools.

"""

from spectools.version import version as __version__

submodules = ["line_data", "line_model", "constants", "utils"]

__all__ = submodules + [
    "__version__",
]


def __dir__():
    return __all__
