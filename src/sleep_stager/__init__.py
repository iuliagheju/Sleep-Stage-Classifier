"""Sleep staging pipeline package."""

from importlib import metadata

try:  # pragma: no cover
    __version__ = metadata.version("sleep-stager")
except metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.1.0"
