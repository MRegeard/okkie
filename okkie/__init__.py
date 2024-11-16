from .link import link

__all__ = ["__version__", "link"]


from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version(__name__)
except PackageNotFoundError:
    # package is not installed
    pass
del _version, PackageNotFoundError
