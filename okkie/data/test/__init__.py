from pathlib import Path

from astropy.utils.data import get_pkg_data_filename

__all__ = ["get_test_filepath"]


def get_test_filepath(filename, package="okkie.data.test", **kwargs):
    """
    Return the full path to a test file in the ``data/test`` directory.

    This function is copied from sunpy: `~sunpy.data.test.get_test_filepath`.

    Parameters
    ----------
    filename : `str`
        The name of the file inside the ``data/test`` directory.
    package : `str`, optional
        The package in which to look for the file. Defaults to "okkie.data.test".

    Returns
    -------
    filepath : `str`
        The full path to the file.

    Notes
    -----
    This is a wrapper around `astropy.utils.data.get_pkg_data_filename` which
    sets the ``package`` kwarg to be 'sunpy.data.test`.
    """
    if isinstance(filename, Path):
        # NOTE: get_pkg_data_filename does not accept Path objects
        filename = filename.as_posix()
    return get_pkg_data_filename(filename, package=package, **kwargs)
