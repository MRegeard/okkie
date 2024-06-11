from astropy.io import fits
from gammapy.utils.scripts import make_path

__all__ = ["obs_write"]


def obs_write(
    self, path, overwrite=False, format="gadf", include_irfs=True, checksum=False
):
    """
    Write this observation into `~pathlib.Path` using the specified format. With event time fix !
    Parameters
    ----------
    path : str or `~pathlib.Path`
        Path for the output file.
    overwrite : bool, optional
        Overwrite existing file. Default is False.
    format : {"gadf"}
        Output format, currently only "gadf" is supported. Default is "gadf".
    include_irfs : bool, optional
        Whether to include irf components in the output file. Default is True.
    checksum : bool, optional
        When True adds both DATASUM and CHECKSUM cards to the headers written to the file.
        Default is False.
    """
    if format != "gadf":
        raise ValueError(f'Only the "gadf" format is supported, got {format}')
    path = make_path(path)
    primary = fits.PrimaryHDU()
    primary.header.update(self.meta.creation.to_header(format))
    hdul = fits.HDUList([primary])
    events = self.events
    if events is not None:
        events_hdu = events.to_table_hdu(format=format)
        events_hdu.header.update(self.pointing.to_fits_header())
        events_hdu.header.update(self.pointing.to_fits_header(time_ref=events.time_ref))
        hdul.append(events_hdu)

    gti = self.gti
    if gti is not None:
        hdul.append(gti.to_table_hdu(format=format))
    if include_irfs:
        for irf_name in self.available_irfs:
            irf = getattr(self, irf_name)
            if irf is not None:
                hdul.append(irf.to_table_hdu(format="gadf-dl3"))
    hdul.writeto(path, overwrite=overwrite, checksum=checksum)
