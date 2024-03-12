import logging
import astropy.units as u
import numpy as np
import pint
import pint.models as pmodels
from astropy.io import fits
from astropy.time import Time
from gammapy.data import EventList
from gammapy.utils.scripts import make_path
from pint import toa
from pint.fermi_toas import load_Fermi_TOAs
from pint.observatory.satellite_obs import get_satellite_observatory

log = logging.getLogger(__name__)


__all__ = ["PhaseMaker", "FermiPhaseMaker"]


class PhaseMaker:
    def __init__(
        self,
        ephemeris_file,
        observatory,
        errors=1 * u.us,
        ephem="DE421",
        include_bipm=True,
        include_gps=True,
        planets=True,
    ):
        self.ephemeris_file = make_path(ephemeris_file)
        self.observatory = observatory
        self.errors = errors
        self.ephem = ephem
        self.include_bipm = include_bipm
        self.include_gps = include_gps
        self.planets = planets
        self.model = pmodels.get_model(self.ephemeris_file)

    @property
    def pint_model(self):
        return self.model

    def show_model(self):
        print(self.model)

    def compute_phases(self, observation):
        time = self._check_times(observation)
        toas = toa.get_TOAs_array(
            times=time,
            obs=self.observatory,
            errors=self.errors,
            ephem=self.ephem,
            include_bipm=self.include_bipm,
            inlude_gps=self.include_gps,
            planets=self.planets,
        )

        phases = self.model.phase(toas, abs_phase=True)[1]
        phases = np.where(phases < 0.0, phases + 1.0, phases)

        return phases

    def _check_times(self, observation):
        time = observation.events.time
        time_min = time.min().tt.mjd
        time_max = time.max().tt.mjd

        model_time_range = Time(
            [self.model.START.value, self.model.FINISH.value],
            scale="tt",
            format="mjd",
        )

        if time_min < model_time_range.value or time_max > model_time_range.value:
            log.warning(
                f"At least one of the time of observation: {observation.obs_id} is outside of the validity range of the timing model"
            )
        return time

    def run(
        self,
        observation,
        column_name="PHASE",
        update_header=True,
        header_entry_name="PH_LOG",
    ):
        table = observation.events.table

        phases = self.compute_phases(observation)
        table[column_name] = phases.astype("float64")

        if update_header:
            table.meta[header_entry_name] = self.update_header()

        new_events = EventList(table)

        new_observation = observation.copy(in_memory=True, events=new_events)

        return new_observation

    def update_header(self, column_name="PHASE", **kwargs):
        # TODO: Make this customizable
        key_model = [
            "PSR",
            "START",
            "FINISH",
            "TZRMDJ",
            "TZRSITE",
            "TZRFREQ",
            "EPHEM",
            "DM",
            "RAJ",
            "DECJ",
        ]

        meta_dict = dict()
        meta_dict["COLUMN_NAME"] = column_name
        meta_dict["EPHEMERIS_FILE"] = str(self.ephemeris_file)
        meta_dict["PINT_VERS"] = pint.__version__

        for key in key_model:
            try:
                meta_dict[key] = getattr(self.model, key).value
            except AttributeError:
                log.warning(f"Could not find {key} in model, skipping.")
                meta_dict[key] = None

        meta_dict["CREATION_DATE"] = Time.now().mjd

        meta_dict.update(kwargs)

        return str(meta_dict)


class FermiPhaseMaker:
    """ """

    def __init__(
        self,
        fermi_event_file,
        fermi_spacecraft_file,
        ephemeris_file,
        ephem="DE421",
        include_bipm=False,
        include_gps=False,
        planets=False,
        weightcolumn=None,
        targetcoord=None,
    ):
        fermi_event_file = make_path(fermi_event_file)
        fermi_spacecraft_file = make_path(fermi_spacecraft_file)
        ephemeris_file = make_path(ephemeris_file)

        self.event_file = fermi_event_file
        self.spacecraft_file = fermi_spacecraft_file
        self.ephemeris_file = ephemeris_file
        self.model = pmodels.get_model(ephemeris_file)
        self.ephem = ephem
        self.include_bipm = include_bipm
        self.include_gps = include_gps
        self.planets = planets
        self.weightcolumn = weightcolumn
        self.targetcoord = targetcoord

    def compute_phase(self, **kwargs):
        get_satellite_observatory("Fermi", self.spacecraft_file, overwrite=True)

        toa_list = load_Fermi_TOAs(
            ft1name=self.event_file,
            weightcolumn=self.weightcolumn,
            targetcoord=self.targetcoord,
            **kwargs,
        )

        ts = toa.get_TOAs_list(
            toa_list=toa_list,
            ephem=self.ephem,
            include_bipm=self.include_bipm,
            include_gps=self.include_gps,
            planets=self.planets,
        )

        phases = self.model.phase(toas=ts, abs_phase=True)[1]
        self.phases = np.where(phases < 0.0, phases + 1.0, phases)

    def write_column_and_meta(
        self, filename=None, column_name="PULSE_PHASE", overwrite=False
    ):
        if filename is None:
            hdulist = fits.open(self.event_file, mode="update")
        else:
            hdulist = fits.open(self.event_file)

        event_hdu = hdulist[1]
        event_header = event_hdu.header
        event_data = event_hdu

        is_check = self._check_column_name(event_hdu=event_hdu, column_name=column_name)

        if is_check:
            phasecol = fits.ColDefs(
                [fits.Column(name=column_name, format="D", array=self.phases)]
            )
            event_header["PHSE_LOG"] = self.make_meta(
                self.ephemeris_file, self.model, column_name=column_name
            )
            bin_table = fits.BinTableHDU.from_columns(
                event_hdu.columns + phasecol, header=event_header, name=event_hdu.name
            )
            hdulist[1] = bin_table

        elif not is_check and overwrite:
            event_data.data[column_name] = self.phases
            event_header["PHSE_LOG"] = self.make_meta(
                self.ephemeris_file, self.model, column_name
            )

        elif not is_check and not overwrite:
            raise ValueError(
                f"Column named {column_name} already exist in file {self.event_file}"
                f"and overwrite is set to {overwrite}."
            )
        if filename is None:
            hdulist.flush(verbose=True, output_verify="warn")
        else:
            hdulist.writeto(
                filename, overwrite=True, checksum=True, output_verify="warn"
            )

    def run(
        self,
        filename,
        column_name="PULSE_PHASE",
        overwrite=True,
        kwargs_meta=None,
        kwargs_phase=None,
    ):
        kwargs_meta = kwargs_meta or {}
        kwargs_phase = kwargs_phase or {}

        self.compute_phase(**kwargs_phase)
        self.write_column_and_meta(
            filename=filename,
            column_name=column_name,
            overwrite=overwrite,
            **kwargs_meta,
        )

    @staticmethod
    def _check_column_name(event_hdu, column_name):
        if column_name not in event_hdu.columns.names:
            return True
        else:
            return False

    @staticmethod
    def make_meta(ephemeris_file, model, column_name="PULSE_NAME", offset=None):
        key_model = [
            "PSR",
            "START",
            "FINISH",
            "TZRMJD",
            "TZRSITE",
            "TZRFREQ",
            "EPHEM",
            "RAJ",
            "DECJ",
        ]

        meta_dict = dict()
        meta_dict["COLUMN_NAME"] = column_name
        meta_dict["EPHEMERIS_FILE"] = str(ephemeris_file)
        meta_dict["PINT_VERS"] = pint.__version__

        for key in key_model:
            try:
                meta_dict[key] = getattr(model, key).value
            except AttributeError:
                log.info(f"Key {key} not found in timing model, will be set to None.")
                meta_dict[key] = None

        meta_dict["PHASE_OFFSET"] = offset
        meta_dict["DATE"] = Time.now().mjd
