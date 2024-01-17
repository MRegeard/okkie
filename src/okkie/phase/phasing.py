import logging
import astropy.units as u
import numpy as np
import pint
import pint.models as pmodels
from astropy.time import Time
from gammapy.data import EventList
from gammapy.utils.scripts import make_path
from pint import toa

log = logging.getLogger(__name__)


class CPhaseMaker:
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
