import logging
import astropy.units as u
import numpy as np
import pint.models as pmodels
from astropy.time import Time
from gammapy.data import EventList
from pint import toa

log = logging.getLogger(__name__)


class CPhaseMaker:
    def __init__(
        self,
        pint_model,
        observatory,
        errors=1 * u.us,
        ephem="DE421",
        include_bipm=True,
        include_gps=True,
        planets=True,
    ):
        self._pint_model = pint_model
        self.observatory = observatory
        self.errors = errors
        self.ephem = ephem
        self.include_bipm = include_bipm
        self.include_gps = include_gps
        self.planets = planets

    @property
    def pint_model(self):
        return self._pint_model

    @pint_model.setter
    def pint_model(self, model):
        if not isinstance(model, pmodels.TimingModel):
            raise TypeError("Model needs to be an instance of TimingModel.")
        else:
            self._pint_model = model

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

        phases = self.pint_model.phase(toas, abs_phase=True)[1]
        phases = np.where(phases < 0.0, phases + 1.0, phases)

        return phases

    def _check_times(self, observation):
        time = observation.events.time
        time_min = time.min().tt.mjd
        time_max = time.max().tt.mjd

        model_time_range = Time(
            [self.pint_model.START.value, self.pint_model.FINISH.value],
            scale="tt",
            format="mjd",
        )

        if time_min < model_time_range.value or time_max > model_time_range.value:
            log.warning(
                f"At least one of the time of observation: {observation.obs_id} is outside of the validity range of the timing model"
            )
        return time

    def run(self, observation, column_name="PHASE"):
        table = observation.events.table

        phases = self.compute_phases(observation)
        table[column_name] = phases.astype("float64")

        new_events = EventList(table)

        new_observation = observation.copy(in_memory=True, events=new_events)

        return new_observation

    def metadata(self):
        pass
