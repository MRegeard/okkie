from astropy.time import Time
import pint.models as pmodels
from pint import toa
import logging
import numpy as np

log = logging.getLogger(__name__)


class CPhaseMaker:
    def __init__(
        self,
        pint_model,
        observatory,
        errors,
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
            raise TypeError("model needs to be an instance of TimingModel.")
        else:
            self._pint_model = model

    def run(self, observation):
        time = self._check_times(observation)
        toas = toa.get_TOAs_array(
            time=time,
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
            [self.model.START.value, self.model.FINISH.value], scale="tt", format="mjd"
        )

        if time_min < model_time_range.value or time_max > model_time_range.value:
            log.warning(
                f"At least one of the time of observation: {observation.obs_id} is outside of the validity range of the timing model"
            )
        return time
