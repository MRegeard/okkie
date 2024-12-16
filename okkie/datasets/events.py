import logging

import numpy as np
from gammapy.datasets import Dataset
from gammapy.modeling.models import DatasetModels, Models
from gammapy.stats.fit_statistics_cython import TRUNCATION_VALUE
from gammapy.utils.scripts import make_name
from scipy.integrate import quad

log = logging.getLogger(__name__)


class EventsDataset(Dataset):
    tag = "EventsDataset"
    stat_type = "unbinned"

    def __init__(
        self,
        events,
        models=None,
        name=None,
        meta_table=None,
        mask_fit=None,
        mask_safe=None,
    ) -> None:
        self.events = events
        self.models = models
        self._name = make_name(name)
        self.meta_table = meta_table
        self.mask_fit = mask_fit
        self.mask_safe = mask_safe

    @property
    def models(self) -> Models:
        """Models set on the dataset as a `~gammapy.modeling.models.Models`."""
        return self._models

    @models.setter
    def models(self, models: Models) -> None:
        """Models setter."""
        if models is not None:
            models = DatasetModels(models)
            models = models.select(datasets_names=self.name)

        self._models = models

    def stat_sum(self):
        # TODO: implement prior
        """Total statistic function value given the current parameters."""
        total = 0.0

        for model in self.models:
            phase_model = model.phase_model
            npred = phase_model(phase=self.events.table["PHASE"]).value
            #            integral = phase_model.integral(0, 1)
            integral = quad(
                phase_model,
                0.0,
                1.0,
            )[0]
            #            log.warning(f"model is {model.name}")
            #            log.warning(f"integral is {integral}")
            npred = np.where(npred <= TRUNCATION_VALUE, TRUNCATION_VALUE, npred)
            total += np.log(integral) * len(npred) - np.sum(npred)
        return -2 * total

    def stat_array(self):
        pass

    @property
    def event_mask(self):
        """Entry for each event whether it is inside the mask or not"""
        if self.mask is None:
            return np.ones(len(self.events.table), dtype=bool)
        coords = self.events.map_coord(self.mask.geom)
        return self.mask.get_by_coord(coords) == 1

    @property
    def events_in_mask(self):
        return self.events.select_row_subset(self.event_mask)


""""
    def stat_sum(self):
        # TODO: implement prior
        Total statistic function value given the current parameters

        response = np.zeros(len(self.events.table))
        total = 0.0

        for model in self.models:
            phase_model = model.phase_model
            npred = phase_model(phase=self.events.table["PHASE"]).value
            integral = phase_model.integral(0, 1)
            #            npred /= integral
            npred_total = np.sum(npred)
            response += npred
            total += npred_total
        response = np.where(response <= TRUNCATION_VALUE, TRUNCATION_VALUE, response)

        logL = np.sum(np.log(response)) - total
        return -2 * logL

"""
