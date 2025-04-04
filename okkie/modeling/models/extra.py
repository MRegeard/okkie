import numpy as np
from naima.extern.validator import validate_scalar

from .utils import validate_ene

__all__ = ["Sigmoid"]


class Sigmoid:
    """"""

    param_names = ["norm", "reference", "delta"]
    _cache = {}
    _queue = []

    def __init__(self, norm, reference, delta):
        self.norm = norm
        self.reference = validate_scalar(
            "reference", reference, domain="positive", physical_type="energy"
        )
        self.delta = validate_scalar(
            "delta", delta, domain="positive", physical_type="energy"
        )

    @staticmethod
    def eval(energy, norm, reference, delta):
        return norm / (1 + np.exp((energy - reference) / delta))

    def _calc(self, energy):
        return self.eval(
            energy.to("eV").value,
            self.norm,
            self.reference.to("eV").value,
            self.delta.to("eV").value,
        )

    def __call__(self, energy):
        energy = validate_ene(energy)
        return self._calc(energy)
