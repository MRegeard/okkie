from gammapy.utils.registry import Registry

from .luminosity import LuminosityModel
from .phase import (
    AsymmetricGaussianPhaseModel,
    AsymmetricLorentzianPhaseModel,
    CompoundPhaseModel,
    ConstantPhaseModel,
    GaussianPhaseModel,
    LorentzianPhaseModel,
    PhaseModel,
)
from .source import SourceModel
from .utils import (
    integrate_asymm_gaussian,
    integrate_asymm_lorentzian,
    integrate_gaussian,
    integrate_lorentzian,
    integrate_periodic_asymm_gaussian,
    integrate_periodic_asymm_lorentzian,
    integrate_periodic_gaussian,
    integrate_periodic_lorentzian,
)

__all__ = [
    "PhaseModel",
    "ConstantPhaseModel",
    "CompoundPhaseModel",
    "LorentzianPhaseModel",
    "AsymmetricLorentzianPhaseModel",
    "GaussianPhaseModel",
    "AsymmetricGaussianPhaseModel",
    "SourceModel",
    "integrate_gaussian",
    "integrate_lorentzian",
    "integrate_asymm_gaussian",
    "integrate_asymm_lorentzian",
    "integrate_periodic_gaussian",
    "integrate_periodic_lorentzian",
    "integrate_periodic_asymm_gaussian",
    "integrate_periodic_asymm_lorentzian",
    "LuminosityModel",
]


PHASE_MODEL_REGISTRY = Registry(
    [
        ConstantPhaseModel,
        CompoundPhaseModel,
        LorentzianPhaseModel,
        AsymmetricLorentzianPhaseModel,
        GaussianPhaseModel,
        AsymmetricGaussianPhaseModel,
    ]
)
"""Registry of phase model classes."""
