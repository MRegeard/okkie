from gammapy.utils.registry import Registry

from .extra import Sigmoid
from .integral import (
    integrate_asymm_gaussian,
    integrate_asymm_lorentzian,
    integrate_gaussian,
    integrate_lorentzian,
    integrate_periodic_asymm_gaussian,
    integrate_periodic_asymm_lorentzian,
    integrate_periodic_gaussian,
    integrate_periodic_lorentzian,
    integrate_trapezoid,
)
from .luminosity import LuminosityModel
from .phase import (
    AsymmetricGaussianNormPhaseModel,
    AsymmetricGaussianPhaseModel,
    AsymmetricLorentzianNormPhaseModel,
    AsymmetricLorentzianPhaseModel,
    CompoundPhaseModel,
    ConstantPhaseModel,
    GatePhaseModel,
    GaussianNormPhaseModel,
    GaussianPhaseModel,
    LorentzianNormPhaseModel,
    LorentzianPhaseModel,
    PhaseModel,
    ScalePhaseModel,
    TemplatePhaseModel,
)
from .source import SourceModel
from .utils import sum_models, validate_ene

__all__ = [
    "PhaseModel",
    "ConstantPhaseModel",
    "CompoundPhaseModel",
    "LorentzianPhaseModel",
    "AsymmetricLorentzianPhaseModel",
    "GaussianPhaseModel",
    "AsymmetricGaussianPhaseModel",
    "SourceModel",
    "TemplatePhaseModel",
    "ScalePhaseModel",
    "GatePhaseModel",
    "integrate_trapezoid",
    "integrate_gaussian",
    "integrate_lorentzian",
    "integrate_asymm_gaussian",
    "integrate_asymm_lorentzian",
    "integrate_periodic_gaussian",
    "integrate_periodic_lorentzian",
    "integrate_periodic_asymm_gaussian",
    "integrate_periodic_asymm_lorentzian",
    "LuminosityModel",
    "sum_models",
    "validate_ene",
    "Sigmoid",
    "GaussianNormPhaseModel",
    "AsymmetricLorentzianNormPhaseModel",
    "LorentzianNormPhaseModel",
    "AsymmetricLorentzianNormPhaseModel",
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
