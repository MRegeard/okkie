from gammapy.utils.registry import Registry
from .phase import (
    AsymetricGaussianPhaseModel,
    AsymmetricLorentzianPhaseModel,
    CompoundPhaseModel,
    ConstantPhaseModel,
    GaussianPhaseModel,
    LorentzianPhaseModel,
    PhaseModel,
)
from .source import SourceModel

__all__ = [
    "PhaseModel",
    "ConstantPhaseModel",
    "CompoundPhaseModel",
    "LorentzianPhaseModel",
    "AsymmetricLorentzianPhaseModel",
    "GaussianPhaseModel",
    "AsymetricGaussianPhaseModel",
    "SourceModel",
]


PHASE_MODEL_REGISTRY = Registry(
    [
        ConstantPhaseModel,
        CompoundPhaseModel,
        LorentzianPhaseModel,
        AsymmetricLorentzianPhaseModel,
        GaussianPhaseModel,
        AsymetricGaussianPhaseModel,
    ]
)
"""Registry of phase model classes."""
