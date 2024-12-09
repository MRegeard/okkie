from gammapy.utils.registry import Registry

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

__all__ = [
    "PhaseModel",
    "ConstantPhaseModel",
    "CompoundPhaseModel",
    "LorentzianPhaseModel",
    "AsymmetricLorentzianPhaseModel",
    "GaussianPhaseModel",
    "AsymmetricGaussianPhaseModel",
    "SourceModel",
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
