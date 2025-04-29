from gammapy.modeling.models import MODEL_REGISTRY, SPECTRAL_MODEL_REGISTRY

from okkie.modeling.models import (
    SourceModel,
    SuperExpCutoffPowerLaw4FGLDR3SpectralModelCor,
)

PACKAGE = ["gammapy"]


def link(package_name):
    """Use to link okkie functionality to other packages classes that okkie classes inherit from.
    This is particularly usefull for deserialisation.

    Parameters
    ----------
    package_name: `str`
        Name of package, {gammapy}.
    """
    if package_name.lower() not in PACKAGE:
        raise ValueError(f"Nothing to link for package {package_name}.")
    if package_name.lower() == "gammapy":
        try:
            MODEL_REGISTRY.get_cls("SourceModel")
        except KeyError:
            MODEL_REGISTRY.append(SourceModel)

        try:
            SPECTRAL_MODEL_REGISTRY.get_cls(
                "SuperExpCutoffPowerLaw4FGLDR3SpectralModelCor"
            )
        except KeyError:
            SPECTRAL_MODEL_REGISTRY.append(
                SuperExpCutoffPowerLaw4FGLDR3SpectralModelCor
            )
