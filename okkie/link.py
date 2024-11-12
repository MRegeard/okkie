from gammapy.modeling.models import MODEL_REGISTRY

from okkie.modeling.models import SourceModel

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
