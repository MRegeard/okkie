import logging

from gammapy.modeling import Parameters

log = logging.getLogger(__name__)

__all__ = ["gammapy_build_parameters_from_dict"]


def gammapy_build_parameters_from_dict(data, default_parameters):
    """Build a `~gammapy.modeling.Parameters` object from input dictionary and default parameter values.
    Reimplementation of `~gammapy.modeling.models.core._build_parameters_from_dict` which is a private function.
    """
    par_data = []

    input_names = [_["name"] for _ in data]

    for par in default_parameters:
        par_dict = par.to_dict()
        try:
            index = input_names.index(par_dict["name"])
            par_dict.update(data[index])
        except ValueError:
            log.warning(
                f"Parameter '{par_dict['name']}' not defined in YAML file."
                f" Using default value: {par_dict['value']} {par_dict['unit']}"
            )
        par_data.append(par_dict)

    return Parameters.from_dict(par_data)
