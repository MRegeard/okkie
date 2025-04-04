import astropy.units as u
from astropy.table import Table
from naima.extern.validator import validate_array, validate_physical_type

__all__ = ["sum_models", "validate_ene"]


def sum_models(models, model_type):
    """Sum all models."""
    summ = getattr(models[0], f"{model_type}_model")
    for m in models[1:]:
        summ += getattr(m, f"{model_type}_model")
    return summ


def validate_ene(ene):
    if isinstance(ene, dict or Table):
        try:
            ene = validate_array(
                "energy", u.Quantity(ene["energy"]), physical_type="energy"
            )
        except KeyError:
            raise TypeError("Table or dict does not have 'ene' column")
    else:
        if not isinstance(ene, u.Quantity):
            ene = u.Quantity(ene)
        validate_physical_type("energy", ene, physical_type="energy")

    return ene
