import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import quantity_support
from gammapy.maps import MapAxis, RegionNDMap
from gammapy.maps.axes import UNIT_STRING_FORMAT
from gammapy.modeling.models import SpectralModel, scale_plot_flux

__all__ = ["LuminosityModel"]


class LuminosityModel:
    def __init__(self, model, distance):
        self.model = model
        self.distance = distance

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        if not isinstance(value, SpectralModel):
            raise TypeError(
                f"`model` must be an instance of `~gammapy.modeling.models.SpectralModel`, got {type(value)}."
            )
        self._model = value

    @property
    def distance(self):
        return self._distance

    @distance.setter
    def distance(self, value):
        if not isinstance(value, u.Quantity):
            raise TypeError(
                f"`distance` must be an instance of `~astropy.units.Quantity`, got {type(value)}."
            )
        self._distance = value

    def __call__(self, energy):
        return self.model(energy) * 4 * np.pi * self.distance**2

    def plot_error(
        self,
        energy_bounds,
        ax=None,
        n_points=100,
        **kwargs,
    ):
        if isinstance(energy_bounds, (tuple, list, u.Quantity)):
            energy_min, energy_max = energy_bounds
            energy = MapAxis.from_energy_bounds(
                energy_min,
                energy_max,
                n_points,
                name="energy",
            )
        elif isinstance(energy_bounds, MapAxis):
            energy = energy_bounds

        ax = plt.gca() if ax is None else ax

        kwargs.setdefault("facecolor", "black")
        kwargs.setdefault("alpha", 0.2)
        kwargs.setdefault("linewidth", 0)

        if ax.yaxis.units is None:
            ax.yaxis.set_units(u.Unit("erg s-1"))

        flux, flux_err = self._get_plot_flux(energy=energy)
        y_lo = scale_plot_flux(flux - flux_err, energy_power=0).quantity[:, 0, 0]
        y_hi = scale_plot_flux(flux + flux_err, energy_power=0).quantity[:, 0, 0]

        with quantity_support():
            ax.fill_between(energy.center, y_lo, y_hi, **kwargs)

        self._plot_format_ax(ax)
        return ax

    def plot(
        self,
        energy_bounds,
        ax=None,
        n_points=100,
        **kwargs,
    ):
        if isinstance(energy_bounds, (tuple, list, u.Quantity)):
            energy_min, energy_max = energy_bounds
            energy = MapAxis.from_energy_bounds(
                energy_min,
                energy_max,
                n_points,
                name="energy",
            )
        elif isinstance(energy_bounds, MapAxis):
            energy = energy_bounds

        ax = plt.gca() if ax is None else ax

        if ax.yaxis.units is None:
            ax.yaxis.set_units(u.Unit("erg s-1"))

        flux, _ = self._get_plot_flux(energy=energy)

        flux = scale_plot_flux(flux, energy_power=0)

        with quantity_support():
            ax.plot(energy.center, flux.quantity[:, 0, 0], **kwargs)

        self._plot_format_ax(ax)
        return ax

    def evaluate_error(self, energy, **kwargs):
        return (
            4 * np.pi * self.distance**2 * self.model.evaluate_error(energy, **kwargs)
        )

    def _get_plot_flux(self, energy):
        flux = RegionNDMap.create(region=None, axes=[energy])
        flux_err = RegionNDMap.create(region=None, axes=[energy])
        flux.quantity, flux_err.quantity = energy.center**2 * self.evaluate_error(
            energy.center
        )
        return flux, flux_err

    @staticmethod
    def _plot_format_ax(ax):
        ax.set_xlabel(f"Energy [{ax.xaxis.units.to_string(UNIT_STRING_FORMAT)}]")
        ax.set_ylabel(f"Luminosity [{ax.yaxis.units.to_string(UNIT_STRING_FORMAT)}]")

        ax.set_xscale("log", nonpositive="clip")
        ax.set_yscale("log", nonpositive="clip")
