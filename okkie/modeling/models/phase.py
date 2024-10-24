import logging
import operator
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from gammapy.maps import MapAxis
from gammapy.modeling import Parameter
from gammapy.modeling.models import ModelBase

log = logging.getLogger(__name__)


class PhaseModel(ModelBase):
    """Phase model base class."""

    _type = "phase"

    def __call__(self, phase):
        kwargs = {par.name: par.quantity for par in self.parameters}
        return self.evaluate(phase, **kwargs)

    def __add__(self, model):
        if not isinstance(model, PhaseModel):
            model = ConstantPhaseModel(const=model)
        return CompoundPhaseModel(self, model, operator.add)

    def __mul__(self, other):
        if isinstance(other, PhaseModel):
            return CompoundPhaseModel(self, other, operator.mul)
        else:
            raise TypeError(f"Multiplication invalid for type {other!r}")

    def __radd__(self, model):
        return self.__add__(model)

    def __sub__(self, model):
        if not isinstance(model, PhaseModel):
            model = ConstantPhaseModel(const=model)
        return CompoundPhaseModel(self, model, operator.sub)

    def __rsub__(self, model):
        return self.__sub__(model)

    def _propagate_error(self, epsilon, fct, **kwargs):
        """Evaluate error for a given function with uncertainty propagation.

        Parameters
        ----------
        fct : `~astropy.units.Quantity`
            Function to estimate the error.
        epsilon : float
            Step size of the gradient evaluation. Given as a
            fraction of the parameter error.
        **kwargs : dict
            Keyword arguments.

        Returns
        -------
        f_cov : `~astropy.units.Quantity`
            Error of the given function.
        """
        eps = np.sqrt(np.diag(self.covariance)) * epsilon

        n, f_0 = len(self.parameters), fct(**kwargs)
        shape = (n, len(np.atleast_1d(f_0)))
        df_dp = np.zeros(shape)

        for idx, parameter in enumerate(self.parameters):
            if parameter.frozen or eps[idx] == 0:
                continue

            parameter.value += eps[idx]
            df = fct(**kwargs) - f_0

            df_dp[idx] = df.value / eps[idx]
            parameter.value -= eps[idx]

        f_cov = df_dp.T @ self.covariance @ df_dp
        f_err = np.sqrt(np.diagonal(f_cov))
        return u.Quantity([np.atleast_1d(f_0.value), f_err], unit=f_0.unit).squeeze()

    def evaluate_error(self, phase, epsilon=1e-4):
        """Evaluate phase model with error propagation.

        Parameters
        ----------
        phase : `~astropy.units.Quantity`
            Phase at which to evaluate.
        epsilon : float, optional
            Step size of the gradient evaluation. Given as a
            fraction of the parameter error. Default is 1e-4.

        Returns
        -------
        counts, counts_error : tuple of `~astropy.units.Quantity`
            Tuple of counts and counts error.
        """
        return self._propagate_error(epsilon=epsilon, fct=self, energy=phase)

    def integral(self, phase_min, phase_max):
        r"""Integrate phase model numerically if no analytical solution defined.

        .. math::
            F(\phi_{min}, \phi_{max}) = \int_{\phi_{min}}^{\phi_{max}} f(\phi) d\phi

        Parameters
        ----------
        phase_min, phase_max : `~astropy.units.Quantity`
            Lower and upper bound of integration range.

        Returns
        -------
        integral: `~astropy.units.Quantity`.
            Integral bteween phase_min and phase_max
        """
        if hasattr(self, "evaluate_integral"):
            kwargs = {par.name: par.quantity for par in self.parameters}
            return self.evaluate_integral(phase_min, phase_max, **kwargs)
        else:
            NotImplemented(f"Integral is not implemented for {self!r}")

    def integral_error(self, phase_min, phase_max, epsilon=1e-4, **kwargs):
        """Evaluate the error of the integral of a given phase model in a given phase range.

        Parameters
        ----------
        phase_min, phase_max :  `~astropy.units.Quantity`
            Lower and upper bound of integration range.
        epsilon : float, optional
            Step size of the gradient evaluation. Given as a
            fraction of the parameter error. Default is 1e-4.


        Returns
        -------
        integral, integral_err : tuple of `~astropy.units.Quantity`
            Integral and assocaited error between phase_min and phase_max.
        """
        return self._propagate_error(
            epsilon=epsilon,
            fct=self.integral,
            phase_min=phase_min,
            phase_max=phase_max,
            **kwargs,
        )

    def plot(
        self,
        phase_bounds,
        ax=None,
        n_points=100,
        **kwargs,
    ):
        # TODO: Move to periodic boundary in Gammapy 1.3

        if isinstance(phase_bounds, (tuple, list, u.Quantity)):
            phase_min, phase_max = phase_bounds
            phase = MapAxis.from_bounds(
                phase_min,
                phase_max,
                n_points,
                name="phase",
            )
        elif isinstance(phase_bounds, MapAxis):
            phase = phase_bounds

        ax = plt.gca() if ax is None else ax

        counts, _ = self._get_plot(phase=phase)

        ax.plot(phase.center, counts, **kwargs)

        self._plot_format_ax(ax)
        return ax

    def plot_error(
        self,
        phase_bounds,
        ax=None,
        n_points=100,
        **kwargs,
    ):
        # TODO: Move to periodic boundary in Gammapy 1.3

        if isinstance(phase_bounds, (tuple, list, u.Quantity)):
            phase_min, phase_max = phase_bounds
            phase = MapAxis.from_bounds(
                phase_min,
                phase_max,
                n_points,
                name="phase",
            )
        elif isinstance(phase_bounds, MapAxis):
            phase = phase_bounds

        ax = plt.gca() if ax is None else ax

        kwargs.setdefault("facecolor", "black")
        kwargs.setdefault("alpha", 0.2)
        kwargs.setdefault("linewidth", 0)

        counts, error = self._get_plot(phase=phase)
        y_lo = counts - error
        y_hi = counts / error

        ax.fill_between(phase.center, y_lo, y_hi, **kwargs)

        self._plot_format_ax(ax)
        return ax

    @staticmethod
    def _plot_format_ax(ax):
        ax.set_xlabel("Phase")
        ax.set_ylabel("Counts")


class ConstantPhaseModel(PhaseModel):
    tag = ["ConstantPhaseModel", "const"]
    const = Parameter("const", "1")

    @staticmethod
    def evaluate(phase, const):
        """Evaluate the model (static function)."""
        return np.ones(np.atleast_1d(phase).shape) * const


class CompoundPhaseModel(PhaseModel):
    pass
