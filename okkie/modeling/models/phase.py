import logging
import operator

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from gammapy.maps import MapAxis
from gammapy.modeling import Parameter
from gammapy.modeling.models import ModelBase
from gammapy.utils.interpolation import ScaledRegularGridInterpolator

from okkie.utils.models import gammapy_build_parameters_from_dict

from .integral import (
    integrate_periodic_asymm_gaussian,
    integrate_periodic_asymm_lorentzian,
    integrate_periodic_gaussian,
    integrate_periodic_lorentzian,
    integrate_trapezoid,
)

log = logging.getLogger(__name__)


__all__ = [
    "PhaseModel",
    "ConstantPhaseModel",
    "CompoundPhaseModel",
    "LorentzianPhaseModel",
    "AsymmetricLorentzianPhaseModel",
    "GaussianPhaseModel",
    "AsymmetricGaussianPhaseModel",
    "TemplatePhaseModel",
    "ScalePhaseModel",
    "GatePhaseModel",
]

DEFAULT_WRAPPING_TRUNCTAION = 5


class PhaseModel(ModelBase):
    """Phase model base class."""

    _type = "phase"
    period = 1
    wrapping_truncation = DEFAULT_WRAPPING_TRUNCTAION

    def __call__(self, phase):
        kwargs = {par.name: par.quantity for par in self.parameters}
        kwargs["period"] = self.period
        kwargs["wrapping_truncation"] = self.wrapping_truncation
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

    @classmethod
    def from_dict(cls, data, **kwargs):
        key0 = next(iter(data))

        if key0 == "phase":
            data = data[key0]

        if data["type"] not in cls.tag:
            raise ValueError(
                f"Invalid model type {data['type']} for class {cls.__name__}"
            )

        parameters = gammapy_build_parameters_from_dict(
            data["parameters"], cls.default_parameters
        )

        return cls.from_parameters(parameters, **kwargs)

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
        return self._propagate_error(epsilon=epsilon, fct=self, phase=phase)

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
            return integrate_trapezoid(self, phase_min, phase_max)

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
            Integral and associated error between phase_min and phase_max.
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

    def _get_plot(self, phase):
        # TODO: handle case with several period"
        return self.evaluate_error(phase.center)


class ConstantPhaseModel(PhaseModel):
    tag = ["ConstantPhaseModel", "const"]
    const = Parameter("const", 1, interp="lin", scale_method="factor1")

    @staticmethod
    def evaluate(phase, const, period, wrapping_truncation):
        """Evaluate the model (static function)."""
        return np.ones(np.atleast_1d(phase).shape) * const

    def integral(self, phase_min, phase_max):
        phase_min %= self.period
        if phase_max != self.period:
            phase_max %= self.period
        return self.const.value * abs(phase_max - phase_min)


class CompoundPhaseModel(PhaseModel):
    tag = ["CompoundPhaseModel", "compound"]

    def __init__(self, model1, model2, operator):
        self.model1 = model1
        self.model2 = model2
        self.operator = operator
        super().__init__()

    @property
    def _models(self):
        return [self.model1, self.model2]

    @property
    def parameters(self):
        return self.model1.parameters + self.model2.parameters

    @property
    def parameters_unique_names(self):
        names = []
        for idx, model in enumerate(self._models):
            for par_name in model.parameters_unique_names:
                components = [f"model{idx+1}", par_name]
                name = ".".join(components)
                names.append(name)
        return names

    def __str__(self):
        return (
            f"{self.__class__.__name__}\n"
            f"    Component 1 : {self.model1}\n"
            f"    Component 2 : {self.model2}\n"
            f"    Operator : {self.operator.__name__}\n"
        )

    def __call__(self, phase):
        val1 = self.model1(phase)
        val2 = self.model2(phase)
        return self.operator(val1, val2)

    def evaluate(self, phase, *args):
        args1 = args[: len(self.model1.parameters)]
        args2 = args[len(self.model1.parameters) :]
        val1 = self.model1.evaluate(phase, *args1)
        val2 = self.model2.evaluate(phase, *args2)
        return self.operator(val1, val2)


class LorentzianPhaseModel(PhaseModel):
    """Lorentzian phase model."""

    tag = ["LorentzianphaseModel", "lor"]
    amplitude = Parameter("amplitude", 1, interp="lin", scale_method="factor1")
    mean = Parameter("mean", 0)
    sigma = Parameter("sigma", 0.1)

    @staticmethod
    def evaluate(phase, amplitude, mean, sigma, period, wrapping_truncation):
        mean = mean % period
        mean = mean.reshape((1,))  # Trick to pass in float or int
        phase = phase % period
        delta_phase = phase - mean
        periodic_shifts = (
            np.arange(-wrapping_truncation, wrapping_truncation + 1) * period
        )
        delta_phase_wrapped = delta_phase[:, np.newaxis] + periodic_shifts

        lorentzian = 1 / (1 + (delta_phase_wrapped / sigma) ** 2)

        normalization = 0.0
        for shift in periodic_shifts:
            normalization += 1 / (1 + (shift / sigma) ** 2)

        values = amplitude * lorentzian.sum(axis=1) / normalization

        return values

    def to_pdf(self):
        """Return a pdf version of the model."""
        norm_amp = 1 / (np.pi * self.sigma.value)
        return self.__class__(amplitude=norm_amp, mean=self.mean, sigma=self.sigma)

    def integral(self, phase_min, phase_max):
        """Integral between `phase_min` and `phase_max`.

        Parameters
        ----------
        phase_min, phase_max: float
            Edges of the integration.

        Returns
        -------
        integral: float
            Value of the integral.
        """
        return integrate_periodic_lorentzian(
            edge_min=phase_min,
            edge_max=phase_max,
            amplitude=self.amplitude.value,
            mean=self.mean.value,
            sigma=self.sigma.value,
            period=self.period,
            truncation=self.wrapping_truncation,
        )


class AsymmetricLorentzianPhaseModel(PhaseModel):
    """Asymmetric Lorentzian phase model."""

    tag = ["AsymmetricLorentzianPhaseModel", "asymlor"]
    amplitude = Parameter("amplitude", 1, interp="lin", scale_method="factor1")
    mean = Parameter("mean", 0.5)
    sigma_1 = Parameter("sigma_1", 0.1)
    sigma_2 = Parameter("sigma_2", 0.1)

    @staticmethod
    def evaluate(phase, amplitude, mean, sigma_1, sigma_2, period, wrapping_truncation):
        mean = mean % period
        mean = mean.reshape((1,))  # Trick to pass in float or int
        phase = phase % period
        delta_phase = phase - mean
        periodic_shifts = (
            np.arange(-wrapping_truncation, wrapping_truncation + 1) * period
        )
        delta_phase_wrapped = delta_phase[:, np.newaxis] + periodic_shifts

        l1 = 1 / (1 + (delta_phase_wrapped / sigma_1) ** 2)
        l2 = 1 / (1 + (delta_phase_wrapped / sigma_2) ** 2)
        lorentzian = np.where(delta_phase_wrapped < 0, l1, l2)

        normalization = 0.0
        for shift in periodic_shifts:
            norm_l1 = 1 / (1 + (shift / sigma_1) ** 2)
            norm_l2 = 1 / (1 + (shift / sigma_2) ** 2)
            normalization += (norm_l1 + norm_l2) / 2

        values = amplitude * lorentzian.sum(axis=1) / normalization

        return values

    def to_pdf(self):
        """Return a pdf version of the model."""
        norm_amp = 2 / (np.pi * (self.sigma_1.value + self.sigma_2.value))
        return self.__class__(
            amplitude=norm_amp,
            mean=self.mean,
            sigma_1=self.sigma_1,
            sigma_2=self.sigma_2,
        )

    def integral(self, phase_min, phase_max):
        """Integral between `phase_min` and `phase_max`.

        Parameters
        ----------
        phase_min, phase_max: float
            Edges of the integration.

        Returns
        -------
        integral: float
            Value of the integral.
        """
        amplitude = self.amplitude.value
        mean = self.mean.value
        sigma_1 = self.sigma_1.value
        sigma_2 = self.sigma_2.value

        return integrate_periodic_asymm_lorentzian(
            edge_min=phase_min,
            edge_max=phase_max,
            amplitude=amplitude,
            mean=mean,
            sigma_1=sigma_1,
            sigma_2=sigma_2,
            period=self.period,
            truncation=self.wrapping_truncation,
        )


class GaussianPhaseModel(PhaseModel):
    """Gaussian phase model."""

    tag = ["GaussianPhaseModel", "gauss"]
    amplitude = Parameter("amplitude", 1, interp="lin", scale_method="factor1")
    mean = Parameter("mean", 0.5)
    sigma = Parameter("sigma", 0.1)

    @staticmethod
    def evaluate(phase, amplitude, mean, sigma, period, wrapping_truncation):
        mean = mean % period
        mean = mean.reshape((1,))  # Trick to pass in float or int
        phase = phase % period
        delta_phase = phase - mean
        periodic_shifts = (
            np.arange(-wrapping_truncation, wrapping_truncation + 1) * period
        )
        delta_phase_wrapped = delta_phase[:, np.newaxis] + periodic_shifts

        gaussians = np.exp(-(delta_phase_wrapped**2) / (2 * sigma**2))

        normalization = sum(
            np.exp(-((shift) ** 2) / (2 * sigma**2)) for shift in periodic_shifts
        )

        return amplitude * gaussians.sum(axis=1) / normalization

    def to_pdf(self):
        """Return a pdf version of the model."""
        norm_amp = 1 / (self.sigma.value * np.sqrt(2 * np.pi))
        return self.__class__(amplitude=norm_amp, sigma=self.sigma, mean=self.mean)

    def integral(self, phase_min, phase_max):
        """Integral between `phase_min` and `phase_max`.

        Parameters
        ----------
        phase_min, phase_max: float
            Edges of the integration.

        Returns
        -------
        integral: float
            Value of the integral.
        """
        return integrate_periodic_gaussian(
            edge_min=phase_min,
            edge_max=phase_max,
            amplitude=self.amplitude.value,
            mean=self.mean.value,
            sigma=self.sigma.value,
            period=self.period,
            truncation=self.wrapping_truncation,
        )


class AsymmetricGaussianPhaseModel(PhaseModel):
    """Asymmetric Gaussian phase model.

    From 3PC Paper Eq. 10.
    """

    tag = ["AsymetricGaussianPhaseModel", "asymgauss"]
    amplitude = Parameter("amplitude", 1, interp="lin", scale_method="factor1")
    mean = Parameter("mean", 0.5)
    sigma_1 = Parameter("sigma_1", 0.1)
    sigma_2 = Parameter("sigma_2", 0.1)

    @staticmethod
    def evaluate(phase, amplitude, mean, sigma_1, sigma_2, period, wrapping_truncation):
        mean = mean % period
        mean = mean.reshape((1,))  # Trick to pass in float or int
        phase = phase % period
        delta_phase = phase - mean
        periodic_shifts = (
            np.arange(-wrapping_truncation, wrapping_truncation + 1) * period
        )
        delta_phase_wrapped = delta_phase[:, np.newaxis] + periodic_shifts

        g1 = np.exp(-(delta_phase_wrapped**2) / (2 * sigma_1**2))
        g2 = np.exp(-(delta_phase_wrapped**2) / (2 * sigma_2**2))
        gaussians = np.where(delta_phase_wrapped < 0, g1, g2)

        normalization = sum(
            0.5
            * (
                np.exp(-((shift) ** 2) / (2 * sigma_1**2))
                + np.exp(-((shift) ** 2) / (2 * sigma_2**2))
            )
            for shift in periodic_shifts
        )

        return amplitude * gaussians.sum(axis=1) / normalization

    def to_pdf(self):
        """Return a pdf version of the model."""
        norm_amp = 2 / (
            (np.sqrt(2 * np.pi)) * (self.sigma_1.value + self.sigma_2.value)
        )
        return self.__class__(
            amplitude=norm_amp,
            mean=self.mean,
            sigma_1=self.sigma_1,
            sigma_2=self.sigma_2,
        )

    def integral(self, phase_min, phase_max):
        """Integral between `phase_min` and `phase_max`.

        Parameters
        ----------
        phase_min, phase_max: float
            Edges of the integration.

        Returns
        -------
        integral: float
            Value of the integral.
        """
        mean = self.mean.value
        sigma_1 = self.sigma_1.value
        sigma_2 = self.sigma_2.value
        amplitude = self.amplitude.value

        return integrate_periodic_asymm_gaussian(
            edge_min=phase_min,
            edge_max=phase_max,
            amplitude=amplitude,
            mean=mean,
            sigma_1=sigma_1,
            sigma_2=sigma_2,
            period=self.period,
            truncation=self.wrapping_truncation,
        )


class TemplatePhaseModel(PhaseModel):
    """A model generated for an array of phase and associated values.

    Parameters
    ----------
    phase : `~numpy.ndarray`
        Array of phases at which the model values are given
    values : `~numpy.ndarray`
        Array with the values of the model at phases ``phase``.
    phase_shift : float
        Shift to apply to the phase to shift the template. Shifted phases wrap at the period.
        Default is 0.
    interp_kwargs : dict
        Interpolation option passed to `~gammapy.utils.interpolation.ScaledRegularGridInterpolator`.
        By default, all values outside the interpolation range are set to NaN.
        If you want to apply linear extrapolation you can pass `interp_kwargs={'extrapolate':
        True, 'method': 'linear'}`. If you want to choose the interpolation
        scaling applied to values, you can use `interp_kwargs={"values_scale": "log"}`.
    """

    tag = ["TemplatePhaseModel", "temp-phase"]
    phase_shift = Parameter("phase_shift", 0)

    def __init__(self, phase, values, phase_shift=0, interp_kwargs=None):
        self.phase = phase
        self.values = values
        interp_kwargs = interp_kwargs or {}
        interp_kwargs.setdefault("values_scale", "lin")
        interp_kwargs.setdefault("points_scale", ("log",))

        if len(phase) == 1:
            interp_kwargs["method"] = "nearest"

        self._evaluate = ScaledRegularGridInterpolator(
            points=(phase,), values=values, **interp_kwargs
        )

        super().__init__()
        self.phase_shift.value = phase_shift

    def evaluate(self, phase, phase_shift, period, wrapping_truncation):
        shifted_phase = (phase + phase_shift) % period
        return self._evaluate((shifted_phase,), clip=True)

    def to_pdf(self, interp_kwargs=None):
        """Return a pdf version of the model."""
        interp_kwargs = interp_kwargs or {}
        norm = 1 / self.integral(0, self.period)
        return self.__class__(
            phase=self.phase,
            values=self.values * norm,
            phase_shift=self.phase_shift.value,
            interp_kwargs=interp_kwargs,
        )


class ScalePhaseModel(PhaseModel):
    """Wrapper to scale another phase model by a norm factor.

    Parameters
    ----------
    model : `PhaseModel`
        Phase model to wrap.
    norm : float
        Multiplicative norm factor for the model value.
        Default is 1.
    """

    tag = ["ScalePhaseModel", "scale"]
    norm = Parameter("norm", 1)

    def __init__(self, model, norm=norm.quantity):
        self.model = model
        self._covariance = None
        super().__init__(norm=norm)

    def evaluate(self, phase, norm, period, wrapping_truncation):
        return norm * self.model(phase)

    def integral(self, phase_min, phase_max, **kwargs):
        return self.norm.value * self.model.integral(phase_min, phase_max, **kwargs)


class GatePhaseModel(PhaseModel):
    tag = ["GatePhaseModel", "gate"]
    amplitude = Parameter("amplitude", 1)
    phi_1 = Parameter("phi_1", "0.25")
    phi_2 = Parameter("phi_2", "0.75")

    def evaluate(self, phase, amplitude, phi_1, phi_2, period, wrapping_truncation):
        return np.where((phase > phi_1) & (phase < phi_2), amplitude, 0)

    def integral(self, phase_min, phase_max, **kwargs):
        return self.amplitude.value * (self.phi_2.value - self.phi_1.value)
