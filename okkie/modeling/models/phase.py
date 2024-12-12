import logging
import operator

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import scipy
from gammapy.maps import MapAxis
from gammapy.modeling import Parameter
from gammapy.modeling.models import ModelBase

from okkie.utils.models import gammapy_build_parameters_from_dict

log = logging.getLogger(__name__)


__all__ = [
    "PhaseModel",
    "ConstantPhaseModel",
    "CompoundPhaseModel",
    "LorentzianPhaseModel",
    "AsymmetricLorentzianPhaseModel",
    "GaussianPhaseModel",
    "AsymmetricGaussianPhaseModel",
]


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

    def _get_plot(self, phase):
        # TODO: handle case with several period"
        return self.evaluate_error(phase.center)


class ConstantPhaseModel(PhaseModel):
    tag = ["ConstantPhaseModel", "const"]
    const = Parameter("const", 1, interp="lin", scale_method="factor1")

    @staticmethod
    def evaluate(phase, const):
        """Evaluate the model (static function)."""
        return np.ones(np.atleast_1d(phase).shape) * const

    def integral(self, phase_min, phase_max):
        return self.const.value * (phase_max - phase_min)


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
    def evaluate(phase, amplitude, mean, sigma):
        """Evaluate the model"""
        phase = phase % 1
        return amplitude / (1 + np.power((phase - mean) / (sigma / 2), 2))

    def to_pdf(self):
        """Return a pdf version of the model."""
        norm_amp = 2 / (np.pi * self.sigma.value)
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
        amplitude = self.amplitude.value
        mean = self.mean.value
        sigma = self.sigma.value
        return amplitude * sigma(
            np.arctan((phase_max - mean) / sigma)
            - np.arctan((phase_min - mean) / sigma)
        )


class AsymmetricLorentzianPhaseModel(PhaseModel):
    """Asymmetric Lorentzian phase model."""

    tag = ["AsymmetricLorentzianPhaseModel", "asymlor"]
    amplitude = Parameter("amplitude", 1, interp="lin", scale_method="factor1")
    mean = Parameter("mean", 0.5)
    sigma_1 = Parameter("sigma_1", 0.1)
    sigma_2 = Parameter("sigma_2", 0.1)

    @staticmethod
    def evaluate(phase, mean, amplitude, sigma_1, sigma_2):
        """Evaluate the model"""
        phase = phase % 1
        l1 = 1 / (1 + ((phase - mean) / sigma_1) ** 2)
        l2 = 1 / (1 + ((phase - mean) / sigma_2) ** 2)

        return amplitude * np.where(phase < mean, l1, l2)

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

        def single_side_integral(p_min, p_max, sigma):
            return sigma * (
                np.arctan((p_max - mean) / sigma) - np.arctan((p_min - mean) / sigma)
            )

        if phase_max <= mean:
            # Entirely on the left side of the mean
            return amplitude * single_side_integral(phase_min, phase_max, sigma_1)
        elif phase_min >= mean:
            # Entirely on the right side of the mean
            return amplitude * single_side_integral(phase_min, phase_max, sigma_2)
        else:
            # Split integral at the mean
            left_integral = single_side_integral(phase_min, mean, sigma_1)
            right_integral = single_side_integral(mean, phase_max, sigma_2)
            return amplitude * (left_integral + right_integral)


class GaussianPhaseModel(PhaseModel):
    """Gaussian phase model."""

    tag = ["GaussianPhaseModel", "gauss"]
    amplitude = Parameter("amplitude", 1, interp="lin", scale_method="factor1")
    mean = Parameter("mean", 0.5)
    sigma = Parameter("sigma", 0.1)

    @staticmethod
    def evaluate(phase, amplitude, mean, sigma):
        phase = phase % 1
        return amplitude * np.exp(-((phase - mean) ** 2) / (2 * sigma**2))

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
        mean = self.mean.value
        sigma = self.sigma.value
        phase_min = (phase_min - mean) / (np.sqrt(2) * sigma)
        phase_max = (phase_max - mean) / (np.sqrt(2) * sigma)
        amplitude = self.amplitude.value * sigma * np.sqrt(np.pi * 2)
        return (
            amplitude
            / 2
            * (scipy.special.erf(phase_max) - scipy.special.erf(phase_min))
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
    def evaluate(phase, amplitude, mean, sigma_1, sigma_2):
        phase = phase % 1
        g1 = np.exp(-((phase - mean) ** 2) / (2 * sigma_2**2))
        g2 = np.exp(-((phase - mean) ** 2) / (2 * sigma_2**2))

        return amplitude * np.where(phase < mean, g1, g2)

    def to_pdf(self):
        """Return a pdf version of the model."""
        norm_amp = np.sqrt(2 / np.pi) * 1 / (self.sigma_1.value + self.sigma_2.value)
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

        def single_side_integral(p_min, p_max, sigma):
            p_min = (p_min - mean) / (np.sqrt(2) * sigma)
            p_max = (p_max - mean) / (np.sqrt(2) * sigma)
            return 0.5 * (scipy.special.erf(p_max) - scipy.special.erf(p_min))

        if phase_max <= mean:
            # Entirely on the left side of the mean
            return amplitude * single_side_integral(phase_min, phase_max, sigma_1)
        elif phase_min >= mean:
            # Entirely on the right side of the mean
            return amplitude * single_side_integral(phase_min, phase_max, sigma_2)
        else:
            # Split integral at the mean
            left_integral = single_side_integral(phase_min, mean, sigma_1)
            right_integral = single_side_integral(mean, phase_max, sigma_2)
            return amplitude * (left_integral + right_integral)
