from __future__ import annotations

import logging
import operator
from typing import TYPE_CHECKING

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from gammapy.maps import MapAxis
from gammapy.modeling import Parameter, Parameters
from gammapy.modeling.models import ModelBase
from gammapy.utils.interpolation import ScaledRegularGridInterpolator

from okkie.utils.models import gammapy_build_parameters_from_dict

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any

from .decorators import VectorizeEvaluate
from .integral import (
    integrate_periodic_asymm_gaussian,
    integrate_periodic_asymm_lorentzian,
    integrate_periodic_gaussian,
    integrate_periodic_lorentzian,
    integrate_trapezoid,
)

log = logging.getLogger(__name__)


__all__ = [
    "AsymmetricGaussianNormPhaseModel",
    "AsymmetricGaussianPhaseModel",
    "AsymmetricLorentzianNormPhaseModel",
    "AsymmetricLorentzianPhaseModel",
    "CompoundPhaseModel",
    "ConstantPhaseModel",
    "GatePhaseModel",
    "GaussianNormPhaseModel",
    "GaussianPhaseModel",
    "LorentzianNormPhaseModel",
    "LorentzianPhaseModel",
    "PhaseModel",
    "ScalePhaseModel",
    "TemplatePhaseModel",
]

DEFAULT_WRAPPING_TRUNCTAION = 5


class PhaseModel(ModelBase):
    """Phase model base class."""

    _type = "phase"
    period = 1
    wrapping_truncation = DEFAULT_WRAPPING_TRUNCTAION

    def __call__(self, phase: u.Quantity) -> u.Quantity:
        kwargs = {par.name: par.quantity.value for par in self.parameters}
        kwargs["period"] = self.period
        kwargs["wrapping_truncation"] = self.wrapping_truncation
        return self.evaluate(phase, **kwargs)

    def __add__(self, model: PhaseModel | float) -> CompoundPhaseModel:
        if not isinstance(model, PhaseModel):
            model = ConstantPhaseModel(const=model)
        return CompoundPhaseModel(self, model, operator.add)

    def __mul__(self, other: PhaseModel) -> CompoundPhaseModel:
        if isinstance(other, PhaseModel):
            return CompoundPhaseModel(self, other, operator.mul)
        raise TypeError(f"Multiplication invalid for type {other!r}")

    def __radd__(self, model: PhaseModel | float) -> CompoundPhaseModel:
        return self.__add__(model)

    def __sub__(self, model: PhaseModel | float) -> CompoundPhaseModel:
        if not isinstance(model, PhaseModel):
            model = ConstantPhaseModel(const=model)
        return CompoundPhaseModel(self, model, operator.sub)

    def __rsub__(self, model: PhaseModel | float) -> CompoundPhaseModel:
        return self.__sub__(model)

    @classmethod
    def from_dict(cls, data: dict[str, Any], **kwargs: Any) -> PhaseModel:
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

    def _propagate_error(
        self,
        epsilon: float,
        fct: Callable[..., u.Quantity],
        **kwargs: Any,
    ) -> u.Quantity:
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

    def evaluate_error(
        self, phase: u.Quantity, epsilon: float = 1e-4
    ) -> tuple[u.Quantity, u.Quantity]:
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

    def integral(self, phase_min: u.Quantity, phase_max: u.Quantity) -> u.Quantity:
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

    def integral_error(
        self,
        phase_min: u.Quantity,
        phase_max: u.Quantity,
        epsilon: float = 1e-4,
        **kwargs: Any,
    ) -> tuple[u.Quantity, u.Quantity]:
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
        phase_bounds: MapAxis | Sequence[float] | u.Quantity,
        ax: plt.Axes | None = None,
        n_points: int = 100,
        **kwargs: Any,
    ) -> plt.Axes:
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
        phase_bounds: MapAxis | Sequence[float] | u.Quantity,
        ax: plt.Axes | None = None,
        n_points: int = 100,
        **kwargs: Any,
    ) -> plt.Axes:
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
    def _plot_format_ax(ax: plt.Axes) -> None:
        ax.set_xlabel("Phase")
        ax.set_ylabel("Counts")

    def _get_plot(self, phase: MapAxis) -> tuple[u.Quantity, u.Quantity]:
        # TODO: handle case with several period"
        return self.evaluate_error(phase.center)


class ConstantPhaseModel(PhaseModel):
    tag = ["ConstantPhaseModel", "const"]
    const = Parameter("const", 1, interp="lin", scale_method="factor1")

    @staticmethod
    @VectorizeEvaluate(params_to_broadcast=("const",), provide_shifts=False)
    def evaluate(phase, const, period, wrapping_truncation, *, _vmeta=None):
        Pnd = _vmeta["phase_ndim"]
        Qshp = _vmeta["param_shape"]
        target = phase.shape[:-1] + Qshp
        return np.broadcast_to(const, target)

    def integral(self, phase_min: float, phase_max: float) -> float:
        phase_min %= self.period
        if phase_max != self.period:
            phase_max %= self.period
        return self.const.value * abs(phase_max - phase_min)


class CompoundPhaseModel(PhaseModel):
    tag = ["CompoundPhaseModel", "compound"]

    def __init__(
        self,
        model1: PhaseModel,
        model2: PhaseModel,
        operator: Callable[[u.Quantity, u.Quantity], u.Quantity],
    ) -> None:
        self.model1 = model1
        self.model2 = model2
        self.operator = operator
        super().__init__()

    @property
    def _models(self) -> list[PhaseModel]:
        return [self.model1, self.model2]

    @property
    def parameters(self) -> Parameters:
        return self.model1.parameters + self.model2.parameters

    @property
    def parameters_unique_names(self) -> list[str]:
        names: list[str] = []
        for idx, model in enumerate(self._models):
            for par_name in model.parameters_unique_names:
                components = [f"model{idx + 1}", par_name]
                name = ".".join(components)
                names.append(name)
        return names

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}\n"
            f"    Component 1 : {self.model1}\n"
            f"    Component 2 : {self.model2}\n"
            f"    Operator : {self.operator.__name__}\n"
        )

    def __call__(self, phase: u.Quantity) -> u.Quantity:
        val1 = self.model1(phase)
        val2 = self.model2(phase)
        return self.operator(val1, val2)

    def evaluate(self, phase: float, *args: float) -> float:
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
    @VectorizeEvaluate(
        params_to_broadcast=("amplitude", "mean", "sigma"),
        mod_by_period=("mean",),
        clips={"sigma": (1e-300, None)},
        provide_shifts=True,
    )
    def evaluate(
        phase,
        amplitude,
        mean,
        sigma,
        period,
        wrapping_truncation,
        *,
        shifts=None,
        _vmeta=None,
    ):
        delta = (phase - mean)[..., None] + shifts
        num = (1.0 / (1.0 + (delta / sigma[..., None]) ** 2)).sum(axis=-1)
        return amplitude * num

    def to_norm(self) -> LorentzianNormPhaseModel:
        """Return the normalized version of the model."""
        return LorentzianNormPhaseModel(mean=self.mean, sigma=self.sigma)

    def integral(self, phase_min: float, phase_max: float) -> float:
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
    @VectorizeEvaluate(
        params_to_broadcast=("amplitude", "mean", "sigma_1", "sigma_2"),
        mod_by_period=("mean",),
        clips={"sigma_1": (1e-300, None), "sigma_2": (1e-300, None)},
        provide_shifts=True,
    )
    def evaluate(
        phase,
        amplitude,
        mean,
        sigma_1,
        sigma_2,
        period,
        wrapping_truncation,
        *,
        shifts=None,
        _vmeta=None,
    ):
        delta = (phase - mean)[..., None] + shifts
        sig = np.where(delta < 0.0, sigma_1[..., None], sigma_2[..., None])
        num = (1.0 / (1.0 + (delta / sig) ** 2)).sum(axis=-1)
        return amplitude * num

    def to_norm(self) -> AsymmetricLorentzianNormPhaseModel:
        """Return the normalized version of the model."""
        return AsymmetricLorentzianNormPhaseModel(
            mean=self.mean, sigma_1=self.sigma_1, sigma_2=self.sigma_2
        )

    def integral(self, phase_min: float, phase_max: float) -> float:
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
    @VectorizeEvaluate(
        params_to_broadcast=("amplitude", "mean", "sigma"),
        mod_by_period=("mean",),
        clips={"sigma": (1e-300, None)},
        provide_shifts=True,
    )
    def evaluate(
        phase,
        amplitude,
        mean,
        sigma,
        period,
        wrapping_truncation,
        *,
        shifts=None,
        _vmeta=None,
    ):
        delta = (phase - mean)[..., None] + shifts
        num = np.exp(-0.5 * (delta / sigma[..., None]) ** 2).sum(axis=-1)
        return amplitude * num

    def to_norm(self) -> GaussianNormPhaseModel:
        """Return the normalized version of the model."""
        return GaussianNormPhaseModel(mean=self.mean, sigma=self.sigma)

    def integral(self, phase_min: float, phase_max: float) -> float:
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
    @VectorizeEvaluate(
        params_to_broadcast=("amplitude", "mean", "sigma_1", "sigma_2"),
        mod_by_period=("mean",),
        clips={"sigma_1": (1e-300, None), "sigma_2": (1e-300, None)},
        provide_shifts=True,
    )
    def evaluate(
        phase,
        amplitude,
        mean,
        sigma_1,
        sigma_2,
        period,
        wrapping_truncation,
        *,
        shifts=None,
        _vmeta=None,
    ):
        delta = (phase - mean)[..., None] + shifts
        sig = np.where(delta < 0.0, sigma_1[..., None], sigma_2[..., None])
        num = np.exp(-0.5 * (delta / sig) ** 2).sum(axis=-1)
        return amplitude * num

    def to_norm(self) -> AsymmetricGaussianNormPhaseModel:
        """Return the normalized version of the model."""
        return AsymmetricGaussianNormPhaseModel(
            mean=self.mean, sigma_1=self.sigma_1, sigma_2=self.sigma_2
        )

    def integral(self, phase_min: float, phase_max: float) -> float:
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

    def __init__(
        self,
        phase: np.ndarray,
        values: np.ndarray,
        phase_shift: float = 0,
        interp_kwargs: dict[str, Any] | None = None,
    ) -> None:
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

    def evaluate(
        self,
        phase: u.Quantity,
        phase_shift: u.Quantity,
        period: float,
        wrapping_truncation: int,
    ) -> u.Quantity:
        shifted_phase = (phase + phase_shift) % period
        return self._evaluate((shifted_phase,), clip=True)

    def to_norm(
        self, interp_kwargs: dict[str, Any] | None = None
    ) -> TemplatePhaseModel:
        """Return the normalized version of the model."""
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
    scale : float
        Multiplicative norm factor for the model value.
        Default is 1.
    """

    tag = ["ScalePhaseModel", "scale"]
    scale = Parameter("scale", 1)

    def __init__(self, model: PhaseModel, scale: float = scale.quantity.value) -> None:
        self.model = model
        self._covariance = None
        super().__init__(scale=scale)

    @property
    def parameters(self) -> Parameters:
        return Parameters([self.scale] + list(self.model.parameters))

    def evaluate(
        self,
        phase,
        scale,
        period: float,
        wrapping_truncation: int,
        **kwargs,
    ) -> np.array:
        """
        Outer-vectorized wrapper:
          - phase : array-like, shape P...
          - scale : scalar or array-like, part of param grid Q...
          - kwargs: inner model parameters (scalar/arrays), together define Q...
          -> returns shape P... × Q...
        """
        phase = np.asarray(phase, float)
        Pshp = phase.shape
        Pnd = phase.ndim

        to_broadcast = [np.asarray(scale, float)]
        kw_names = list(kwargs.keys())
        for name in kw_names:
            to_broadcast.append(np.asarray(kwargs[name], float))

        if to_broadcast:
            bcast = np.broadcast_arrays(*to_broadcast)
            scale_b = bcast[0]
            j = 1
            for name in kw_names:
                kwargs[name] = bcast[j]
                j += 1
            Qshp = scale_b.shape
        else:
            scale_b = np.asarray(scale, float)
            Qshp = ()

        scale_rs = scale_b.reshape((1,) * Pnd + Qshp)

        vals = self.model.evaluate(
            phase=phase,
            period=period,
            wrapping_truncation=wrapping_truncation,
            **kwargs,
        )
        return scale_rs * vals

    def integral(
        self,
        phase_min: float,
        phase_max: float,
        **kwargs: Any,
    ) -> float:
        return self.scale.value * self.model.integral(phase_min, phase_max, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """
        If an attribute isn't found on this wrapper, try to fetch a Parameter
        (or attribute) from the wrapped model. This allows `scale.mean.value = ...`.
        """
        # Forward to parameters by name
        params = getattr(self.model, "parameters", None)
        if params is not None and name in params.names:
            return params[name]
        # Or forward any other attribute transparently
        if hasattr(self.model, name):
            return getattr(self.model, name)
        raise AttributeError(f"{self.__class__.__name__!s} has no attribute {name!r}")

    def __dir__(self) -> list[str]:
        """Improve tab-completion: include wrapped parameter names."""
        base = set(super().__dir__())
        try:
            wrapped = set(self.model.parameters.names)
        except Exception:
            wrapped = set()
        return sorted(base | wrapped)


class GatePhaseModel(PhaseModel):
    tag = ["GatePhaseModel", "gate"]
    amplitude = Parameter("amplitude", 1)
    phi_1 = Parameter("phi_1", "0.25")
    phi_2 = Parameter("phi_2", "0.75")

    def evaluate(
        self,
        phase: u.Quantity,
        amplitude: u.Quantity,
        phi_1: u.Quantity,
        phi_2: u.Quantity,
        period: float,
        wrapping_truncation: int,
    ) -> u.Quantity:
        return np.where((phase > phi_1) & (phase < phi_2), amplitude, 0)

    def integral(
        self, phase_min: u.Quantity, phase_max: u.Quantity, **kwargs: Any
    ) -> u.Quantity:
        return self.amplitude.value * (self.phi_2.value - self.phi_1.value)


class GaussianNormPhaseModel(PhaseModel):
    """Gaussian wrapped phase model that is always a PDF (∫_0^P = 1)."""

    tag = ["GaussianNormPhaseModel", "gaussnorm"]
    mean = Parameter("mean", 0.5)
    sigma = Parameter("sigma", 0.1)

    @staticmethod
    @VectorizeEvaluate(
        params_to_broadcast=("mean", "sigma"),
        mod_by_period=("mean",),
        clips={"sigma": (1e-300, None)},
        provide_shifts=True,
    )
    def evaluate(
        phase, mean, sigma, period, wrapping_truncation, *, shifts=None, _vmeta=None
    ):
        P = float(period)
        K = int(wrapping_truncation)
        Pnd = _vmeta["phase_ndim"]
        Qshp = _vmeta["param_shape"]

        delta = (phase - mean)[..., None] + shifts
        num = np.exp(-0.5 * (delta / sigma[..., None]) ** 2).sum(axis=-1)

        # Denominator in Q-shape
        axes = tuple(range(Pnd))
        mean_q = np.squeeze(mean, axis=axes)
        sigma_q = np.squeeze(sigma, axis=axes)
        den_q = integrate_periodic_gaussian(
            0.0, P, amplitude=1.0, mean=mean_q, sigma=sigma_q, period=P, truncation=K
        )  # Q...
        den = np.clip(den_q, 1e-300, np.inf).reshape((1,) * Pnd + Qshp)
        return num / den

    def integral(self, phase_min: float, phase_max: float, **kwargs: Any) -> float:
        full = integrate_periodic_gaussian(
            edge_min=0.0,
            edge_max=self.period,
            amplitude=1.0,
            mean=self.mean.value,
            sigma=self.sigma.value,
            period=self.period,
            truncation=self.wrapping_truncation,
        )
        part = integrate_periodic_gaussian(
            edge_min=phase_min,
            edge_max=phase_max,
            amplitude=1.0,
            mean=self.mean.value,
            sigma=self.sigma.value,
            period=self.period,
            truncation=self.wrapping_truncation,
        )
        full = max(float(full), 1e-300)
        return float(part) / full


class AsymmetricGaussianNormPhaseModel(PhaseModel):
    """Asymmetric Gaussian wrapped phase model that is always a PDF (∫_0^P = 1)."""

    tag = ["AsymmetricGaussianNormPhaseModel", "asymgaussnorm"]
    mean = Parameter("mean", 0.5)
    sigma_1 = Parameter("sigma_1", 0.1)
    sigma_2 = Parameter("sigma_2", 0.1)

    @staticmethod
    @VectorizeEvaluate(
        params_to_broadcast=("mean", "sigma_1", "sigma_2"),
        mod_by_period=("mean",),
        clips={"sigma_1": (1e-300, None), "sigma_2": (1e-300, None)},
        provide_shifts=True,
    )
    def evaluate(
        phase,
        mean,
        sigma_1,
        sigma_2,
        period,
        wrapping_truncation,
        *,
        shifts=None,
        _vmeta=None,
    ):
        P = float(period)
        K = int(wrapping_truncation)
        Pnd = _vmeta["phase_ndim"]
        Qshp = _vmeta["param_shape"]

        delta = (phase - mean)[..., None] + shifts
        sig = np.where(delta < 0.0, sigma_1[..., None], sigma_2[..., None])
        num = np.exp(-0.5 * (delta / sig) ** 2).sum(axis=-1)

        axes = tuple(range(Pnd))
        mean_q = np.squeeze(mean, axis=axes)
        s1_q = np.squeeze(sigma_1, axis=axes)
        s2_q = np.squeeze(sigma_2, axis=axes)
        den_q = integrate_periodic_asymm_gaussian(
            0.0,
            P,
            amplitude=1.0,
            mean=mean_q,
            sigma_1=s1_q,
            sigma_2=s2_q,
            period=P,
            truncation=K,
        )
        den = np.clip(den_q, 1e-300, np.inf).reshape((1,) * Pnd + Qshp)
        return num / den

    def integral(self, phase_min: float, phase_max: float, **kwargs: Any) -> float:
        full = integrate_periodic_asymm_gaussian(
            edge_min=0.0,
            edge_max=self.period,
            amplitude=1.0,
            mean=self.mean.value,
            sigma_1=self.sigma_1.value,
            sigma_2=self.sigma_2.value,
            period=self.period,
            truncation=self.wrapping_truncation,
        )
        part = integrate_periodic_asymm_gaussian(
            edge_min=phase_min,
            edge_max=phase_max,
            amplitude=1.0,
            mean=self.mean.value,
            sigma_1=self.sigma_1.value,
            sigma_2=self.sigma_2.value,
            period=self.period,
            truncation=self.wrapping_truncation,
        )
        full = max(float(full), 1e-300)
        return float(part) / full


class LorentzianNormPhaseModel(PhaseModel):
    """Lorentzian wrapped phase model that is always a PDF (∫_0^P = 1)."""

    tag = ["LorentzianNormPhaseModel", "lornorm"]
    mean = Parameter("mean", 0.0)
    sigma = Parameter("sigma", 0.1)

    @staticmethod
    @VectorizeEvaluate(
        params_to_broadcast=("mean", "sigma"),
        mod_by_period=("mean",),
        clips={"sigma": (1e-300, None)},
        provide_shifts=True,
    )
    def evaluate(
        phase, mean, sigma, period, wrapping_truncation, *, shifts=None, _vmeta=None
    ):
        P = float(period)
        K = int(wrapping_truncation)
        Pnd = _vmeta["phase_ndim"]
        Qshp = _vmeta["param_shape"]

        delta = (phase - mean)[..., None] + shifts
        num = (1.0 / (1.0 + (delta / sigma[..., None]) ** 2)).sum(axis=-1)

        axes = tuple(range(Pnd))
        mean_q = np.squeeze(mean, axis=axes)
        sigma_q = np.squeeze(sigma, axis=axes)
        den_q = integrate_periodic_lorentzian(
            0.0, P, amplitude=1.0, mean=mean_q, sigma=sigma_q, period=P, truncation=K
        )
        den = np.clip(den_q, 1e-300, np.inf).reshape((1,) * Pnd + Qshp)
        return num / den

    def integral(self, phase_min: float, phase_max: float, **kwargs: Any) -> float:
        full = integrate_periodic_lorentzian(
            edge_min=0.0,
            edge_max=self.period,
            amplitude=1.0,
            mean=self.mean.value,
            sigma=self.sigma.value,
            period=self.period,
            truncation=self.wrapping_truncation,
        )
        part = integrate_periodic_lorentzian(
            edge_min=phase_min,
            edge_max=phase_max,
            amplitude=1.0,
            mean=self.mean.value,
            sigma=self.sigma.value,
            period=self.period,
            truncation=self.wrapping_truncation,
        )
        full = max(float(full), 1e-300)
        return float(part) / full


class AsymmetricLorentzianNormPhaseModel(PhaseModel):
    """Asymmetric Lorentzian wrapped phase model that is always a PDF (∫_0^P = 1)."""

    tag = ["AsymmetricLorentzianNormPhaseModel", "asymlornorm"]
    mean = Parameter("mean", 0.0)
    sigma_1 = Parameter("sigma_1", 0.1)
    sigma_2 = Parameter("sigma_2", 0.1)

    @staticmethod
    @VectorizeEvaluate(
        params_to_broadcast=("mean", "sigma_1", "sigma_2"),
        mod_by_period=("mean",),
        clips={"sigma_1": (1e-300, None), "sigma_2": (1e-300, None)},
        provide_shifts=True,
    )
    def evaluate(
        phase,
        mean,
        sigma_1,
        sigma_2,
        period,
        wrapping_truncation,
        *,
        shifts=None,
        _vmeta=None,
    ):
        P = float(period)
        K = int(wrapping_truncation)
        Pnd = _vmeta["phase_ndim"]
        Qshp = _vmeta["param_shape"]

        delta = (phase - mean)[..., None] + shifts
        sig = np.where(delta < 0.0, sigma_1[..., None], sigma_2[..., None])
        num = (1.0 / (1.0 + (delta / sig) ** 2)).sum(axis=-1)

        axes = tuple(range(Pnd))
        mean_q = np.squeeze(mean, axis=axes)
        s1_q = np.squeeze(sigma_1, axis=axes)
        s2_q = np.squeeze(sigma_2, axis=axes)
        den_q = integrate_periodic_asymm_lorentzian(
            0.0,
            P,
            amplitude=1.0,
            mean=mean_q,
            sigma_1=s1_q,
            sigma_2=s2_q,
            period=P,
            truncation=K,
        )
        den = np.clip(den_q, 1e-300, np.inf).reshape((1,) * Pnd + Qshp)
        return num / den

    def integral(self, phase_min: float, phase_max: float, **kwargs: Any) -> float:
        full = integrate_periodic_asymm_lorentzian(
            edge_min=0.0,
            edge_max=self.period,
            amplitude=1.0,
            mean=self.mean.value,
            sigma_1=self.sigma_1.value,
            sigma_2=self.sigma_2.value,
            period=self.period,
            truncation=self.wrapping_truncation,
        )
        part = integrate_periodic_asymm_lorentzian(
            edge_min=phase_min,
            edge_max=phase_max,
            amplitude=1.0,
            mean=self.mean.value,
            sigma_1=self.sigma_1.value,
            sigma_2=self.sigma_2.value,
            period=self.period,
            truncation=self.wrapping_truncation,
        )
        full = max(float(full), 1e-300)
        return float(part) / full
