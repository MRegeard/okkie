import itertools
import operator
from typing import Optional
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import quantity_support
from gammapy.maps import RegionNDMap
from gammapy.modeling import Covariance, Parameter, Parameters
from gammapy.modeling.models import ModelBase
from gammapy.utils.scripts import make_name

__all__ = [
    "GenericModel",
    "ConstantGenericModel",
    "LorentzianGenericModel",
    "AsymmetricLorentzianGenericModel",
    "LogNormalGenericModel",
    "GenericSourceModel",
]


class GenericModel(ModelBase):
    """Generic model class."""

    _type = "generic"

    def __init__(self, eval_axes=None, **kwargs):
        self.eval_axes = eval_axes
        super().__init__(**kwargs)

    def __call__(self, eval_ax):
        kwargs = {par.name: par.quantity for par in self.parameters}
        return self.evaluate(eval_ax, **kwargs)

    def __add__(self, model):
        return CompoundGenericModel([self, model], operator.add)

    def __mull__(self, model):
        return CompoundGenericModel([self, model], operator.mul)

    def __sub__(self, model):
        if not isinstance(model, GenericModel):
            raise TypeError(f"Invalid type: {type(model)}")
        return CompoundGenericModel([self, model], operator.sub)

    def __radd__(self, model):
        return self.__add__(model)

    def __rsub__(self, model):
        return self.__sub__(model)

    def plot(self, axis, ax=None, n_points=100, **kwargs):
        """Plot the model"""
        ax = plt.gca() if ax is None else ax

        axis_shape, _ = self._get_plot_shape(axis)

        with quantity_support():
            ax.plot(axis.center, axis_shape.quantity[:, 0, 0], **kwargs)

        self._plot_format_ax(ax, axis)

        return ax

    def plot_error(self, axis, ax=None, n_points=100, **kwargs):
        ax = plt.gca() if ax is None else ax

        kwargs.setdefault("facecolor", "black")
        kwargs.setdefault("alpha", 0.2)
        kwargs.setdefault("linewidth", 0)

        axis_shape, axis_shape_err = self._get_plot_shape(axis)
        y_lo = (axis_shape - axis_shape_err).quantity[:, 0, 0]
        y_hi = (axis_shape + axis_shape_err).quantity[:, 0, 0]

        with quantity_support():
            ax.fill_between(axis.center, y_lo, y_hi, **kwargs)

        self._plot_format_ax(ax, axis)
        return ax

    def _get_plot_shape(self, axis):
        shape = RegionNDMap.create(region=None, axes=[axis])
        shape_err = RegionNDMap.create(region=None, axes=[axis])

        shape.quantity, shape_err.quantity = self.evaluate_error(axis.center)

        return shape, shape_err

    def evaluate_error(self, axis, epsilon=1e-3):
        """Evaluate the model"""
        return self._propagate_error(epsilon=epsilon, fct=self, eval_ax=axis)

    def _propagate_error(self, epsilon, fct, **kwargs):
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

    def _plot_format_ax(self, ax, axis):
        ax.set_xlabel(f"{axis.name} [{axis.unit}]")
        norm_param = list(
            itertools.compress(
                self.parameters, [par._is_norm for par in self.parameters]
            )
        )[0]
        ax.set_ylabel(f"Model evaluation [{norm_param.unit}]")
        ax.set_ylim(0, None)
        return ax


class CompoundGenericModel(GenericModel):
    """Compound generic model. This class is used to combine different generic models."""

    tag = ["CompoundGenericModel", "compoundgen"]

    def __init__(self, models, operator):
        if not isinstance(models, list):
            models = [models]
        self.models = models
        self.operator = operator
        self._check_eval_axes(self.models)
        super().__init__(eval_axes=self.models[0].eval_axes)

    @property
    def parameters(self):
        parameters = Parameters()
        for model in self.models:
            parameters += model.parameters
        return parameters

    def __str__(self):
        s = "CompoundGenericModel\n"
        for model in self.models:
            s += model.__str__() + "\n"
        return s

    def __call__(self, eval_ax):
        val = [model(eval_ax) for model in self.models]
        if self.operator == operator.add:
            return self._add_vals(val)

    def __add__(self, model):
        if not isinstance(model, GenericModel):
            raise TypeError(f"Invalid type: {type(model)}")
        if isinstance(model, CompoundGenericModel):
            return CompoundGenericModel(self.models + model.models, operator.add)
        return CompoundGenericModel(self.models + [model], operator.add)

    def evaluate(self, eval_ax, *args):
        slice_list = self._get_slice(self.models)
        val = [
            model.evaluate(eval_ax, *args[s])
            for model, s in zip(self.models, slice_list)
        ]
        if self.operator == operator.add:
            return self._add_vals(val)

    @staticmethod
    def _add_vals(vals):
        return sum(vals)

    @staticmethod
    def _get_slice(models):
        """get the slice of the parameters"""
        slice_list = []
        lenght = [len(model.parameters) for model in models]
        for idx, le in enumerate(lenght):
            if idx == 0:
                slice_list.append(slice(0, le))
            else:
                slice_list.append(slice(lenght[idx - 1], lenght[idx - 1] + le))
        return slice_list

    @staticmethod
    def _check_eval_axes(models):
        eval_axes_list = [model.eval_axes for model in models]
        if len(set(eval_axes_list)) != 1:
            raise ValueError("All models must have the same `eval_axes`.")


class ConstantGenericModel(GenericModel):
    """Constant phase model."""

    tag = ["ConstantGenericModel", "constgen"]
    const = Parameter("const", 1, is_norm=True)

    @staticmethod
    def evaluate(eval_ax, const):
        """Evaluate the model"""
        return np.ones(np.atleast_1d(eval_ax).shape) * const


class LorentzianGenericModel(GenericModel):
    """Lorentzian generic model."""

    tag = ["LorentzianGenericModel", "lor"]
    amplitude = Parameter("amplitude", 1, is_norm=True)
    center = Parameter("center", 0)
    width = Parameter("width", 1)

    @staticmethod
    def evaluate(eval_ax, center, amplitude, width):
        """Evaluate the model"""
        return amplitude / (1 + np.power((eval_ax - center) / width, 2))


class AsymmetricLorentzianGenericModel(GenericModel):
    """Asymmetric Lorentzian generic model."""

    tag = ["AsymmetricLorentzianGenericModel", "asymlor"]
    amplitude = Parameter("amplitude", 1, is_norm=True)
    center = Parameter("center", 0.5)
    width_1 = Parameter("width_1", 0.1)
    width_2 = Parameter("width_2", 0.1)

    @staticmethod
    def evaluate(eval_ax, center, amplitude, width_1, width_2):
        """Evaluate the model"""
        l1 = amplitude / (1 + np.power((eval_ax - center) / width_1, 2))
        l2 = amplitude / (1 + np.power((eval_ax - center) / width_2, 2))
        return np.where(eval_ax < center, l1, l2)


class LogNormalGenericModel(GenericModel):
    """LogNormal generic model."""

    tag = ["LogNormalGenericModel", "lognorm"]
    amplitude = Parameter("amplitude", 1, is_norm=True)
    center = Parameter("center", 0)
    width = Parameter("width", 1)

    @staticmethod
    def evaluate(eval_ax, center, amplitude, width):
        """Evaluate the model"""
        return amplitude * (
            1
            / (eval_ax * width * np.sqrt(2 * np.pi))
            * np.exp(-0.5 * np.power((np.log(eval_ax) - center) / width, 2))
        )

    @classmethod
    def from_expected(cls, amplitude, center, width):
        """Create a lognormal model from expected values.

        FIXME: This is not a good solution because center and width are correlated.

        Parameters
        ----------
        amplitude : float
            Amplitude of the lognormal distribution at center.
        center : float
            Center of the lognormal distribution.
        width : float
            Width of the lognormal distribution.
        """
        mu = np.log(
            np.power(center, 2) / np.sqrt(np.power(width, 2) + np.power(center, 2))
        )
        sigma = np.sqrt(np.log(np.power(width, 2) / np.power(center, 2) + 1))
        a = amplitude * center * np.sqrt(2 * np.pi) * sigma
        return cls(amplitude=a, center=mu, width=sigma)


class GenericSourceModel(ModelBase):
    """Source model class.

    Parameters
    ----------

    """

    def __init__(
        self,
        models: GenericModel,
        name: Optional[str] = None,
        datasets_names: Optional[str] = None,
        covariance_data: Optional[Covariance] = None,
    ) -> None:
        self._models = models
        self._name = make_name(name)
        self.datasets_names = datasets_names
        super().__init__(covariance_data=covariance_data)

    def _check_covariance(self) -> None:
        if not self.parameters == self._covariance.parameters:
            self._covariance = Covariance.from_stack(
                [model.covariance for model in self._models],
            )

    @property
    def models(self) -> GenericModel:
        return self._models

    @models.setter
    def models(self, models: GenericModel) -> None:
        if not isinstance(models, list):
            models = [models]
        self._models = models

    @property
    def covariance(self) -> Covariance:
        self._check_covariance()

        for model in self._models:
            self._covariance.set_subcovariance(model.covariance)
        return self._covariance

    @covariance.setter
    def covariance(self, covariance: Covariance) -> None:
        self._check_covariance()
        self._covariance.data = covariance

        for model in self._models:
            model.covariance = self._covariance.get_subcovariance(
                model.covariance.parameters
            )

    @property
    def name(self) -> str:
        return self._name

    @property
    def parameters(self) -> Parameters:
        parameters = []

        for model in self._models:
            parameters.append(model.parameters)

        return Parameters.from_stack(parameters)

    def __call__(self, axes):
        assert len(axes) == len(self._models), "Invalid number of axes"
        return self.evaluate(axes)

    def evaluate(self, axes):
        value = 1
        for model, ax in zip(self._models, axes):
            value = value * model(axes)
        return value

    def to_dict(self, full_output=False):
        pass

    @classmethod
    def from_dict(cls, data):
        pass

    def __str__(self):
        return f"GenericSourceModel: {self.models.__class__.__name__}"

    @classmethod
    def create(cls, phase_model, **kwargs):
        pass

    def freeze(self):
        pass

    def unfreeze(self):
        pass
