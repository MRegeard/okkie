import numpy as np
from gammapy.datasets import Dataset
from gammapy.maps import Geom, Map
from gammapy.modeling.models import DatasetModels, Models
from gammapy.stats import (
    CashCountsStatistic,
    WStatCountsStatistic,
)
from gammapy.utils.scripts import make_name

__all__ = ["CountsDataset"]


class CountsDataset(Dataset):
    """
    Counts dataset.

    Parameters
    ----------
    counts: `gammapy.maps.Map`, optional
        Counts map.
    models: `gammapy.modeling.models.Models`, optional
        Models.
    name: str, optional
        Name of the dataset.
    meta_table: `astropy.table.Table`, optional
        Table listing information on observations used to create the dataset.
        One line per observation for stacked datasets.
    """

    tag = "CountDataset"

    def __init__(
        self,
        counts=None,
        models=None,
        name=None,
        meta_table=None,
        mask_fit=None,
        mask_safe=None,
        stat_type="cash",
    ):
        self.counts = counts
        self.models = models
        self._name = make_name(name)
        self.meta_table = meta_table
        self.mask_fit = mask_fit
        self.mask_safe = mask_safe

        self.stat_type = stat_type

    @property
    def models(self) -> Models:
        """Models set on the dataset as a `~gammapy.modeling.models.Models`."""
        return self._models

    @models.setter
    def models(self, models: Models) -> None:
        """Models setter."""
        if models is not None:
            models = DatasetModels(models)
            models = models.select(datasets_names=self.name)

        self._models = models

    def __str__(self) -> str:
        """String representation."""
        str_ = f"{self.__class__.__name__}\n"
        str_ += "-" * len(self.__class__.__name__) + "\n"
        str_ += "\n"
        str_ += "\t{:32}: {{name}} \n\n".format("Name")

        str_ += "\t{:32}: {{counts:.0f}} \n".format("Total counts")
        str_ += "\t{:32}: {{npred:.2f}}\n\n".format("Predicted counts")

        str_ += "\t{:32}: {{n_bins}} \n".format("Number of total bins")
        str_ += "\t{:32}: {{n_fit_bins}} \n\n".format("Number of fit bins")

        # likelihood section
        str_ += "\t{:32}: {{stat_type}}\n".format("Fit statistic type")
        str_ += "\t{:32}: {{stat_sum:.2f}}\n\n".format(
            "Fit statistic value (-2 log(L))"
        )

        info = self.info_dict()
        str_ = str_.format(**info)

        # model section
        n_models, n_pars, n_free_pars = 0, 0, 0
        if self.models is not None:
            n_models = len(self.models)
            n_pars = len(self.models.parameters)
            n_free_pars = len(self.models.parameters.free_parameters)

        str_ += "\t{:32}: {} \n".format("Number of models", n_models)
        str_ += "\t{:32}: {}\n".format("Number of parameters", n_pars)
        str_ += "\t{:32}: {}\n\n".format("Number of free parameters", n_free_pars)

        if self.models is not None:
            str_ += "\t" + "\n\t".join(str(self.models).split("\n")[2:])

        return str_.expandtabs(tabsize=2)

    @property
    def _geom(self) -> Geom:
        """Dataset geometry"""
        return self.counts.geom or ValueError("`counts` needs to be defined.")

    def npred(self):
        """Predicted counts map."""
        total_npred = Map.from_geom(self._geom, dtype=np.float64)

        for model in self.models:
            data = model(phase=self.counts.geom.axes["phase"].center).value
            npred = Map.from_geom(self._geom, data=data, dtype=np.float64)
            total_npred.stack(npred)
        return total_npred

    @classmethod
    def create(cls, geom, **kwargs):
        """Create empty `CountsDataset` with the given geometry.

        Parameters
        ----------
        geom : `~gammapy.maps.Geom`
            Reference geometry

        Returns
        -------
        empty_counts_dataset : `CountsDataset`
            Empty counts dataset
        """
        counts = Map.from_geom(geom)
        return cls(counts=counts, **kwargs)

    def plot_residuals(self, ax=None, method="diff", **kwargs):
        counts = self.counts.copy()
        npred = self.npred()
        residuals = self._compute_residuals(counts, npred, method)
        if self.stat_type == "wstat":
            counts_off = self.counts_off

            with np.errstate(invalid="ignore"):
                alpha = self.background / counts_off

            mu_sig = self.npred_signal()
            stat = WStatCountsStatistic(
                n_on=counts,
                n_off=counts_off,
                alpha=alpha,
                mu_sig=mu_sig,
            )
        elif self.stat_type == "cash":
            stat = CashCountsStatistic(counts.data, npred.data)
        excess_error = stat.error

        if method == "diff":
            yerr = excess_error
        elif method == "diff/sqrt(model)":
            yerr = excess_error / np.sqrt(npred.data)
        else:
            raise ValueError(
                'Invalid method, choose between "diff" and "diff/sqrt(model)"'
            )

        kwargs.setdefault("color", kwargs.pop("c", "black"))
        ax = residuals.plot(ax, yerr=yerr, **kwargs)
        counts = self.counts.copy()
        npred = self.npred()
        residuals = self._compute_residuals(counts, npred, method)
        ax.axhline(0, color=kwargs["color"], lw=0.5)

        label = self._residuals_labels[method]
        ax.set_ylabel(f"Residuals ({label})")
        ax.set_yscale("linear")
        ymin = 1.05 * np.nanmin(residuals.data - yerr)
        ymax = 1.05 * np.nanmax(residuals.data + yerr)
        ax.set_ylim(ymin, ymax)
        return ax

    def residuals_cumsum(self, method="diff"):
        counts = self.counts.copy()
        npred = self.npred()
        residuals = self._compute_residuals(counts, npred, method)
        residuals.data = residuals.data.cumsum()
        return residuals

    def plot_residuals_cumsum(self, ax=None, method="diff", **kwargs):
        residuals = self.residuals_cumsum(method=method)

        kwargs.setdefault("color", kwargs.pop("c", "black"))
        ax = residuals.plot(ax, **kwargs)
        ax.axhline(0, color=kwargs["color"], lw=0.5)

        label = self._residuals_labels[method]
        ax.set_ylabel(f"Residuals ({label})")
        ax.set_yscale("linear")
        ymin = 1.05 * np.nanmin(residuals.data)
        ymax = 1.05 * np.nanmax(residuals.data)
        ax.set_ylim(ymin, ymax)
        return ax
