import astropy.units as u
from astropy.time import Time
from gammapy.maps import MapAxis, WcsGeom
from gammapy.modeling import CovarianceMixin, Parameters
from gammapy.modeling.models import (
    ModelBase,
    Models,
    SkyModel,
    SpatialModel,
    SpectralModel,
    TemplateNPredModel,
    TemporalModel,
)
from gammapy.utils.scripts import make_name
from .phase import PhaseModel

# TODO: import CovarianceMixin to gammapy.modeling.covariance in gammapy 1.3

__all__ = ["SourceModel"]


class SourceModel(CovarianceMixin, ModelBase):
    tag = ["SourceModel", "source-model"]
    _apply_irf_default = {"exposure": True, "psf": True, "edisp": True}

    def __init__(
        self,
        spectral_model=None,
        spatial_model=None,
        temporal_model=None,
        phase_model=None,
        name=None,
        apply_irf=None,
        datasets_names=None,
        covariance_data=None,
    ):
        self.spatial_model = spatial_model
        self.spectral_model = spectral_model
        self.temporal_model = temporal_model
        self._name = make_name(name)

        if apply_irf is None:
            apply_irf = self._apply_irf_default.copy()

        self.apply_irf = apply_irf
        self.datasets_names = datasets_names
        self._check_unit()

        super().__init__(covariance_data=covariance_data)

    @property
    def _models(self):
        models = (
            self.spectral_model,
            self.spatial_model,
            self.temporal_model,
            self.phase_model,
        )
        return [model for model in models if model is not None]

    def _check_unit(self):
        axis = MapAxis.from_energy_bounds(
            "0.1 TeV", "10 TeV", nbin=1, name="energy_true"
        )

        geom = WcsGeom.create(skydir=self.position, npix=(2, 2), axes=[axis])
        time = Time(55555, format="mjd")
        if self.apply_irf["exposure"]:
            ref_unit = u.Unit("cm-2 s-1 MeV-1")
        else:
            ref_unit = u.Unit("")
        obt_unit = self.spectral_model(axis.center).unit

        if self.spatial_model:
            obt_unit = obt_unit * self.spatial_model.evaluate_geom(geom).unit
            ref_unit = ref_unit / u.sr

        if self.temporal_model:
            if u.Quantity(self.temporal_model(time)).unit.is_equivalent(
                self.spectral_model(axis.center).unit
            ):
                obt_unit = (
                    (
                        self.temporal_model(time)
                        * self.spatial_model.evaluate_geom(geom).unit
                    )
                    .to(obt_unit.to_string())
                    .unit
                )
            else:
                obt_unit = obt_unit * u.Quantity(self.temporal_model(time)).unit

        if not obt_unit.is_equivalent(ref_unit):
            raise ValueError(
                f"SkyModel unit {obt_unit} is not equivalent to {ref_unit}"
            )

    @property
    def name(self):
        return self._name

    @property
    def parameters(self):
        parameters = []

        parameters.append(self.spectral_model.parameters)

        if self.spatial_model is not None:
            parameters.append(self.spatial_model.parameters)

        if self.temporal_model is not None:
            parameters.append(self.temporal_model.parameters)

        return Parameters.from_stack(parameters)

    @property
    def parameters_unique_names(self):
        """List of unique parameter names. Return formatted as par_type.par_name."""
        names = []
        for model in self._models:
            for par_name in model.parameters_unique_names:
                components = [model.type, par_name]
                name = ".".join(components)
                names.append(name)
        return names

    @property
    def spatial_model(self):
        """Spatial model as a `~gammapy.modeling.models.SpatialModel` object."""
        return self._spatial_model

    @spatial_model.setter
    def spatial_model(self, model):
        if not (model is None or isinstance(model, SpatialModel)):
            raise TypeError(f"Invalid type: {model!r}")

        self._spatial_model = model

    @property
    def spectral_model(self):
        """Spectral model as a `~gammapy.modeling.models.SpectralModel` object."""
        return self._spectral_model

    @spectral_model.setter
    def spectral_model(self, model):
        if not (model is None or isinstance(model, SpectralModel)):
            raise TypeError(f"Invalid type: {model!r}")
        self._spectral_model = model

    @property
    def temporal_model(self):
        """Temporal model as a `~gammapy.modeling.models.TemporalModel` object."""
        return self._temporal_model

    @temporal_model.setter
    def temporal_model(self, model):
        if not (model is None or isinstance(model, TemporalModel)):
            raise TypeError(f"Invalid type: {model!r}")

        self._temporal_model = model

    @property
    def phase_model(self):
        """Phase model as a `okkie.modeling.models.PhaseModel` object."""
        return self._phase_model

    @phase_model.setter
    def phase_model(self, model):
        if not (model is None or isinstance(model, PhaseModel)):
            raise TypeError(f"Invalid type: {model!r}")

        self._phase_model = model

    @property
    def position(self):
        """Position as a `~astropy.coordinates.SkyCoord`."""
        return getattr(self.spatial_model, "position", None)

    @property
    def position_lonlat(self):
        """Spatial model center position `(lon, lat)` in radians and frame of the model."""
        return getattr(self.spatial_model, "position_lonlat", None)

    @property
    def evaluation_bin_size_min(self):
        """Minimal spatial bin size for spatial model evaluation."""
        if (
            self.spatial_model is not None
            and self.spatial_model.evaluation_bin_size_min is not None
        ):
            return self.spatial_model.evaluation_bin_size_min
        else:
            return None

    @property
    def evaluation_radius(self):
        """Evaluation radius as an `~astropy.coordinates.Angle`."""
        return self.spatial_model.evaluation_radius

    @property
    def evaluation_region(self):
        """Evaluation region as an `~astropy.coordinates.Angle`."""
        return self.spatial_model.evaluation_region

    @property
    def frame(self):
        return self.spatial_model.frame

    def __add__(self, other):
        if isinstance(other, (Models, list)):
            return Models([self, *other])
        elif isinstance(other, (SkyModel, TemplateNPredModel)):
            return Models([self, other])
        else:
            raise TypeError(f"Invalid type: {other!r}")

    def __radd__(self, model):
        return self.__add__(model)

    def __call__(self, lon, lat, energy, time=None):
        return self.evaluate(lon, lat, energy, time)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"spatial_model={self.spatial_model!r}, "
            f"spectral_model={self.spectral_model!r})"
            f"temporal_model={self.temporal_model!r})"
            f"phase_model={self.phase_model!r})"
        )

    def evaluate(self, lon=None, lat=None, energy=None, time=None, phase=None):
        """Evaluate the model at given points.

        The model evaluation follows numpy broadcasting rules.

        Return differential surface brightness cube.
        At the moment in units: ``cm-2 s-1 TeV-1 deg-2``.

        Parameters
        ----------
        lon, lat : `~astropy.units.Quantity`, optional
            Spatial coordinates. Default is None.
        energy : `~astropy.units.Quantity`, optional
            Energy coordinate. Default is None.
        time: `~astropy.time.Time`, optional
            Time coordinate. Default is None.
        phase: `~numpy.ndarray`
            Phase coordinate. Default is None.

        Returns
        -------
        value : `~astropy.units.Quantity`
            Model value at the given point.
        """
        if (lon and lat and energy and time and phase) is None:
            raise ValueError("Found no axis to evaluate models")
        value = 1
        if (self.spectral_model is not None) and (energy is not None):
            value = self.spectral_model(energy)  # pylint:disable=not-callable
        # TODO: case if self.temporal_model is not None, introduce time in arguments ?

        if (self.spatial_model is not None) and ((lon and lat) is not None):
            if self.spatial_model.is_energy_dependent and (energy is not None):
                spatial = self.spatial_model(lon, lat, energy)
            else:
                spatial = self.spatial_model(lon, lat)

            value = value * spatial  # pylint:disable=not-callable

        if (self.temporal_model is not None) and (time is not None):
            value = value * self.temporal_model(time)

        if (self.phase_model is not None) and (phase is not None):
            value = value * self.phase_model(phase)

        return value
