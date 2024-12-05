import logging

import astropy.constants as const
import astropy.units as u
import numpy as np
from gammapy.modeling import Parameter, Parameters
from gammapy.modeling.models import SpectralModel
from naima.extern.validator import (
    validate_array,
    validate_physical_type,
    validate_scalar,
)
from naima.radiative import BaseElectron
from naima.utils import trapz_loglog

from .pulsar import Pulsar

log = logging.getLogger(__name__)

__all__ = ["Curvature", "NaimaSpectralModel"]

e = const.e.gauss

mec2 = (const.m_e * const.c**2).cgs
mec2_unit = u.Unit(mec2)

ar = (4 * const.sigma_sb / const.c).to("erg/(cm3 K4)")
r0 = (e**2 / mec2).to("cm")


def _validate_ene(ene):
    from astropy.table import Table

    if isinstance(ene, dict or Table):
        try:
            ene = validate_array(
                "energy", u.Quantity(ene["energy"]), physical_type="energy"
            )
        except KeyError:
            raise TypeError("Table or dict does not have 'energy' column")
    else:
        if not isinstance(ene, u.Quantity):
            ene = u.Quantity(ene)
        validate_physical_type("energy", ene, physical_type="energy")

    return ene


class NaimaSpectralModel(SpectralModel):
    r"""A wrapper for Naima models.

    For more information see :ref:`naima-spectral-model`.

    Parameters
    ----------
    radiative_model : `~naima.models.BaseRadiative`
        An instance of a radiative model defined in `~naima.models`.
    distance : `~astropy.units.Quantity`, optional
        Distance to the source. If set to 0, the intrinsic differential
        luminosity will be returned. Default is 1 kpc.
    seed : str or list of str, optional
        Seed photon field(s) to be considered for the `radiative_model` flux computation,
        in case of a `~naima.models.InverseCompton` model. It can be a subset of the
        `seed_photon_fields` list defining the `radiative_model`. Default is the whole list
        of photon fields.
    nested_models : dict
        Additional parameters for nested models not supplied by the radiative model,
        for now this is used  only for synchrotron self-compton model.
    """

    tag = ["NaimaSpectralModel", "naima"]

    def __init__(
        self,
        radiative_model,
        distance=1.0 * u.kpc,
        seed=None,
        nested_models=None,
        use_cache=False,
    ):
        import naima

        self.radiative_model = radiative_model
        self.radiative_model._memoize = use_cache
        self.distance = u.Quantity(distance)
        self.seed = seed

        if nested_models is None:
            nested_models = {}

        self.nested_models = nested_models

        if isinstance(self.particle_distribution, naima.models.TableModel):
            param_names = ["amplitude"]
        else:
            param_names = self.particle_distribution.param_names

        parameters = []

        for name in param_names:
            value = getattr(self.particle_distribution, name)
            parameter = Parameter(name, value)
            parameters.append(parameter)

        # In case of a synchrotron radiative model, append B to the fittable parameters
        if "B" in self.radiative_model.param_names:
            value = self.radiative_model.B
            parameter = Parameter("B", value)
            parameters.append(parameter)

        # In case of a synchrotron self compton model, append B and Rpwn to the fittable parameters
        if self.include_ssc:
            B = self.nested_models["SSC"]["B"]
            radius = self.nested_models["SSC"]["radius"]
            parameters.append(Parameter("B", B))
            parameters.append(Parameter("radius", radius, frozen=True))

        if "Rc" in self.radiative_model.param_names:
            value = self.radiative_model.Rc
            parameter = Parameter("Rc", value)
            parameters.append(parameter)

        self.default_parameters = Parameters(parameters)
        self.ssc_energy = np.logspace(-7, 9, 100) * u.eV
        super().__init__()

    @property
    def include_ssc(self):
        """Whether the model includes an SSC component."""
        import naima

        is_ic_model = isinstance(self.radiative_model, naima.models.InverseCompton)
        return is_ic_model and "SSC" in self.nested_models

    @property
    def ssc_model(self):
        """Synchrotron model."""
        import naima

        if self.include_ssc:
            return naima.models.Synchrotron(
                self.particle_distribution,
                B=self.B.quantity,
                Eemax=self.radiative_model.Eemax,
                Eemin=self.radiative_model.Eemin,
            )

    @property
    def particle_distribution(self):
        """Particle distribution."""
        return self.radiative_model.particle_distribution

    def _evaluate_ssc(
        self,
        energy,
    ):
        """
        Compute photon density spectrum from synchrotron emission for synchrotron self-compton
        model, assuming uniform synchrotron emissivity inside a sphere of radius R (see Section
        4.1 of Atoyan & Aharonian 1996).

        Based on :
        https://naima.readthedocs.io/en/latest/examples.html#crab-nebula-ssc-model

        """
        Lsy = self.ssc_model.flux(
            self.ssc_energy, distance=0 * u.cm
        )  # use distance 0 to get luminosity
        phn_sy = Lsy / (4 * np.pi * self.radius.quantity**2 * const.c) * 2.24
        # The factor 2.24 comes from the assumption on uniform synchrotron
        # emissivity inside a sphere

        if "SSC" not in self.radiative_model.seed_photon_fields:
            self.radiative_model.seed_photon_fields["SSC"] = {
                "isotropic": True,
                "type": "array",
                "energy": self.ssc_energy,
                "photon_density": phn_sy,
            }
        else:
            self.radiative_model.seed_photon_fields["SSC"]["photon_density"] = phn_sy

        dnde = self.radiative_model.flux(
            energy, seed=self.seed, distance=self.distance
        ) + self.ssc_model.flux(energy, distance=self.distance)
        return dnde

    def _update_naima_parameters(self, **kwargs):
        """Update Naima model parameters."""
        for name, value in kwargs.items():
            setattr(self.particle_distribution, name, value)

        if "B" in self.radiative_model.param_names:
            self.radiative_model.B = self.B.quantity

    def evaluate(self, energy, **kwargs):
        """Evaluate the model.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy to evaluate the model at.

        Returns
        -------
        dnde : `~astropy.units.Quantity`
            Differential flux at given energy.
        """
        self._update_naima_parameters(**kwargs)

        if self.include_ssc:
            dnde = self._evaluate_ssc(energy.flatten())
        elif self.seed is not None:
            dnde = self.radiative_model.flux(
                energy.flatten(), seed=self.seed, distance=self.distance
            )
        else:
            dnde = self.radiative_model.flux(energy.flatten(), distance=self.distance)

        dnde = dnde.reshape(energy.shape)
        unit = 1 / (energy.unit * u.cm**2 * u.s)
        return dnde.to(unit)

    def to_dict(self, full_output=True):
        # for full_output to True otherwise broken
        return super().to_dict(full_output=True)

    @classmethod
    def from_dict(cls, data, **kwargs):
        raise NotImplementedError(
            "Currently the NaimaSpectralModel cannot be read from YAML"
        )

    @classmethod
    def from_parameters(cls, parameters, **kwargs):
        raise NotImplementedError(
            "Currently the NaimaSpectralModel cannot be built from a list of parameters."
        )


class Curvature(BaseElectron):
    def __init__(self, particle_distribution, Rc=1e6 * u.cm, **kwargs):
        super().__init__(particle_distribution)

        self.Rc = validate_scalar("Rc", Rc, physical_type=u.get_physical_type(u.m))
        self.Eemin = 1 * u.GeV
        self.Eemax = 1e9 * mec2
        self.nEed = 100
        self.param_names += [
            "Rc",
        ]
        self.__dict__.update(**kwargs)

    def _spectrum(self, photon_energy):
        """Compute intrinsic synchrotron differential spectrum for energies in ``photon_energy``

        Compute synchrotron for random magnetic field according to approximation
        of Aharonian, Kelner, and Prosekin 2010, PhysRev D 82, 3002
        (`arXiv:1006.1045 <http://arxiv.org/abs/1006.1045>`_).

        Parameters
        ----------
        photon_energy : :class:`~astropy.units.Quantity` instance
        Photon energy array.
        """
        outspecene = _validate_ene(photon_energy)

        from scipy.special import cbrt

        def Gtilde(x):
            """
            AKP10 Eq. D7

            Factor ~2 performance gain in using cbrt(x)**n vs x**(n/3.)
            """
            gt1 = 1.808 * cbrt(x) / np.sqrt(1 + 3.4 * cbrt(x) ** 2.0)
            gt2 = 1 + 2.210 * cbrt(x) ** 2.0 + 0.347 * cbrt(x) ** 4.0
            gt3 = 1 + 1.353 * cbrt(x) ** 2.0 + 0.217 * cbrt(x) ** 4.0
            return gt1 * (gt2 / gt3) * np.exp(-x)

        def Ftilde(x):
            """
            AKP10 Eq. D6

            Factor ~2 performance gain in using cbrt(x)**n vs x**(n/3.)
            """
            ft1 = 2.15 * cbrt(x) * np.sqrt(cbrt(1 + 3.06 * x))
            ft2 = 1 + 0.884 * cbrt(x) ** 2.0 + 0.471 * cbrt(x) ** 4.0
            ft3 = 1 + 1.64 * cbrt(x) ** 2.0 + 0.217 * cbrt(x) ** 4.0
            return ft1 * (ft2 / ft3) * np.exp(-x)

        log.debug("calc_sy: Starting synchrotron computation with AKB2010...")

        CS1_0 = np.sqrt(3) * e.value**2 * np.vstack(self._gam)
        CS1_1 = (
            2
            * np.pi
            * const.hbar.cgs.value
            * self.Rc.to("cm").value
            * outspecene.to("erg").value
        )
        CS1 = CS1_0 / CS1_1
        Ec = 3 * const.hbar.cgs.value * const.c.cgs.value * self._gam**3
        Ec /= 2 * self.Rc.to("cm").value

        EgEc = outspecene.to("erg").value / np.vstack(Ec)
        dNdE = CS1 * Ftilde(EgEc)
        # return units
        spec = (
            trapz_loglog(np.vstack(self._nelec) * dNdE, self._gam, axis=0) / u.s / u.erg
        )
        spec = spec.to("1/(s eV)")

        return spec


class PulsarSynchrotron:
    def __init__(self, source, e_peak=None, spectral_model=None, luminosity=None):
        self.source = source
        if (e_peak is None) and (spectral_model is None):
            raise ValueError("`e_peak` and `spectral_model` are both set to `None`.")
        if e_peak is None:
            e_peak = self._init_e_peak_from_model(spectral_model)
        else:
            self.e_peak = e_peak
        self.spectral_model = spectral_model

    @staticmethod
    def _init_e_peak_from_model(model):
        try:
            return model.e_peak
        except AttributeError:
            raise TypeError(
                "`e_peak` is set to `None` and `spectral_model` does not implement an `e_peak` attribute."
            )

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, value):
        if not isinstance(value, Pulsar):
            raise TypeError("`source` must be an instance of `Pulsar`.")
        self._source = value

    @property
    def e_peak(self):
        return self._e_peak

    @e_peak.setter
    def e_peak(self, value):
        if not isinstance(value, u.Quantity):
            raise TypeError(
                "`e_peak` must be an instance of `~astropy.units.Quantity`."
            )
        self._e_peak = u.Quantity(value, "GeV")

    @property
    def spectral_model(self):
        return self._spectral_model

    @spectral_model.setter
    def spectral_model(self, value):
        if not isinstance(value, SpectralModel):
            raise TypeError(
                "`spectral_model` must be an instance of `~gammapy.modeling.models.SpectralModel`."
            )
        self._spectral_model = value