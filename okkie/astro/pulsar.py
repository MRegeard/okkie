import logging

import astropy.constants as const
import astropy.units as u
import numpy as np
import pint.models as pmodels
from astropy.coordinates import Angle, SkyCoord
from gammapy.utils.scripts import make_name, make_path

log = logging.getLogger(__name__)


__all__ = [
    "Pulsar",
    "GJ_density",
    "PulsarGeom",
    "rlc",
    "DEFAULT_R_NS",
    "DEFAULT_M_NS",
    "B_CONST",
]

DEFAULT_R_NS = 1.2e6 * u.cm

DEFAULT_M_NS = 1.4 * const.M_sun

B_CONST = np.sqrt(
    3 * const.mu0 * const.c**3 * DEFAULT_M_NS / (80 * np.pi**3 * DEFAULT_R_NS**4)
)


rlc = u.def_unit("rlc", represents=u.Quantity(1, ""))


def GJ_density(Omega, r, B, alpha, theta):
    """
    Return the Goldreich and Julian charge density (ref).

    Parameters
    ----------
    Omega: `~astropy.units.Quantity`
        Angular freqeuncy.
    r: `~astropy.units.Quantity`
        Radius in unit of rlc (use `~okkie.astro.rlc.rlc`).
    B: `~astropy.units.Quantity`
        Magnetic field strenght at `r`.
    alpha: `~astropy.units.Quantity`
        Magnetic angle. Angle between the rotation axis and the magnetic dipole axis.
    theta: `~astropy.units.Quantity`
        Latitude angle, define from the rotation axis pole.
    """
    cgs_density = (
        -Omega
        * B.to("G")
        * np.cos(alpha)
        / (2 * np.pi * const.c.cgs)
        * (1 / (1 - (r * np.sin(theta) / (1 * rlc)) ** 2))
    )
    return (cgs_density.value * u.Unit("Fr cm-3") / const.e.gauss) * const.e.si


class Pulsar:
    """Class representing a pulsar.

    Parameters
    ----------
    P0: `~astropy.units.Quantity`
        Period of the pulsar.
    P1: `~astropy.units.Quantity`
        Period derivative od the pulsar.
    B_NS: `~astropy.units.Quantity`, optional
        Strength of the magnetic field of the neutron star at the surface. If None,
        computed from `P0` and `P1`. Default is None.
    R_NS: `~astropy.units.Quantity`, optional
        Radius of the neutron star. Defaults is `1.2e6 cm`.
    age: `~astropy.units.Quantity`, optional
        Age of the pulsar. Default is None.
    geom: `PulsarGeom`, optional
        Pulsar geometry. If None, defaults to default value of `PulsarGeom`. Default is None.
    M_NS: `~astropy.units.Quantity`, optional
        Mass of the neutron star. Default is `1.4 * M_sun`.
    dist: `~astropy.units.Quantity`, optional
        Distance of the pulsar. Default is `1 kpc`.
    name: str, optional
        Name of the pulsar.
    positon: `~astropy.coordinates.SkyCoord`, optional
        Positon of the pulsar. If None, trying to set the pulsar position using `name`.
        If this fails, setting position to galactic center. Default is None.
    """

    def __init__(
        self,
        P0,
        P1,
        B_NS=None,
        R_NS=DEFAULT_R_NS,
        age=None,
        geom=None,
        M_NS=DEFAULT_M_NS,
        dist=1 * u.kpc,
        name=None,
        position=None,
    ):
        self.P0 = P0
        self.P1 = P1
        self.R_NS = R_NS
        self.B_NS = B_NS
        self.age = age
        if geom is None:
            self.geom = PulsarGeom()
        else:
            self.geom = geom
        self.M_NS = M_NS
        self.dist = dist
        self._name = make_name(name)
        self.position = position

    @property
    def B_NS(self):
        """Magnetic field strength at the neutron star surface."""
        return self._B_NS

    @B_NS.setter
    def B_NS(self, value):
        if value is None:
            self._B_NS = (B_CONST * np.sqrt(self.P0 * self.P1)).to("G")
        else:
            if not isinstance(value, u.Quantity):
                raise TypeError("Assigned value must be a `astropy.units.Quantity`.")
            self._B_NS = u.Quantity(value, "G")

    @property
    def P0(self):
        """Period of the pulsar."""
        return self._P0

    @P0.setter
    def P0(self, value):
        if not isinstance(value, u.Quantity):
            raise TypeError("Assigned value must be a `astropy.units.Quantity`.")
        self._P0 = u.Quantity(value, "s")

    @property
    def F0(self):
        """Frequency of the pulsar (1/`P0`)."""
        return 1 / self.P0

    @property
    def P1(self):
        """Derivative of the period of the pulsar."""
        return self._P1

    @P1.setter
    def P1(self, value):
        self._P1 = u.Quantity(value, "")

    @property
    def F1(self):
        "Derivative of the frequency of the pulsar (-`P1`/`P0`**2)." ""
        return -self.P1 / (self.P0) ** 2

    @property
    def R_NS(self):
        """Radius of the neutron star."""
        return self._R_NS

    @R_NS.setter
    def R_NS(self, value):
        if not isinstance(value, u.Quantity):
            raise TypeError("Assigned value must be a `astropy.units.Quantity`.")
        self._R_NS = u.Quantity(value, "cm")

    @property
    def age(self):
        """Age of the pulsar."""
        return self._age

    @age.setter
    def age(self, value):
        if value is None:
            self._age = None
        elif not isinstance(value, u.Quantity):
            raise TypeError("Assigned value must be a `astropy.units.Quantity`.")
        else:
            self._age = u.Quantity(value, "yr")

    @property
    def R_LC(self):
        """Radius of the light cylinder."""
        return const.c * self.P0 / (2 * np.pi)

    def B(self, radius):
        """Compute the strength of the magnetic field as a function of the radius.

        Parameters
        ----------
        radius: `~astropy.units.Quantity`
            Radius to compute the magnetic field value.
        Returns: `~astropy.units.Quantity`
            Value of the magnetic filed at the given radius.
        """
        if radius.unit is rlc:
            radius = radius.value * self.R_LC
        return (self.B_NS / (radius / self.R_NS) ** 3).to("G")

    @property
    def B_LC(self):
        """Value of the magnetic field at `R_LC`."""
        return self.B(1 * rlc)

    @property
    def M_NS(self):
        """Mass of the neutron star."""
        return self._M_NS

    @M_NS.setter
    def M_NS(self, value):
        if not isinstance(value, u.Quantity):
            raise TypeError("Assigned value must be a `astropy.units.Quantity`.")
        self._M_NS = u.Quantity(value, "g")

    @property
    def I_NS(self):
        """Moment of inertia of the neutron star."""
        return 2 / 5 * self.M_NS * self.R_NS**2

    @property
    def E_dot(self):
        """Energy loss of the pulsar."""
        return (-((2 * np.pi) ** 2) * self.I_NS * self.P1 / (self.P0) ** 3).to(
            "erg s-1"
        )

    @property
    def dist(self):
        """Distance of the pulsar."""
        return self._dist

    @dist.setter
    def dist(self, value):
        if not isinstance(value, u.Quantity):
            raise TypeError("assigned value must be a `astropy.units.quantity`.")
        self._dist = u.Quantity(value, "kpc")

    @property
    def name(self):
        """Name of the pulsar."""
        return self._name

    @property
    def position(self):
        """Position of the pulsar."""
        return self._position

    @position.setter
    def position(self, value):
        if value is None:
            try:
                position = SkyCoord.from_name(self.name)
                self._position = position
            except Exception as e:
                log.warning("""Error while trying to get the pulsar position. This is likely due to not properly defined pulsar name:
{self.name}, or connection error.
Setting position to Galactic center.""")
                self._position = SkyCoord(0, 0, unit="deg", frame="galactic")
        else:
            if not isinstance(value, SkyCoord):
                raise TypeError(
                    "`position` must be an instance of `~astropy.coordinates.SkyCoord`."
                )
            self._position = value

    @property
    def Omega(self):
        """Rotational angular frequency."""
        return 2 * np.pi / self.P0

    @property
    def geom(self):
        """Pulsar geometry."""
        return self._geom

    @geom.setter
    def geom(self, value):
        if not isinstance(value, PulsarGeom):
            raise TypeError(f"`geom` must be of type `PulsarGeom`, got {value.type}")
        self._geom = value

    def GJ_density(self, radius, theta=None):
        """Compute the Goldreich and Julian charge density.

        Parameters
        ----------
        radius: `~astropy.units.Quantity`
            Radius to compute the density.
        theta: `~astropy.units.Quantity` or float
            Latitude angle to compute the density. The Latitude is taken from the rotation axis pole.

        Returns
        -------
        density: Goldreich and Julian density.
        """
        theta = theta or self.geom.zeta
        if radius.unit.is_equivalent(u.cm):
            radius = (radius / self.R_LC).to("") * rlc
        return GJ_density(self.Omega, radius, self.B(radius), self.geom.alpha, theta)

    @classmethod
    def from_frequency(cls, F0, F1, **kwargs):
        """Create from frequency instead of period."""
        if not isinstance(F0, u.Quantity):
            log.info("No unit found for `F0`, assuming `Hz`.")
            F0 = u.Quantity(F0, "Hz")
        if not isinstance(F1, u.Quantity):
            log.info("No unit found for `F1`, assuming `Hz2`.")
            F1 = u.Quantity(F1, "Hz2")
        P0 = 1 / F0
        P1 = -F1 / (F0) ** 2
        return cls(P0=P0, P1=P1, **kwargs)

    @classmethod
    def from_ephemeris_file(cls, filename, **kwargs):
        """Create from an ephemeris file."""
        filename = make_path(filename)
        model = pmodels.get_model(filename)
        name = model["PSR"].value
        F0 = u.Quantity(model["F0"].value, "Hz")
        F1 = u.Quantity(model["F1"].value, "Hz2")
        raj = Angle(model["RAJ"].value, unit=u.h)
        decj = Angle(model["DECJ"].value, unit=u.deg)
        position = SkyCoord(raj, decj, frame="icrs")
        kwargs.setdefault("position", position)
        kwargs.setdefault("name", name)
        kwargs.setdefault("F0", F0)
        kwargs.setdefault("F1", F1)

        return cls.from_frequency(**kwargs)


class PulsarGeom:
    def __init__(self, alpha=None, zeta=None):
        """Class representing a pulsar geometry.

        Parameters
        ----------
        alpha:
            Magnetic inclination angle (angle between the rotation axis and the magnetic axis).
        zeta:
            Viewing angle (angle between the rotation axis and the line of sight).

        """
        self.alpha = alpha or 90 * u.deg
        self.zeta = zeta or 90 * u.deg

    @property
    def alpha(self):
        """Magnetic inclination angle."""
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if not isinstance(value, u.Quantity):
            raise TypeError("assigned value must be a `astropy.units.quantity`.")
        self._alpha = u.Quantity(value, "deg")

    @property
    def zeta(self):
        """Viewing angle."""
        return self._zeta

    @zeta.setter
    def zeta(self, value):
        if not isinstance(value, u.Quantity):
            raise TypeError("assigned value must be a `astropy.units.quantity`.")
        self._zeta = u.Quantity(value, "deg")
