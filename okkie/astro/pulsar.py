import logging
import astropy.constants as const
import astropy.units as u
import numpy as np
import pint.models as pmodels
from astropy.coordinates import Angle, SkyCoord
from gammapy.utils.scripts import make_name, make_path

log = logging.getLogger(__name__)


__all__ = ["Pulsar", "GJ_density", "PulsarGeom", "rlc"]

DEFAULT_R_NS = 1.2e6 * u.cm

DEFAULT_M_NS = 1.4 * const.M_sun

B_CONST = np.sqrt(
    3 * const.mu0 * const.c**3 * DEFAULT_M_NS / (80 * np.pi**3 * DEFAULT_R_NS**4)
)


rlc = u.def_unit("rlc", represents=u.Quantity(1, ""))


def GJ_density(Omega, r, B, alpha, zeta):
    """
    Return the Goldreich and Juian charge density (ref).

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
    zeta: `~astropy.units.Quantity`
        Viewing angle. Angle between the rotation axis and the line of sight.
    """
    cgs_density = (
        -Omega
        * B.to("G")
        * np.cos(alpha)
        / (2 * np.pi * const.c.cgs)
        * (1 / (1 - (r * np.sin(zeta) / (1 * rlc)) ** 2))
    )
    return (cgs_density.value * u.Unit("Fr cm-3") / const.e.gauss) * const.e.si


class Pulsar:
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
        return self._P0

    @P0.setter
    def P0(self, value):
        if not isinstance(value, u.Quantity):
            raise TypeError("Assigned value must be a `astropy.units.Quantity`.")
        self._P0 = u.Quantity(value, "s")

    @property
    def F0(self):
        return 1 / self.P0

    @property
    def P1(self):
        return self._P1

    @P1.setter
    def P1(self, value):
        self._P1 = u.Quantity(value, "")

    @property
    def R_NS(self):
        return self._R_NS

    @R_NS.setter
    def R_NS(self, value):
        if not isinstance(value, u.Quantity):
            raise TypeError("Assigned value must be a `astropy.units.Quantity`.")
        self._R_NS = u.Quantity(value, "cm")

    @property
    def age(self):
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
        return const.c * self.P0 / (2 * np.pi)

    def B(self, radius):
        if radius.unit is rlc:
            radius = radius.value * self.R_LC
        return (self.B_NS / (radius / self.R_NS) ** 3).to("G")

    @property
    def B_LC(self):
        return self.B(1 * rlc)

    @property
    def M_NS(self):
        return self._M_NS

    @M_NS.setter
    def M_NS(self, value):
        if not isinstance(value, u.Quantity):
            raise TypeError("Assigned value must be a `astropy.units.Quantity`.")
        self._M_NS = u.Quantity(value, "g")

    @property
    def I_NS(self):
        return 2 / 5 * self.M_NS * self.R_NS**2

    @property
    def E_dot(self):
        return (-((2 * np.pi) ** 2) * self.I_NS * self.P1 / (self.P0) ** 3).to(
            "erg s-1"
        )

    @property
    def dist(self):
        return self._dist

    @dist.setter
    def dist(self, value):
        if not isinstance(value, u.Quantity):
            raise TypeError("assigned value must be a `astropy.units.quantity`.")
        self._dist = u.Quantity(value, "kpc")

    @property
    def name(self):
        return self._name

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        if value is None:
            try:
                position = SkyCoord.from_name(self.name)
            except Exception as e:
                raise GetPulsarPositionError(
                    f"""Error while trying to get the pulsar position. This is likely due to not properly defined pulsar name:
                                             {self.name}, or connection error. See error message below.\n{e}"""
                )
            self._position = position
        else:
            if not isinstance(value, SkyCoord):
                raise TypeError(
                    "`position` must be an instance of `~astropy.coordinates.SkyCoord`."
                )
            self._position = value

    @property
    def Omega(self):
        return 2 * np.pi / self.P0

    @property
    def geom(self):
        return self._geom

    @geom.setter
    def geom(self, value):
        if not isinstance(value, PulsarGeom):
            raise TypeError(f"`geom` must be of type `PulsarGeom`, got {value.type}")
        self._geom = value

    def GJ_density(self, radius, theta=None):
        theta = theta or self.geom.zeta
        if radius.unit.is_equivalent(u.cm):
            radius = (radius / self.R_LC).to("") * rlc
        return GJ_density(self.Omega, radius, self.B(radius), self.geom.alpha, theta)

    @classmethod
    def from_frequency(cls, F0, F1, **kwargs):
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
    def from_timing_model(cls, filename, **kwargs):
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
        self.alpha = alpha or 0 * u.deg
        self.zeta = zeta or 90 * u.deg

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if not isinstance(value, u.Quantity):
            raise TypeError("assigned value must be a `astropy.units.quantity`.")
        self._alpha = u.Quantity(value, "deg")

    @property
    def zeta(self):
        return self._zeta

    @zeta.setter
    def zeta(self, value):
        if not isinstance(value, u.Quantity):
            raise TypeError("assigned value must be a `astropy.units.quantity`.")
        self._zeta = u.Quantity(value, "deg")


class GetPulsarPositionError(Exception):
    pass
