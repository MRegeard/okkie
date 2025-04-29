import numpy as np
from gammapy.modeling import Parameter
from gammapy.modeling.models import SpectralModel

__all__ = ["SuperExpCutoffPowerLaw4FGLDR3SpectralModelCor"]


class SuperExpCutoffPowerLaw4FGLDR3SpectralModelCor(SpectralModel):
    r"""Spectral super exponential cutoff power-law model used for 4FGL-DR3.

    See equations (2) and (3) of https://arxiv.org/pdf/2201.11184.pdf
    For more information see :ref:`super-exp-cutoff-powerlaw-4fgl-dr3-spectral-model`.

    Parameters
    ----------
    amplitude : `~astropy.units.Quantity`
        :math:`\phi_0`.  Default is 1e-12 cm-2 s-1 TeV-1.
    reference : `~astropy.units.Quantity`
        :math:`E_0`. Default is 1 TeV.
    expfactor : `~astropy.units.Quantity`
        :math:`a`, given as dimensionless value. Default is 1e-2.
    index_1 : `~astropy.units.Quantity`
        :math:`\Gamma_1`. Default is 1.5.
    index_2 : `~astropy.units.Quantity`
        :math:`\Gamma_2`. Default is 2.
    """

    tag = ["SuperExpCutoffPowerLaw4FGLDR3SpectralModelCor", "secpl-4fgl-dr3-cor"]
    amplitude = Parameter(
        name="amplitude",
        value="1e-12 cm-2 s-1 TeV-1",
        scale_method="scale10",
        interp="log",
    )
    reference = Parameter("reference", "1 TeV", frozen=True)
    expfactor = Parameter("expfactor", "1e-2")
    index_1 = Parameter("index_1", 1.5)
    index_2 = Parameter("index_2", 2)

    @staticmethod
    def evaluate(energy, amplitude, reference, expfactor, index_1, index_2):
        """Evaluate the model (static function)."""
        # https://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/source_models.html#PLSuperExpCutoff4
        pwl = amplitude * (energy / reference) ** (-index_1)
        cutoff = (energy / reference) ** (expfactor / index_2) * np.exp(
            expfactor / index_2**2 * (1 - (energy / reference) ** index_2)
        )

        mask = np.abs(index_2 * np.log(energy / reference)) < 1e-2
        ln_ = np.log(energy[mask] / reference)
        power = -expfactor * (
            ln_ / 2.0 + index_2 / 6.0 * ln_**2.0 + index_2**2.0 / 24.0 * ln_**3
        )
        cutoff[mask] = (energy[mask] / reference) ** power
        return pwl * cutoff

    @property
    def e_peak(self):
        r"""Spectral energy distribution peak energy (`~astropy.units.Quantity`).

        This is the peak in E^2 x dN/dE and is given by Eq. 21 of https://iopscience.iop.org/article/10.3847/1538-4357/acee67:

        .. math::
            E_{Peak} = E_{0} \left[1+\frac{\Gamma_2}{a}(2 - \Gamma_1)\right]^{\frac{1}{\Gamma_2}}
        """
        reference = self.reference.quantity
        index_1 = self.index_1.quantity
        index_2 = self.index_2.quantity
        expfactor = self.expfactor.quantity
        index_0 = index_1 - expfactor / index_2
        if (
            ((index_2 < 0) and (index_0 < 2))
            or (expfactor <= 0)
            or ((index_2 > 0) and (index_0 >= 2))
        ):
            return np.nan * reference.unit
        return reference * (1 + (index_2 / expfactor) * (2 - index_1)) ** (1 / index_2)
