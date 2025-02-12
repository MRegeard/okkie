import numpy as np
import scipy

__all__ = [
    "integrate_trapezoid",
    "integrate_gaussian",
    "integrate_lorentzian",
    "integrate_asymm_gaussian",
    "integrate_asymm_lorentzian",
    "integrate_periodic_gaussian",
    "integrate_periodic_lorentzian",
    "integrate_periodic_asymm_gaussian",
    "integrate_periodic_asymm_lorentzian",
]


def integrate_trapezoid(func, edge_min, edge_max):
    """"""
    xaxis = np.linspace(edge_min, edge_max, 1000)
    return scipy.integrate.trapezoid(y=func(xaxis), x=xaxis)


def integrate_gaussian(edge_min, edge_max, amplitude, mean, sigma):
    """Integrate a Gaussian.

    # TODO: Add maths formula.

    Parameters
    ----------
    edge_min, edge_max: float
        Edges of the integration.
    amplitude: float
        Ampltiude of the Gaussian.
    mean: float
        Mean of the Gaussian.
    sigma: float
        Sigma of the Gaussian.

    Returns
    -------
    integral: float
        Value of the integral.
    """
    edge_min = (edge_min - mean) / (np.sqrt(2) * sigma)
    edge_max = (edge_max - mean) / (np.sqrt(2) * sigma)
    amplitude = amplitude * sigma * np.sqrt(np.pi * 2)
    return amplitude / 2 * (scipy.special.erf(edge_max) - scipy.special.erf(edge_min))


def integrate_lorentzian(edge_min, edge_max, amplitude, mean, sigma):
    """Integrate a Lorentzian.

    # TODO: Add maths formula.

    Parameters
    ----------
    edge_min, edge_max: float
        Edges of the integration.
    amplitude: float
        Ampltiude of the Lorentzian.
    mean: float
        Mean of the Lorentzian.
    sigma: float
        Sigma of the Lorentzian.

    Returns
    -------
    integral: float
        Value of the integral.
    """
    return (
        amplitude
        * sigma
        * (np.arctan((edge_max - mean) / sigma) - np.arctan((edge_min - mean) / sigma))
    )


def integrate_asymm_gaussian(edge_min, edge_max, amplitude, mean, sigma_1, sigma_2):
    """Integrate an asymmetric Gaussian.

    # TODO: Add maths formula.

    Parameters
    ----------
    edge_min, edge_max: float
        Edges of the integration.
    amplitude: float
        Ampltiude of the asymmetric Gaussian.
    mean: float
        Mean of the asymmetric Gaussian.
    sigma_1: float
        Leading sigma of the asymmetric Gaussian.
    sigma_2: float
        Trailling sigma of the asymmetric Gaussian.

    Returns
    -------
    integral: float
        Value of the integral.
    """
    if edge_max <= mean:
        # Entirely on the left side of the mean
        return integrate_gaussian(
            edge_min=edge_min,
            edge_max=edge_max,
            amplitude=amplitude,
            mean=mean,
            sigma=sigma_1,
        )
    elif edge_min >= mean:
        # Entirely on the right side of the mean
        return integrate_gaussian(
            edge_min=edge_min,
            edge_max=edge_max,
            amplitude=amplitude,
            mean=mean,
            sigma=sigma_2,
        )
    else:
        # Split integral at the mean
        left_integral = integrate_gaussian(
            edge_min=edge_min,
            edge_max=edge_max,
            amplitude=amplitude,
            mean=mean,
            sigma=sigma_1,
        )
        right_integral = integrate_gaussian(
            edge_min=edge_min,
            edge_max=edge_max,
            amplitude=amplitude,
            mean=mean,
            sigma=sigma_2,
        )
        return left_integral + right_integral


def integrate_asymm_lorentzian(edge_min, edge_max, amplitude, mean, sigma_1, sigma_2):
    """Integrate an asymmetric Lorentzian.

    # TODO: Add maths formula.

    Parameters
    ----------
    edge_min, edge_max: float
        Edges of the integration.
    amplitude: float
        Ampltiude of the asymmetric Lorentzian.
    mean: float
        Mean of the asymmetric Lorentzian.
    sigma_1: float
        Leading sigma of the asymmetric Lorentzian.
    sigma_2: float
        Trailling sigma of the asymmetric Lorentzian.

    Returns
    -------
    integral: float
        Value of the integral.
    """
    if edge_max <= mean:
        # Entirely on the left side of the mean
        return integrate_lorentzian(
            edge_min=edge_min,
            edge_max=edge_max,
            amplitude=amplitude,
            mean=mean,
            sigma=sigma_1,
        )
    elif edge_min >= mean:
        # Entirely on the right side of the mean
        return integrate_lorentzian(
            edge_min=edge_min,
            edge_max=edge_max,
            amplitude=amplitude,
            mean=mean,
            sigma=sigma_2,
        )
    else:
        # Split integral at the mean
        left_integral = integrate_lorentzian(
            edge_min=edge_min,
            edge_max=edge_max,
            amplitude=amplitude,
            mean=mean,
            sigma=sigma_1,
        )
        right_integral = integrate_lorentzian(
            edge_min=edge_min,
            edge_max=edge_max,
            amplitude=amplitude,
            mean=mean,
            sigma=sigma_2,
        )
        return left_integral + right_integral


def integrate_periodic_gaussian(
    edge_min, edge_max, amplitude, mean, sigma, period=1, truncation=5
):
    """
    Compute the integral of a Gaussian function over [edge_min, edge_max] wrapping at `period`.

    # TODO: Add maths formula.

    Parameters
    ----------
    edge_min, edge_max : float
        Integration range.
    amplitude : float
        Amplitude of the Gaussian.
    mean : float
        Mean of the Gaussian.
    sigma : float
        Standard deviation of the Gaussian.
    period : float
        Period at which to wrap.
    truncation : int
        Number of periods to include in the sum.

    Returns
    -------
    integral : float
        Value of the integral.
    """
    mean %= period

    if (edge_max % period) < (edge_min % period):
        return integrate_periodic_gaussian(
            edge_min=edge_min,
            edge_max=period,
            amplitude=amplitude,
            mean=mean,
            sigma=sigma,
            period=period,
            truncation=truncation,
        ) + integrate_periodic_gaussian(
            edge_min=0,
            edge_max=edge_max,
            amplitude=amplitude,
            mean=mean,
            sigma=sigma,
            period=period,
            truncation=truncation,
        )

    k_values = np.arange(-truncation, truncation + 1)
    mean_images = mean + k_values * period

    integral = sum(
        integrate_gaussian(
            edge_min=edge_min,
            edge_max=edge_max,
            amplitude=amplitude,
            mean=mu,
            sigma=sigma,
        )
        for mu in mean_images
    )
    return integral


def integrate_periodic_asymm_gaussian(
    edge_min, edge_max, amplitude, mean, sigma_1, sigma_2, period=1, truncation=5
):
    """
    Compute the integral of an asymmetric Gaussian function over [edge_min, edge_max] wrapping at `period`.

    # TODO: Add maths formula.

    Parameters
    ----------
    edge_min, edge_max : float
        Integration range.
    amplitude : float
        Amplitude of the Gaussian.
    mean : float
        Mean of the Gaussian.
    sigma_1 : float
        Leading sigma of the asymmetric Gaussian.
    sigma_2 : float
        Trailling sigma of the asymmetric Gaussian.
    period : float
        Period of the Gaussian.
    truncation : int
        Number of periods to include in the sum.

    Returns
    -------
    integral : float
        Value of the integral.
    """
    mean %= period

    if (edge_max % period) < (edge_min % period):
        return integrate_periodic_asymm_gaussian(
            edge_min=edge_min,
            edge_max=period,
            amplitude=amplitude,
            mean=mean,
            sigma_1=sigma_1,
            sigma_2=sigma_2,
            period=period,
            truncation=truncation,
        ) + integrate_periodic_asymm_gaussian(
            edge_min=0,
            edge_max=edge_max,
            amplitude=amplitude,
            mean=mean,
            sigma_1=sigma_1,
            sigma_2=sigma_2,
            period=period,
            truncation=truncation,
        )

    k_values = np.arange(-truncation, truncation + 1)
    mean_images = mean + k_values * period

    integral = sum(
        integrate_asymm_gaussian(
            edge_min=edge_min,
            edge_max=edge_max,
            amplitude=amplitude,
            mean=mu,
            sigma_1=sigma_1,
            sigma_2=sigma_2,
        )
        for mu in mean_images
    )
    return integral


def integrate_periodic_lorentzian(
    edge_min, edge_max, amplitude, mean, sigma, period=1, truncation=5
):
    """
    Compute the integral of a Lorentzian function over [edge_min, edge_max] wrapping at `period`.

    # TODO: Add maths formula.

    Parameters
    ----------
    edge_min, edge_max : float
        Integration range.
    amplitude : float
        Amplitude of the Lorentzian.
    mean : float
        Mean of the Lorentzian.
    sigma : float
        Standard deviation of the Lorentzian.
    period : float
        Period at which to wrap.
    truncation : int
        Number of periods to include in the sum.

    Returns
    -------
    integral : float
        Value of the integral.
    """
    mean %= period

    if (edge_max % period) < (edge_min % period):
        return integrate_periodic_lorentzian(
            edge_min=edge_min,
            edge_max=period,
            amplitude=amplitude,
            mean=mean,
            sigma=sigma,
            period=period,
            truncation=truncation,
        ) + integrate_periodic_lorentzian(
            edge_min=0,
            edge_max=edge_max,
            amplitude=amplitude,
            mean=mean,
            sigma=sigma,
            period=period,
            truncation=truncation,
        )

    k_values = np.arange(-truncation, truncation + 1)
    mean_images = mean + k_values * period

    integral = sum(
        integrate_lorentzian(
            edge_min=edge_min,
            edge_max=edge_max,
            amplitude=amplitude,
            mean=mu,
            sigma=sigma,
        )
        for mu in mean_images
    )
    return integral


def integrate_periodic_asymm_lorentzian(
    edge_min, edge_max, amplitude, mean, sigma_1, sigma_2, period=1, truncation=5
):
    """
    Compute the integral of an asymmetric Lorentzian function over [edge_min, edge_max] wrapping at `period`.

    # TODO: Add maths formula.

    Parameters
    ----------
    edge_min, edge_max : float
        Integration range.
    amplitude : float
        Amplitude of the Lorentzian.
    mean : float
        Mean of the Lorentzian.
    sigma_1 : float
        Leading sigma of the asymmetric Lorentzian.
    sigma_2 : float
        Trailling sigma of the asymmetric Lorentzian.
    period : float
        Period of the Gaussian.
    truncation : int
        Number of periods to include in the sum.

    Returns
    -------
    integral : float
        Value of the integral.
    """
    mean %= period

    if (edge_max % period) < (edge_min % period):
        return integrate_periodic_asymm_lorentzian(
            edge_min=edge_min,
            edge_max=period,
            amplitude=amplitude,
            mean=mean,
            sigma_1=sigma_1,
            sigma_2=sigma_2,
            period=period,
            truncation=truncation,
        ) + integrate_periodic_asymm_lorentzian(
            edge_min=0,
            edge_max=edge_max,
            amplitude=amplitude,
            mean=mean,
            sigma_1=sigma_1,
            sigma_2=sigma_2,
            period=period,
            truncation=truncation,
        )

    k_values = np.arange(-truncation, truncation + 1)
    mean_images = mean + k_values * period

    integral = sum(
        integrate_asymm_lorentzian(
            edge_min=edge_min,
            edge_max=edge_max,
            amplitude=amplitude,
            mean=mu,
            sigma_1=sigma_1,
            sigma_2=sigma_2,
        )
        for mu in mean_images
    )
    return integral
