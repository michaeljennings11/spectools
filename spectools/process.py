import numpy as np

from spectools import constants


def v_doppler(lambda_obs: np.ndarray, lambda_source: np.ndarray) -> np.ndarray:
    """Calculates Doppler shift in km/s of observed wavelength
       compared to source wavelength.

    .. math::
        v = c (\\lambda_obs/\\lambda_source - 1)
    """
    lambda_obs = np.asarray(lambda_obs)
    lambda_source = np.asarray(lambda_source)
    if np.any(lambda_obs <= 0):
        raise ValueError("Wavelength values lambda_obs must be greater than zero.")
    if np.any(lambda_source <= 0):
        raise ValueError("Wavelength values lambda_source must be greater than zero.")

    if lambda_source.ndim > 0:
        lambda_source = lambda_source[:, None]

    velocity = constants.C_KMS * (lambda_obs / lambda_source - 1.0)

    return velocity


def w_doppler(velocity: np.ndarray, lambda_source: np.ndarray) -> np.ndarray:
    """Calculates wavelength in Angstroms from Doppler shift in km/s
       relative to source wavelength.

    .. math::
        \\lambda_obs = \\lambda_source (v / c + 1)
    """
    velocity = np.asarray(velocity)
    lambda_source = np.asarray(lambda_source)
    if np.any(lambda_source <= 0):
        raise ValueError("Wavelength values lambda_source must be greater than zero.")

    if lambda_source.ndim > 0:
        lambda_source = lambda_source[:, None]

    lambda_obs = lambda_source * (velocity / constants.C_KMS + 1.0)

    return lambda_obs


def dw_doppler(velocity: float, lambda_source: float) -> float:
    """Calculates Doppler shifted wavelength interval.

    .. math::
        \\Delta{\\lambda} = \\lambda_0 (v / c)
    """
    if lambda_source <= 0:
        raise ValueError("Wavelength value lambda_source must be greater than zero.")
    dw = lambda_source * (velocity / constants.C_KMS)
    return dw


def z_correct(lambda_obs: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Redshift corrects wavelength.

    .. math::
        \\lambda_0 = \\lambda / (1 + z)
    """
    lambda_obs = np.asarray(lambda_obs)
    z = np.asarray(z)
    if np.any(lambda_obs <= 0):
        raise ValueError("Wavelength value lambda_obs must be greater than zero.")
    if np.any(z < 0):
        raise ValueError("Redshift z must be positive.")

    if z.ndim > 0:
        z = z[:, None]
    return lambda_obs / (1.0 + z)


def z_project(lambda_rest: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Redshifts wavelength.

    .. math::
        \\lambda = \\lambda_0 (1 + z)
    """
    lambda_rest = np.asarray(lambda_rest)
    z = np.asarray(z)
    if np.any(lambda_rest <= 0):
        raise ValueError("Wavelength values lambda_rest must be greater than zero.")
    if np.any(z < 0):
        raise ValueError("Redshift z must be positive.")

    if z.ndim > 0:
        z = z[:, None]
    return lambda_rest * (1.0 + z)


def convolve_gauss(x, y, sigma: float):
    y_conv = []
    for xs in x:
        gkv = np.exp(-((x - xs) ** 2) / (2 * (sigma**2)))
        gkv /= gkv.sum()
        y_conv.append((y * gkv).sum())

    return y_conv
