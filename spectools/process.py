import numpy as np

from spectools import constants


def w_doppler(v, lambda_s):
    lambda_o = lambda_s * (v / constants.C_KMS + 1.0)
    return lambda_o


def convolve_gauss(x, y, sigma):
    y_conv = []
    for xs in x:
        gkv = np.exp(-((x - xs) ** 2) / (2 * (sigma**2)))
        gkv /= gkv.sum()
        y_conv.append((y * gkv).sum())

    return y_conv
