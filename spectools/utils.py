import numpy as np

C_CMS = 2.99792e10  # speed of light [cm/s]
C_KMS = 2.99792e5  # speed of light [km/s]

ROMAN = [
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
]


def int_to_roman(number):
    result = ""
    for arabic, roman in ROMAN:
        (factor, number) = divmod(number, arabic)
        result += roman * factor
    return result


def first_numid(text):
    idx_num = [i for i, char in enumerate(text) if char.isdigit()][0]
    return idx_num


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def nlayers(n):
    return 2 * n - 1


def w_doppler(v, lambda_s):
    lambda_o = lambda_s * (v / C_KMS + 1.0)
    return lambda_o


def convolve_gauss(x, y, sigma):
    y_conv = []
    for xs in x:
        gkv = np.exp(-((x - xs) ** 2) / (2 * (sigma**2)))
        gkv /= gkv.sum()
        y_conv.append((y * gkv).sum())

    return y_conv
