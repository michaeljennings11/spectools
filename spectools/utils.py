import numpy as np

from spectools import constants

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
