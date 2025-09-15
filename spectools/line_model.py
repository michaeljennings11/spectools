import numpy as np
import pandas as pd

from spectools import constants, line_data, process


class LineModel:
    def __init__(self, line_name, v_arr=None, vres=None):
        self.Line = line_data.LineData(line_name)
        self.line_name = self.Line.line_name
        self.line_data = self.Line.line_data
        [setattr(self, key, value) for key, value in self.line_data.items()]
        self.dE = self.Line.dE
        if vres is None:
            self.vres = 1
        else:
            self.vres = vres
        if v_arr == None:
            self.v_arr = np.arange(-1000, 1000 + self.vres, self.vres)
        else:
            self.v_arr = v_arr
        self.wave_arr = process.w_doppler(self.v_arr, self.wave)
        self.flux_arr = None


class VoigtModel(LineModel):
    def __init__(self, line_name, v_arr=None, vres=None):
        super().__init__(line_name, v_arr=None, vres=None)

    def _H_approx(self, a: float, x: np.ndarray):
        """Approximation of the dimensionless convolution of Lorentzian and Maxwellian distributions.
        Taken from Smith, A. et al. (2015).
        """
        Ai = [
            15.75328153963877,
            286.9341762324778,
            19.05706700907019,
            28.22644017233441,
            9.526399802414186,
            35.29217026286130,
            0.8681020834678775,
        ]
        Bi = [
            0.0003300469163682737,
            0.5403095364583999,
            2.676724102580895,
            12.82026082606220,
            3.21166435627278,
            32.032981933420,
            9.0328158696,
            23.7489999060,
            1.82106170570,
        ]
        z = x**2
        conditions = [z <= 3, (z > 3) & (z < 25), z >= 25]  # piecewise conditions

        def H1(z: np.ndarray) -> np.ndarray:
            return np.exp(-z) * (
                1.0
                - a
                * (
                    Ai[0]
                    + Ai[1] / (z - Ai[2] + Ai[3] / (z - Ai[4] + Ai[5] / (z - Ai[6])))
                )
            )

        def H2(z: np.ndarray) -> np.ndarray:
            return np.exp(-z) + a * (
                Bi[0]
                + Bi[1]
                / (
                    z
                    - Bi[2]
                    + Bi[3] / (z + Bi[4] + Bi[5] / (z - Bi[6] + Bi[7] / (z - Bi[8])))
                )
            )

        def H3(z: np.ndarray) -> np.ndarray:
            return (a / np.sqrt(np.pi)) / (z - 1.5 - 1.5 / (z - 3.5 - 5 / (z - 5.5)))

        Hs = [H1, H2, H3]
        H = np.piecewise(z, conditions, Hs)
        return H

    def abs_profile(
        self,
        b: float,
        log_n: float,
        vout: float = 0,
        cf: float = 1,
        wave_arr: np.ndarray = None,
        xspace: str = "wavelength",
        yspace: str = "flux",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculates absorption profile from the radiative transfer equation assuming only absorption."""

        wave0 = self.wave
        f = self.f
        A = self.A
        if wave_arr == None:
            wave_arr = self.wave_arr

        N = 10**log_n

        x0 = wave0 * (
            (vout * 1e5) / constants.C_CMS + 1.0
        )  # wavelength from line center in Angstroms
        x = (constants.C_KMS / b) * (
            1.0 - x0 / wave_arr
        )  # number of doppler widths from line center
        a = wave0 * 1.0e-8 * A / (4.0 * np.pi * b * 1e5)  # damping parameter
        phi = (
            wave0 * 1.0e-8 * self._H_approx(a, x) / np.sqrt(np.pi) / (b * 1e5)
        )  # line profile function
        sigma_cross = (
            np.pi * constants.E**2 * f / constants.M_E / constants.C_CMS * phi
        )  # ion cross section in wavelength space
        tau = sigma_cross * N  # optical depth
        self.flux_arr = (
            cf * np.exp(-tau) + 1 - cf
        )  # normalized flux assuming partial covering

        if xspace == "wavelength":
            xdata = wave_arr
        elif xspace == "velocity":
            xdata = v_arr
        if yspace == "flux":
            ydata = self.flux_arr
        elif yspace == "tau":
            ydata = tau

        return xdata, ydata


class GaussianModel(LineModel):
    def __init__(self, line_name, v_arr=None, vres=None):
        super().__init__(line_name, v_arr=None, vres=None)

    def abs_profile(
        self,
        b: float,
        log_n: float,
        vout: float = 0,
        cf: float = 1,
        wave_arr: np.ndarray = None,
        xspace: str = "wavelength",
        yspace: str = "flux",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculates absorption profile from the radiative transfer equation assuming only absorption."""

        wave0 = self.wave
        f = self.f
        A = self.A
        if wave_arr == None:
            wave_arr = self.wave_arr

        N = 10**log_n

        x0 = wave0 * (
            (vout * 1e5) / constants.C_CMS + 1.0
        )  # wavelength from line center in Angstroms
        x = (constants.C_KMS / b) * (
            1.0 - x0 / wave_arr
        )  # number of doppler widths from line center
        phi = (1 / (b * 1e5)) * np.exp(
            -((x * wave_arr / wave0) ** 2)
        )  # line profile function
        sigma_cross = (
            f
            * np.pi
            * wave0
            * 1.0e-8
            * constants.E**2
            / (constants.M_E * constants.C_CMS)
            * phi
        )  # ion cross section in wavelength space
        tau = sigma_cross * N  # optical depth
        self.flux_arr = (
            cf * np.exp(-tau) + 1 - cf
        )  # normalized flux assuming partial covering

        if xspace == "wavelength":
            xdata = wave_arr
        elif xspace == "velocity":
            xdata = v_arr
        if yspace == "flux":
            ydata = self.flux_arr
        elif yspace == "tau":
            ydata = tau

        return xdata, ydata
