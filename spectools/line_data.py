import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from spectools import utils

dir_path = os.path.dirname(os.path.realpath(__file__))
NIST_lines_dir = os.path.join(dir_path, "NIST_lines")


def load_elementDataFrame(element):
    full_key_names = [
        "element",
        "ion",
        "wave",
        "A",
        "f",
        "Acc",
        "El",
        "Eu",
        "confl",
        "terml",
        "Jl",
        "confu",
        "termu",
        "Ju",
    ]
    if element != "H":
        key_names = full_key_names
        ncols = 13
    else:
        ncols = 11
        key_names = full_key_names[2:]
    wave_label = "ritz_wl_vac(A)"
    file = NIST_lines_dir + f"/{element}_NIST.csv"
    try:
        df = (
            pd.read_csv(file, usecols=np.arange(0, ncols + 1))
            .replace(
                ["=", '"', r"\[", r"\]", r"\(", r"\)", r"\+x", r"\?", "&dagger", ";"],
                ["", "", "", "", "", "", "", "", "", ""],
                regex=True,
            )
            .infer_objects()
        )
        df[[wave_label, "Aki(s^-1)", "fik", "Ei(eV)", "Ek(eV)"]] = df[
            [wave_label, "Aki(s^-1)", "fik", "Ei(eV)", "Ek(eV)"]
        ].astype("float")
        df.columns = key_names
        if element == "H":
            df.insert(loc=0, column=full_key_names[1], value=1)
            df.insert(loc=0, column=full_key_names[0], value="H")
        return df
    except OSError:
        raise ValueError(
            f"{element} file not found in NIST_lines directory.\n\
You may download the {element}.csv from the NIST webpage if it is available\n\
and place it in the NIST_lines directory."
        )


def load_ionDataFrame(ion):
    idx_num = utils.first_numid(ion)
    element = ion[:idx_num]
    df = load_elementDataFrame(element)
    ion_state = int(ion[idx_num:])
    ok_ion_states = set(df["ion"])
    ion_state_exception_message = f"Desired ionization state is not within element's atomic number range.\n\
    Please choose an ionization state for {element} within {min(ok_ion_states)}-{max(ok_ion_states)}"
    if ion_state not in ok_ion_states:
        raise ValueError(ion_state_exception_message)
    return df.loc[(df["ion"] == ion_state)]


def load_lineDataFrame(line_name, return_df=False):
    ion, line = line_name.split("_")
    df = load_ionDataFrame(ion)
    line_list = df["wave"].values
    _, idx = utils.find_nearest(line_list, float(line))
    df_line = df.iloc[idx]
    if return_df:
        return df_line, df
    else:
        return df_line


def get_Elevels(line):
    line_series, df = load_lineDataFrame(line, return_df=True)
    iterm = line_series["terml"]
    kterm = line_series["termu"]
    iconf = line_series["confl"]
    kconf = line_series["confu"]
    df_group = df.loc[
        (df["terml"] == iterm)
        & (df["termu"] == kterm)
        & (df["confl"] == iconf)
        & (df["confu"] == kconf)
    ]
    return df_group


def get_transitionProbabilities(line):
    df_group = get_Elevels(line)
    ntrans = len(df_group)
    Jis = df_group["Jl"].values
    Jks = df_group["Ju"].values
    Ji_set = np.unique(Jis)
    Jk_set = np.unique(Jks)
    Ji_pairs = Ji_set
    Jk_pairs = Jk_set
    id_lower_pairs = [np.argwhere(Jis == Ji_pair).ravel() for Ji_pair in Ji_pairs]
    id_upper_pairs = [np.argwhere(Jks == Jk_pair).ravel() for Jk_pair in Jk_pairs]
    prob_up = np.zeros(ntrans)
    for i, id_lower_pair in enumerate(id_lower_pairs):
        A_lower_pair = df_group["A"].values[id_lower_pair]
        prob_upward_pair = A_lower_pair / A_lower_pair.sum()
        prob_up[id_lower_pairs[i]] = prob_upward_pair
    prob_down = np.zeros(ntrans)
    for i, id_upper_pair in enumerate(id_upper_pairs):
        A_upper_pair = df_group["A"].values[id_upper_pair]
        prob_downward_pair = A_upper_pair / A_upper_pair.sum()
        prob_down[id_upper_pairs[i]] = prob_downward_pair
    return prob_up, prob_down


def level_diagram(line: str) -> None:
    ion, _ = line.split("_")
    idx_num = utils.first_numid(ion)
    element = ion[:idx_num]
    ion_state = int(ion[idx_num:])
    ion_roman = element + utils.int_to_roman(ion_state)
    df_group = get_Elevels(line)
    p_ups, p_dns = get_transitionProbabilities(line)

    #     if ion=='HI':
    #         wave_label = 'ritz_wl_vac(A)'
    #     else:
    #         wave_label = 'obs_wl_vac(A)'

    # grab line group data
    w_lines = df_group["wave"].values
    nlines = len(w_lines)
    Elowers = df_group["El"].values
    Euppers = df_group["Eu"].values
    iterm = df_group["terml"].values[0]
    kterm = df_group["termu"].values[0]
    if kterm == "":
        kterm = df_group["confu"].values[0]
    Jis = df_group["Jl"].values
    Jks = df_group["Ju"].values

    # format line group data
    kiterms = np.array([kterm, iterm])
    Jis_set = sorted(set(Jis))
    Jks_set = sorted(set(Jks))
    Js = np.concatenate((Jis_set, Jks_set))
    El_set = set(Elowers)
    Eu_set = set(Euppers)
    E_set = sorted(El_set.union(Eu_set))
    nlower = len(El_set)
    nupper = len(Eu_set)
    nlevels = len(E_set)
    El_rank = rankdata(Elowers, method="dense") - 1
    Eu_rank = rankdata(Euppers, method="dense") + (nlower - 1)

    # formatting constants
    nmiddle = 5
    pad_lines = 14
    pad_ikterm = 1
    pad_jterm = 5
    indent = f"{'':{pad_lines}}"

    # formatting indices
    idx_ulevels = np.arange(0, utils.nlayers(nupper), 2)
    idx_llevels = -(np.arange(0, utils.nlayers(nlower), 2)[::-1] + 1)
    idx_levels = np.concatenate((idx_ulevels, idx_llevels))
    idx_linesl = np.arange(int(pad_lines / 2), pad_lines * nlines, pad_lines) - 1
    idx_linesr = np.arange(int(pad_lines / 2), pad_lines * nlines, pad_lines)
    idx_kterm = int(np.median(idx_ulevels))
    idx_iterm = int(np.median(idx_llevels))
    idx_kiterm = np.array([idx_kterm, idx_iterm])

    idx_down = np.empty(nlines, dtype=int)
    idx_up = np.empty(nlines, dtype=int)
    for i in range(nlines):
        idx_down[i] = (idx_levels[::-1][El_rank[i]]) - 1
        idx_up[i] = (idx_levels[::-1][Eu_rank[i]]) + 1

    # create level diagram grid
    nrows = utils.nlayers(nupper) + nmiddle + utils.nlayers(nlower)
    ncols = nlines * pad_lines

    # idx_uprob = utils.nlayers(nupper) + nmiddle//2 - 1
    idx_uprob = idx_ulevels + 2
    idx_dprob = idx_llevels - 2

    grid = np.empty((nrows, ncols + pad_ikterm), dtype=object)
    grid[:] = " "  # fill grid with single spaces
    for i in range(nlines):
        # left side
        grid[idx_up[i] : idx_down[i] + 1, idx_linesl[i]] = (
            "|"  # fill level transition arrow bars
        )
        p_up = p_ups[i]
        if p_up == 1.0:
            p_str = np.array(list("100%"))
            grid[idx_uprob, idx_linesl[i] - 4 : idx_linesl[i]] = p_str
        else:
            p_str = np.array(list(str(round(p_ups[i], 3) * 100) + "%"))
            grid[idx_uprob, idx_linesl[i] - 5 : idx_linesl[i]] = p_str
        # right side
        grid[idx_up[i] - 1 : idx_down[i], idx_linesr[i]] = (
            "|"  # fill level transition arrow bars
        )
        p_dn = p_dns[i]
        if p_dn == 1.0:
            p_str = np.array(list("100%"))
            grid[idx_dprob, idx_linesr[i] + 1 : idx_linesr[i] + 5] = p_str
        else:
            p_str = np.array(list(str(round(p_dns[i], 3) * 100) + "%"))
            grid[idx_dprob, idx_linesr[i] + 1 : idx_linesr[i] + 6] = p_str
    grid[idx_levels, :-pad_ikterm] = "-"  # fill energy level dashes
    grid[idx_down, idx_linesr] = "v"  # fill lower level arrow caps
    grid[idx_up, idx_linesl] = "^"  # fill upper level arrow caps
    for i, kiterm in enumerate(kiterms):
        grid[idx_kiterm[i], -1] = kiterm  # fill kiterms to right side of grid

    # annotate grid with energy levels and Jterms
    energy_str = np.empty(nrows, dtype=object)
    j_str = np.empty(nrows, dtype=object)
    energy_str[:] = " "  # fill energy level E(eV) strings with single spaces
    j_str[:] = " "  # fill energy level J strings with single spaces
    for i, E in enumerate(E_set):
        energy_str[idx_levels[::-1][i]] = f"{str(E)+'eV':>{pad_lines}}"
        j_str[idx_levels[::-1][i]] = f"{str(Js[i]):>{pad_jterm}}"

    # print diagram ion title
    print(f"{indent}{ion_roman:^{ncols}}")

    # print grid to console
    for i, gridline in enumerate(grid):
        print(f"{energy_str[i]:>{pad_lines}}" + "".join(gridline) + j_str[i])

    # print line wavelengths under grid
    lines_str = "".join([f"{str(w_line)+u'\u212B':^{pad_lines}}" for w_line in w_lines])
    print(f"{indent}{lines_str}")


class LineData:
    def __init__(self, line_name: str):
        self.line_name = line_name
        self.line_data = load_lineDataFrame(line_name)
        [setattr(self, key, value) for key, value in self.line_data.items()]
        self.dE = self.Eu - self.El

    def get_linegroup(self):
        return get_Elevels(self.line_name)

    def print_leveldiagram(self):
        level_diagram(self.line_name)


class AbsLineModel:
    def __init__(self, line_name: str, v_arr=None, vres=None):
        self.line_name = line_name
        self.line_data = load_lineDataFrame(line_name)
        [setattr(self, key, value) for key, value in self.line_data.items()]

        if vres is None:
            self.vres = 1
        else:
            self.vres = vres
        if v_arr == None:
            self.v_arr = np.arange(-1000, 1000 + self.vres, self.vres)
        else:
            self.v_arr = v_arr
        self.wave_arr = utils.w_doppler(self.v_arr, self.wave)
        self.flux_arr = None

    def H_approx(self, a, x):
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
        conditions = [z <= 3, (z > 3) & (z < 25), z >= 25]

        def H1(z):
            return np.exp(-z) * (
                1.0
                - a
                * (
                    Ai[0]
                    + Ai[1] / (z - Ai[2] + Ai[3] / (z - Ai[4] + Ai[5] / (z - Ai[6])))
                )
            )

        def H2(z):
            return np.exp(-z) + a * (
                Bi[0]
                + Bi[1]
                / (
                    z
                    - Bi[2]
                    + Bi[3] / (z + Bi[4] + Bi[5] / (z - Bi[6] + Bi[7] / (z - Bi[8])))
                )
            )

        def H3(z):
            return (a / np.sqrt(np.pi)) / (z - 1.5 - 1.5 / (z - 3.5 - 5 / (z - 5.5)))

        Hs = [H1, H2, H3]
        H = np.piecewise(z, conditions, Hs)
        return H

    def voigt_profile(
        self,
        b: float,
        log_n: float,
        vout: float = 0,
        cf: float = 1,
        wave_arr: np.ndarray = None,
        wave0: float = None,
        f: float = None,
        A: float = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if wave0 == None:
            wave0 = self.wave
        if f == None:
            f = self.f
        if A == None:
            A = self.A
        if wave_arr == None:
            wave_arr = self.wave_arr
        c = utils.C_CMS
        m_e = 9.1094e-28  # g
        e = 4.8032e-10  # cgs units
        cspeed = c / 1e5
        #         print(f"vout = {vout}")
        x0 = wave0 * (
            (vout * 1e5) / c + 1.0
        )  # This is the wavelength from line center in Angstroms
        C_a = np.sqrt(np.pi) * e**2 * f * wave0 * 1.0e-8 / m_e / c / (b * 1e5)
        a = wave0 * 1.0e-8 * A / (4.0 * np.pi * b * 1e5)
        x = (cspeed / b) * (1.0 - x0 / wave_arr)
        N = 10**log_n
        tau = np.float64(C_a) * N * self.H_approx(a, x)
        #         print(f"cf = {cf}")
        self.flux_arr = (
            cf * np.exp(-tau) + 1 - cf
        )  # This converts the optical depth to Flux units using the radiative transfer equation
        return wave_arr, self.flux_arr, tau

    # def convolve(self, vsig: float, x_arr: np.ndarray = None, y_arr: np.ndarray = None):
    #     if x_arr is None:
    #         x_arr = self.wave_arr
    #     if y_arr is None:
    #         y_arr = self.flux_arr
    #     c = 2.99792e10
    #     cspeed = c / 1e5
    #     wave0 = self.line_wave
    #     sigma_pixel = vsig / (
    #         (x_arr[1] - x_arr[0]) * cspeed / wave0
    #     )  # & $  ; gaussian sigma of convolution kernel in pixels
    #     c_arr = gaussian_filter(y_arr, sigma_pixel)
    #     return c_arr
