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


def level_diagram(line: str) -> None:
    ion, _ = line.split("_")
    idx_num = utils.first_numid(ion)
    element = ion[:idx_num]
    ion_state = int(ion[idx_num:])
    ion_roman = element + utils.int_to_roman(ion_state)
    df_group = get_Elevels(line)

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
    idx_lines = np.arange(int(pad_lines / 2), pad_lines * nlines, pad_lines)
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
    grid = np.empty((nrows, ncols + pad_ikterm), dtype=object)
    grid[:] = " "  # fill grid with single spaces
    for i in range(nlines):
        grid[idx_up[i] : idx_down[i], idx_lines[i]] = (
            "|"  # fill level transition arrow bars
        )
    grid[idx_levels, :-pad_ikterm] = "-"  # fill energy level dashes
    grid[idx_down, idx_lines] = "v"  # fill lower level arrow caps
    grid[idx_up, idx_lines] = "^"  # fill upper level arrow caps
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
    lines_str = "".join([f"{str(w_line)+'A':^{pad_lines}}" for w_line in w_lines])
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
