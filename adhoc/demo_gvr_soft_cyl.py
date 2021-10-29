from typing import Union

from pathlib import Path

import numpy as np

from iww_gvr.main import fill, gvr_soft_cyl, writeWWINP_cyl, ww_item_cyl, zoneDEF_cyl
from iww_gvr.meshtal_module import Meshtal
from tqdm import tqdm


def demo_gvr_soft(
    gvrname: str,
    beta: int,
    soft: float,
    fname: Path,
    meshtally_number: int,
    degree: str,
    option: str,
) -> ww_item_cyl:
    mesh = Meshtal(fname)
    m = mesh.mesh[meshtally_number]
    assert not m.cart, "Demo is to check cylinder mesh conversion"
    gvr = gvr_soft_cyl(m, mesh, gvrname)
    gvr.info()
    # while True:
    #     degree = input(" Please insert Yes or No for Hole-filling approach: ")
    #     if degree == "Yes" or degree == "No":
    #         break
    if degree == "Yes":
        # while True:
        #     option = input(
        #         " Please select an option for the problem geometry [all, auto]: "
        #     )
        #     if option == "all" or option == "auto":
        #         break
        #     else:
        #         print(" Not a valid option")
        zoneID, factor = zoneDEF_cyl(gvr, option)
        gvr.zoneID = zoneID

    mesh_values = gvr.wwme[0][0]
    fluxinp_max = mesh_values.max()
    norm_factor = 2.0 / (beta + 1) / fluxinp_max
    ww_inp = mesh_values * norm_factor
    if soft != 1.0:
        assert 0 < soft < 1.0
        ww_inp = np.power(ww_inp, soft)
    # with tqdm(unit=" K slices", desc=" GVR", total=len(gvr.K) - 1) as bar:
    #     # for k in range(0, len(gvr.K) - 1):
    #     for k in range(0, len(gvr.K) - 2):
    #         for j in range(0, len(gvr.J) - 1):
    #             for i in range(0, len(gvr.I) - 1):
    #                 ww_inp[k, j, i] = np.power(
    #                     mesh_values[k, j, i] / fluxinp_max * (2 / (beta + 1)),
    #                     soft,
    #                 )  # Van Vick/A.Davis
    #         bar.update()

    gvr.wwme[0][0] = ww_inp
    if degree == "Yes":  # Hole filling
        ww_mod = gvr.wwme[0]
        for g in range(int(gvr.d["B1_ne"][0])):
            z = []  # A 3d zoneID
            for i in range(len(zoneID)):
                z.append([])
                for m in range(len(gvr.J) - 1):
                    z[i].append(zoneID[i])
            while True:
                # For some unknown reason the fill operation when using cyl coordinates
                # does not clear all the holes at once, several iterations are required
                holes = ww_mod[g] == 0
                holes = holes * z
                holes = holes == 1
                if len(np.argwhere(holes)) == 0:
                    break
                ww_mod[g] = fill(ww_mod[g], holes)
        gvr.wwme[0] = ww_mod
    # Modification of ww
    gvr.ww[0] = gvr.wwme[0].flatten()
    return gvr


def main():
    gvrname = "ep11-1.wwinp"
    beta = 5
    soft = 0.5
    fname: Path = Path("1.m")
    meshtally_number = 24
    degree = "No"
    option = "all"
    gvr = demo_gvr_soft(gvrname, beta, soft, fname, meshtally_number, degree, option)
    writeWWINP_cyl(gvr)


if __name__ == "__main__":
    main()
