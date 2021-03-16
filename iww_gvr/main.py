"""
########################################################################################################
# Copyright 2019 F4E | European Joint Undertaking for ITER and the Development                         #
# of Fusion Energy (‘Fusion for Energy’). Licensed under the EUPL, Version 1.2                         #
# or - as soon they will be approved by the European Commission - subsequent versions                  #
# of the EUPL (the “Licence”). You may not use this work except in compliance                          #
# with the Licence. You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl.html       #
# Unless required by applicable law or agreed to in writing, software distributed                      #
# under the Licence is distributed on an “AS IS” basis, WITHOUT WARRANTIES                             #
# OR CONDITIONS OF ANY KIND, either express or implied. See the Licence permissions                    #
# and limitations under the Licence.                                                                   #
########################################################################################################
"""

from typing import Any, Dict, Iterable, List


import numpy as np
import math
from numpy import ndarray as array

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib as mpl

import os
import sys


from copy import (
    deepcopy,
)  # To copy a ww class instance in soft() and not modify the original ww
from pyevtk.hl import (
    gridToVTK,
)  # https://pypi.org/project/pyevtk/, pip install pyevtk for alvaro works like this
import vtk
from tqdm import tqdm  # Progress bars
from scipy import ndimage as nd  # Filling holes in zoneDEF 'auto'
from scipy.spatial.transform import Rotation as R  # For the rotation matrices in cyl

from itertools import chain  # used to desnest list

try:
    from vtkmodules.util import numpy_support
except ImportError:
    from vtk.util import numpy_support  # noqa   # old VTK library

from iww_gvr import meshtal_module  # It needs to be compatible with Python 3!!!


class ww_item:
    """
    Cartesian mesh.

    Attributes:

    d    : dictionary

    X    : X bins

    Y    : Y bins

    Z    : Z bins

    name : filename

    bins : No. of voxels per particle

    degree: covering model parameter
      - degree[0] = zoneID (flagging of cells within the domain)
      - degree[1] = factor (fraction of cell within the domain)

    vol  : average voxel volume [cm3]

    dim  : average voxel sizes [dX,dY,dZ] cm

    par  : number of weight mesh parts: 1 or 2

    min  : minimum values list        [for e in eb[0]@min|ParNo1,for e in eb[1]@min|ParNo2]

    max  : maximum values list        [for e in eb[0]@max|ParNo1,for e in eb[1]@max|ParNo2]

    eb   : ww energy bin list         [[]|ParNo1,[]|ParNo2]

    wwe  : ww set list                [[[]e_i,[]e_i+1, ...,[]e_n]|ParNo1,[[]e_i,[]e_i+1, ...,[]e_n]|ParNo2]

    wwme : ww set numpy array
        [[[k,j,i]e_i,[k,j,i]e_i+1, ....,,[k,j,i]e_n]|ParNo1,[[k,j,i]e_i,[k,j,i]e_i+1, ....,,[k,j,i]e_n ]|ParNo2]

    ratio: max ratio of voxel with nearby values (shape as self.wwme)

    coord: it states 'cart' for cartesian coordinates

    """

    def __init__(
        self,
        filename: str,
        X: array,
        Y: array,
        Z: array,
        nbins: int,
        No_Particle: int,
        ww1,
        eb1,
        ww2,
        eb2,
        dict: Dict[str, Any],
    ):

        self.d: Dict[str, Any] = dict

        self.coord: str = "cart"

        self.X: array = X
        self.Y: array = Y
        self.Z: array = Z

        self.name: str = filename

        assert isinstance(nbins, int), f"Parameter 'nbins' is to be integer, '{nbins}' is given."
        self.bins: int = nbins

        self.degree: List = []     # TODO dvp: clarify type of this list entries

        self.vol: float = (
            (self.X[-1] - self.X[0])
            * (self.Y[-1] - self.Y[0])
            * (self.Z[-1] - self.Z[0])
            / self.bins
        )

        self.dim: List[float] = [
            (self.X[-1] - self.X[0]) / (len(self.X) - 1),
            (self.Y[-1] - self.Y[0]) / (len(self.Y) - 1),
            (self.Z[-1] - self.Z[0]) / (len(self.Z) - 1),
        ]

        self.par: int = No_Particle

        if self.par > 1:
            ww = [ww1, ww2]
            self.eb: List[array] = [eb1, eb2]
        else:
            ww = [ww1]
            self.eb: List[array] = [eb1]

        self.wwe: List[array] = []
        values: List[array] = []

        for j in range(0, int(self.par)):
            for i in range(0, len(self.eb[j])):
                values.append(
                    ww[j][((i + 0) * int(self.bins)) : ((i + 1) * int(self.bins))]
                )

            self.wwe.append(values)
            values = []

        self.wwme: List[array] = []
        for j in range(0, int(self.par)):
            for i in range(0, len(self.eb[j])):
                vector = np.array(self.wwe[j][i])
                values.append(
                    vector.reshape(len(self.Z) - 1, len(self.Y) - 1, len(self.X) - 1)
                )
            self.wwme.append(values)
            values = []

        self.ratio: List[array] = []
        for j in range(0, self.par):
            for i in range(0, len(self.eb[j])):
                vector = np.ones(int(self.bins))
                values.append(
                    vector.reshape(len(self.Z) - 1, len(self.Y) - 1, len(self.X) - 1)
                )

            self.ratio.append(values)
            values = []

        self.min = []
        self.max = []
        for j in range(0, int(self.par)):
            self.min.append([min(l) for l in self.wwe[j]])
            self.max.append([max(l) for l in self.wwe[j]])

    def info(self) -> None:
        """
        Print the information of this weight window mesh
        """
        print("\n The following WW file has been analysed:  " + self.name + "\n")

        Part_A = "From"
        Part_B = "To"
        Part_C = "No. Bins"

        print(
            "{:>10}".format("")
            + "\t"
            + Part_A.center(15, "-")
            + "\t"
            + Part_B.center(15, "-")
            + "\t"
            + Part_C.center(15, "-")
        )

        line_X = (
            "{:>10}".format("X -->")
            + "\t"
            + "{:^15}".format("{:8.2f}".format(self.d["B2_Xo"]))
            + "\t"
            + "{:^15}".format("{:8.2f}".format(self.d["B2_Xf"]))
            + "\t"
            + "{:^15}".format(len(self.X) - 1)
        )
        line_Y = (
            "{:>10}".format("Y -->")
            + "\t"
            + "{:^15}".format("{:8.2f}".format(self.d["B2_Yo"]))
            + "\t"
            + "{:^15}".format("{:8.2f}".format(self.d["B2_Yf"]))
            + "\t"
            + "{:^15}".format(len(self.Y) - 1)
        )
        line_Z = (
            "{:>10}".format("Z -->")
            + "\t"
            + "{:^15}".format("{:8.2f}".format(self.d["B2_Zo"]))
            + "\t"
            + "{:^15}".format("{:8.2f}".format(self.d["B2_Zf"]))
            + "\t"
            + "{:^15}".format(len(self.Z) - 1)
        )

        print(line_X)
        print(line_Y)
        print(line_Z)
        print("\n The mesh coordinates are cartesian.")
        print(
            "\n The file contain {0} particle/s and {1} voxels!".format(
                self.par, self.bins * self.par
            )
        )

        if self.par == 1:
            print("\n ***** Particle No.1 ****")
            print(" Energy[{0}]: {1}\n\n".format(len(self.eb[0]), self.eb[0]))
        elif self.par == 2:
            print("\n ***** Particle No.1 ****")
            print(" Energy[{0}]: {1}".format(len(self.eb[0]), self.eb[0]))

            print("\n ***** Particle No.2 ****")
            print(" Energy[{0}]: {1}\n\n".format(len(self.eb[1]), self.eb[1]))

    def soft(self, zoneID: array) -> "ww_item":
        """
        Normalize, (hole-)fill and soften the weight window mesh.

        Args:
            zoneID:

        """
        flag = True
        while flag:
            soft = input(" Insert the softening factor: ")
            if ISnumber(soft):
                soft = float(soft)
                flag = False
            else:
                print(" Please insert a number!")
                flag = True

        flag = True
        while flag:
            norm = input(" Insert the normalization factor: ")
            if ISnumber(norm):
                norm = float(norm)
                flag = False
            else:
                print(" Please insert a number!")
                flag = True

        if self.par == 2:
            flag = True
            while flag:
                NoParticle = input(" Insert the No.Particle to modify[0,1]: ")
                if NoParticle == "0" or NoParticle == "1":
                    NoParticle = int(NoParticle)
                    flag = False
                else:
                    print(" Please insert 0 or 1!")
                    flag = True
        else:
            NoParticle = 0

        ww_out = []
        ww_mod = self.wwme  # TODO dvp: this creates modifiable alias to self.wwme, is it intended?

        if len(zoneID) > 1:  # Hole-filling
            for g in range(0, len(self.eb[NoParticle])):
                z = np.tile(zoneID, (len(self.Z) - 1, 1, 1))  # A 3d zoneID
                holes = ww_mod[NoParticle][g] == 0
                holes = holes * z
                holes = holes == 1
                ww_mod[NoParticle][g] = fill(ww_mod[NoParticle][g], holes)

            # if len(zoneID) >1 :                                         # Hole-filling
            #    value =[]
            #    ww_mod=self.wwme
            #    for g in range (0,len(self.eb[NoParticle])):
            #        for k in range (2, len(self.Z)-2):
            #            for j in range (2, len(self.Y)-2):
            #                for i in range (2, len(self.X)-2):
            #                    if self.wwme[NoParticle][g][k,j,i]==0:
            #                        if zoneID[j,i]==1:
            #                            BOX=[]
            #                            BOX=self.wwme[NoParticle][g][(k-2):(k+2),(j-2):(j+2),(i-2):(i+2)]
            #                            No_Values=np.size(np.nonzero(BOX))
            #
            #                            if No_Values>0:
            #                                del value
            #                                # *** To impose the average within the BOX matrix ***
            #                                value=np.sum(np.sum(BOX))/No_Values
            #                                # ** To impose the minimum value ***
            #                                # value = np.min(BOX[np.nonzero(BOX)])
            #                                ww_mod[NoParticle][g][k,j,i]=value

            # Modification of wwme
            for e in range(0, len(self.eb[NoParticle])):
                self.wwme[NoParticle][e] = np.power(ww_mod[NoParticle][e] * norm, soft)

            # Modification of wwe (denesting list wihth the itertools)
            for e in range(0, len(self.eb[NoParticle])):
                step1 = self.wwme[NoParticle][e].tolist()
                step2 = list(chain(*step1))
                self.wwe[NoParticle][e] = list(chain(*step2))

        else:
            # Modification of wwme
            for e in range(0, len(self.eb[NoParticle])):
                self.wwme[NoParticle][e] = np.power(
                    self.wwme[NoParticle][e] * norm, soft
                )

            # Modification of wwe (denesting list wihth the itertools)
            for e in range(0, len(self.eb[NoParticle])):
                step1 = self.wwme[NoParticle][e].tolist()
                step2 = list(chain(*step1))
                self.wwe[NoParticle][e] = list(chain(*step2))

        return self

    def add(self):
        """
            Add a ww set to the ww  (dvp: ?)
        """
        flag = True
        while flag:
            soft = input(" Insert the softening factor: ")
            if ISnumber(soft):
                soft = float(soft)
                flag = False
            else:
                print(" Please insert a number!")
                flag = True

        flag = True
        while flag:
            norm = input(" Insert the normalization factor: ")
            if ISnumber(norm):
                norm = float(norm)
                flag = False
            else:
                print(" Please insert a number!")
                flag = True

        value = []
        # Modification of wwme
        for e in range(0, len(self.eb[0])):
            value.append(np.power(self.wwme[0][e] * norm, soft))

        self.wwme.append(value)

        value = []
        # Modification of wwe (denesting list wihth the itertools)
        for e in range(0, len(self.eb[0])):
            step1 = self.wwme[1][e].tolist()
            step2 = list(chain(*step1))
            value.append(list(chain(*step2)))

        self.wwe.append(value)

        self.par = 2
        self.d["B1_ni"] = 2
        self.d["B2_par"] = False

        self.eb.append(self.eb[0])

        self.min.append([min(l) for l in self.wwe[1]])
        self.max.append([max(l) for l in self.wwe[1]])

        self.ratio.append(self.ratio[0])

        return self

    # Function to remove a ww set to the ww
    def remove(self):
        flag = True
        while flag:
            NoParticle = input(" Insert the weight windows set to remove[0,1]: ")
            if ISnumber(NoParticle):
                NoParticle = int(NoParticle)
                flag = False
            else:
                print(" Please insert a number!")
                flag = True

        del self.min[NoParticle]
        del self.max[NoParticle]

        del self.wwe[NoParticle]
        del self.eb[NoParticle]

        del self.wwm[NoParticle]
        del self.wwme[NoParticle]

        self.par = 1
        self.d["B1_ni"] = 1

        return self


#######################################
#### Functions for WW manipulation ####
#######################################

# Function to open and parse the ww file hence creating a ww class item
def load(InputFile):
    # To Import ww file

    # Line counter
    L_COUNTER = 0

    BLOCK_NO = 1  # This parameter define the BLOCK position in the file

    # Variables for BLOCK No.1
    B1_if = 0
    B1_iv = 0
    B1_ni = 0
    B1_nr = 0
    B1_ne = []

    # Variables for BLOCK No.2
    B2_nfx = 0
    B2_nfy = 0
    B2_nfz = 0
    B2_Xo = 0
    B2_Yo = 0
    B2_Zo = 0
    B2_Xf = 0
    B2_Yf = 0
    B2_Zf = 0
    B2_ncx = 0
    B2_ncy = 0
    B2_ncz = 0

    B2_X = False
    B2_Y = False
    B2_3 = False
    vec_coarse = [[], [], []]
    vec_fine = [[], [], []]

    # Variables for BLOCK No.3
    B3_eb1 = []
    ww1 = []

    B3_eb2 = []
    ww2 = []

    nlines = 0  # For the bar progress
    for line in open(InputFile).readlines():
        nlines += 1
    bar = tqdm(unit=" lines read", desc=" Reading", total=nlines)
    # Function to load WW
    with open(InputFile, "r") as infile:
        for line in infile:
            if BLOCK_NO == 1:
                # print ("Block No.1")
                if L_COUNTER == 0:
                    info = line[50:]
                    line = line[:50]

                    split = line.split()

                    B1_if = int(split[0])
                    B1_iv = int(split[1])
                    B1_ni = int(split[2])
                    B1_nr = int(split[3])

                    L_COUNTER += 1

                elif L_COUNTER == 1:

                    split = line.split()

                    for item in split:
                        B1_ne.append(item)

                    # Modification for only neutron WW created by Advantge
                    if (B1_ni == 2) and (int(B1_ne[1]) == 0):
                        B2_par = False
                        B1_ni = 1
                        B1_ne = B1_ne[:1]
                        # Modification for only Photon WW
                    elif (B1_ni == 2) and (int(B1_ne[0]) == 0):
                        B2_par = True  # ww2 set imposed in the ww1 position *** only photon case ***
                        B1_ni = 1  # As if only set was contained
                        B1_ne[0] = B1_ne[1]
                        del B1_ne[1]
                    else:
                        B2_par = False  # ww2 set imposed in the ww2 position

                    BLOCK_NO = 2  # TURN ON SWITCH FOR BLOCK No. 2

                    L_COUNTER = 0  # CLEAN L_COUNTER

            elif BLOCK_NO == 2:
                split = line.split()
                split = [float(i) for i in split]
                if L_COUNTER == 0:
                    # print ("Block No.2")

                    B2_nfx = int(float(split[0]))
                    B2_nfy = int(float(split[1]))
                    B2_nfz = int(float(split[2]))
                    B2_Xo = float(split[3])
                    B2_Yo = float(split[4])
                    B2_Zo = float(split[5])

                    L_COUNTER += 1

                elif L_COUNTER == 1:
                    # print(line)
                    B2_ncx = float(split[0])
                    B2_ncy = float(split[1])
                    B2_ncz = float(split[2])
                    L_COUNTER += 1
                    B2_X = True

                elif B2_X:
                    if len(split) == 4:
                        if vec_coarse[0] == []:
                            vec_coarse[0].append(split[0])
                        vec_fine[0].append(split[1])
                        vec_coarse[0].append(split[2])
                    if len(split) == 6:
                        if vec_coarse[0] == []:
                            vec_coarse[0].append(split[0])
                        vec_fine[0].append(split[1])
                        vec_coarse[0].append(split[2])
                        vec_fine[0].append(split[4])
                        vec_coarse[0].append(split[5])
                    if split[-1] == 1.0000 and len(split) != 6:
                        B2_X = False
                        B2_Y = True

                elif B2_Y:
                    if len(split) == 4:
                        if vec_coarse[1] == []:
                            vec_coarse[1].append(split[0])
                        vec_fine[1].append(split[1])
                        vec_coarse[1].append(split[2])
                    if len(split) == 6:
                        if vec_coarse[1] == []:
                            vec_coarse[1].append(split[0])
                        vec_fine[1].append(split[1])
                        vec_coarse[1].append(split[2])
                        vec_fine[1].append(split[4])
                        vec_coarse[1].append(split[5])
                    if split[-1] == 1.0000:
                        B2_Y = False
                        B2_Z = True

                elif B2_Z:
                    if len(split) == 4:
                        if vec_coarse[2] == []:
                            vec_coarse[2].append(split[0])
                        vec_fine[2].append(split[1])
                        vec_coarse[2].append(split[2])
                    if len(split) == 6:
                        if vec_coarse[2] == []:
                            vec_coarse[2].append(split[0])
                        vec_fine[2].append(split[1])
                        vec_coarse[2].append(split[2])
                        vec_fine[2].append(split[4])
                        vec_coarse[2].append(split[5])
                    if split[-1] == 1.0000:
                        B2_Z = False
                        BLOCK_NO = 3  # TURN ON SWITCH FOR BLOCK No. 3

                        nbins = float(B2_nfx) * float(B2_nfy) * float(B2_nfz)
                        X = [vec_coarse[0][0]]
                        for i in range(1, len(vec_coarse[0])):
                            X = np.concatenate(
                                (
                                    X,
                                    np.linspace(
                                        X[-1],
                                        vec_coarse[0][i],
                                        int(vec_fine[0][i - 1] + 1),
                                    )[1:],
                                )
                            )
                        B2_Xf = X[-1]

                        Y = [vec_coarse[1][0]]
                        for i in range(1, len(vec_coarse[1])):
                            Y = np.concatenate(
                                (
                                    Y,
                                    np.linspace(
                                        Y[-1],
                                        vec_coarse[1][i],
                                        int(vec_fine[1][i - 1] + 1),
                                    )[1:],
                                )
                            )
                        B2_Yf = Y[-1]

                        Z = [vec_coarse[2][0]]
                        for i in range(1, len(vec_coarse[2])):
                            Z = np.concatenate(
                                (
                                    Z,
                                    np.linspace(
                                        Z[-1],
                                        vec_coarse[2][i],
                                        int(vec_fine[2][i - 1] + 1),
                                    )[1:],
                                )
                            )
                        B2_Zf = Z[-1]

                        L_COUNTER = 0

            elif BLOCK_NO == 3:
                split = line.split()
                if L_COUNTER == 0:
                    for item in split:
                        B3_eb1.append(float(item))

                    if len(B3_eb1) == int(B1_ne[0]):
                        L_COUNTER += 1

                elif L_COUNTER == 1:
                    for item in split:
                        ww1.append(float(item))
                    if len(ww1) == (nbins * int(B1_ne[0])):
                        L_COUNTER += 1

                elif L_COUNTER == 2:
                    for item in split:
                        B3_eb2.append(float(item))

                    if len(B3_eb2) == int(B1_ne[1]):
                        L_COUNTER += 1

                elif L_COUNTER == 3:
                    for item in split:
                        ww2.append(float(item))
                    if len(ww2) == (nbins * int(B1_ne[1])):
                        L_COUNTER += 1

            bar.update()
    bar.close()
    # WW dictionary
    dict = {}

    dict = {
        "B1_if": B1_if,
        "B1_iv": B1_iv,
        "B1_ni": B1_ni,
        "B1_nr": B1_nr,
        "B1_ne": B1_ne,
        "B2_Xo": B2_Xo,
        "B2_Yo": B2_Yo,
        "B2_Zo": B2_Zo,
        "B2_Xf": B2_Xf,
        "B2_Yf": B2_Yf,
        "B2_Zf": B2_Zf,
        "B2_par": B2_par,
        "vec_coarse": vec_coarse,
        "vec_fine": vec_fine,
    }

    if B1_ni > 1:
        ww = ww_item(InputFile, X, Y, Z, nbins, B1_ni, ww1, B3_eb1, ww2, B3_eb2, dict)
    else:
        ww = ww_item(InputFile, X, Y, Z, nbins, B1_ni, ww1, B3_eb1, 0, 0, dict)
    return ww


# Function to export the ww set in VTK or to the MCNP input format
def write(wwdata, wwfiles, index):
    print(write_menu)

    ans, fname = answer_loop("write")

    if ans == "end":
        sys.exit("\n Thanks for using iWW-GVR! See you soon!")
    else:
        outputFile = wwfiles[index] + "_2write"
        ww = wwdata[index]

    if ans == "vtk":  # To export to VTK
        if ww.coord == "cyl":
            writeVTK_cyl(ww)
        else:
            # Create and fill the "cellData" dictionary
            dictName = []
            dictValue = []
            for j in range(0, int(ww.par)):
                for i in range(0, len(ww.eb[j])):
                    dictValue.append(ww.wwe[j][i])
                    if ww.d["B2_par"] == True:
                        dictName.append(
                            "WW_ParNo" + str(j + 2) + "_E=" + str(ww.eb[j][i]) + "_MeV"
                        )
                    else:
                        dictName.append(
                            "WW_ParNo" + str(j + 1) + "_E=" + str(ww.eb[j][i]) + "_MeV"
                        )

            if (
                max([e.max() for e in ww.ratio[0]]) != 1
            ):  # To be improved just to check if matrix is all one.
                for j in range(len(ww.ratio)):
                    maxratio = np.ones(np.shape(ww.ratio[0][0]))
                    oratio = np.array(ww.ratio)
                    for z in range(len(ww.Z) - 1):
                        for y in range(len(ww.Y) - 1):
                            for x in range(len(ww.X) - 1):
                                maxratio[z][y][x] = max(oratio[j, ..., z, y, x])
                    dictValue.append(maxratio)
                    dictName.append("[RATIO]_WW_ParNo" + str(1 + j))

                # for j in range (0,int(ww.par)):
                #    for e in range (0,len(ww.eb[j])):
                #        dictValue.append(ww.ratio[j][e])
                #        print(123)
                #        if ww.d['B2_par']==True:
                #            dictName.append('[RATIO]_WW_ParNo'+str(j+2)+'_E='+str(ww.eb[j][i])+'_MeV')
                #        else:
                #            dictName.append('[RATIO]_WW_ParNo'+str(j+1)+'_E='+str(ww.eb[j][i])+'_MeV')
                #
            for i in range(0, len(dictValue)):
                dictValue[i] = np.reshape(dictValue[i], int(ww.bins))

            zipDict = zip(dictName, dictValue)
            cellData = dict(zipDict)

            # Export to VTR format
            gridToVTK(
                "./" + wwfiles[index],
                np.array(ww.X),
                np.array(ww.Y),
                np.array(ww.Z),
                cellData,
            )
        print(" VTK... written!")

    elif ans == "wwinp":  # To export to WW MCNP format
        if ww.coord == "cyl":
            writeWWINP_cyl(ww)
        else:
            with open(outputFile, "w") as outfile:

                line_A = "{:>10}".format("{:.0f}".format(ww.d["B1_if"]))
                line_B = "{:>10}".format("{:.0f}".format(ww.d["B1_iv"]))
                if ww.d["B2_par"] == True:
                    line_C = "{:>10}".format("{:.0f}".format(ww.d["B1_ni"] + 1))
                else:
                    line_C = "{:>10}".format("{:.0f}".format(ww.d["B1_ni"]))
                line_D = "{:>10}".format("{:.0f}".format(ww.d["B1_nr"]))
                outfile.write(line_A + line_B + line_C + line_D + "\n")

                if (ww.par == 1) and ww.d["B2_par"] == False:
                    line_A = "{:>10}".format("{:.0f}".format(len(ww.eb[0])))
                    line_B = "{:>10}".format("")
                elif (ww.par == 1) and ww.d["B2_par"] == True:
                    line_A = "{:>10}".format("{:.0f}".format(0))
                    line_B = "{:>10}".format("{:.0f}".format(len(ww.eb[0])))
                else:
                    line_A = "{:>10}".format("{:.0f}".format(len(ww.eb[0])))
                    line_B = "{:>10}".format("{:.0f}".format(len(ww.eb[1])))

                outfile.write(line_A + line_B + "\n")

                line_A = "{:>9}".format("{:.2f}".format(len(ww.X) - 1))
                line_B = "{:>13}".format("{:.2f}".format(len(ww.Y) - 1))
                line_C = "{:>13}".format("{:.2f}".format(len(ww.Z) - 1))
                line_D = "{:>13}".format("{:.2f}".format(ww.d["B2_Xo"]))
                line_E = "{:>13}".format("{:.2f}".format(ww.d["B2_Yo"]))
                line_F = "{:>12}".format("{:.2f}".format(ww.d["B2_Zo"]))
                outfile.write(
                    line_A + line_B + line_C + line_D + line_E + line_F + "    \n"
                )

                line_A = "{:>9}".format("{:.2f}".format(len(ww.d["vec_coarse"][0]) - 1))
                line_B = "{:>13}".format(
                    "{:.2f}".format(len(ww.d["vec_coarse"][1]) - 1)
                )
                line_C = "{:>13}".format(
                    "{:.2f}".format(len(ww.d["vec_coarse"][2]) - 1)
                )
                line_D = "{:>13}".format("{:.2f}".format(1))
                outfile.write(line_A + line_B + line_C + line_D + "    \n")

                l = []
                for i in range(len(ww.d["vec_coarse"][0])):
                    l.append(ww.d["vec_coarse"][0][i])
                    try:
                        l.append(ww.d["vec_fine"][0][i])
                    except:
                        pass
                s = ""

                for i in l:
                    s = s + " {: 1.5e}".format(i)
                    if len(s.split()) == 6:
                        outfile.write(s + "\n")
                        s = " {: 1.5e}".format(1)
                    if len(s.split()) == 3:
                        s = s + " {: 1.5e}".format(1)
                outfile.write(s + "\n")

                l = []
                for i in range(len(ww.d["vec_coarse"][1])):
                    l.append(ww.d["vec_coarse"][1][i])
                    try:
                        l.append(ww.d["vec_fine"][1][i])
                    except:
                        pass
                s = ""
                for i in l:
                    s = s + " {: 1.5e}".format(i)
                    if len(s.split()) == 6:
                        outfile.write(s + "\n")
                        s = " {: 1.5e}".format(1)
                    if len(s.split()) == 3:
                        s = s + " {: 1.5e}".format(1)
                outfile.write(s + "\n")

                l = []
                for i in range(len(ww.d["vec_coarse"][2])):
                    l.append(ww.d["vec_coarse"][2][i])
                    try:
                        l.append(ww.d["vec_fine"][2][i])
                    except:
                        pass
                s = ""
                for i in l:
                    s = s + " {: 1.5e}".format(i)
                    if len(s.split()) == 6:
                        outfile.write(s + "\n")
                        s = " {: 1.5e}".format(1)
                    if len(s.split()) == 3:
                        s = s + " {: 1.5e}".format(1)
                outfile.write(s + "\n")

                # ********* Writing of WW values *********
                for par in range(0, ww.par):
                    jj = 0
                    value = 0
                    line_new = []
                    counter = 0

                    # Writing of energy bins
                    for item in ww.eb[par]:

                        if jj < 5:
                            line_new = "{:>13}".format("{:.4e}".format(item))
                            outfile.write(line_new)
                            jj = jj + 1
                            if counter == len(ww.eb[par]) - 1:
                                outfile.write("\n")
                                jj = 0
                                counter = 0
                        else:
                            line_new = "{:>13}".format("{:.4e}".format(item))
                            outfile.write(line_new)
                            outfile.write("\n")
                            jj = 0
                        counter = counter + 1

                    jj = 0
                    value = 0
                    line_new = []
                    counter = 0

                    # Writing of ww bins
                    for e in range(0, len(ww.eb[par])):
                        bar = tqdm(
                            unit=" lines",
                            desc=" Writing energy bin",
                            total=len(ww.wwe[par][e]),
                        )
                        for item in ww.wwe[par][e]:
                            bar.update()
                            # print(type(item))
                            # print(item)
                            value = float(item)

                            # To avoid MCNP fatal error dealing with WW bins with more than 2 exponentials
                            if value >= 1e100:
                                value = 9.99e99
                                print(
                                    "***warning: WW value >= 1e+100 reduced to 9.99e+99!***"
                                )

                            if jj < 5:
                                line_new = "{:>13}".format("{:.4e}".format(value))
                                outfile.write(line_new)
                                jj = jj + 1

                                if counter == len(ww.wwe[par][e]) - 1:
                                    outfile.write("\n")
                                    jj = 0
                                    counter = 0
                                else:
                                    counter = counter + 1

                            else:
                                line_new = "{:>13}".format("{:.4e}".format(value))
                                outfile.write(line_new)
                                outfile.write("\n")
                                jj = 0

                                if counter == len(ww.wwe[par][e]) - 1:
                                    # outfile.write('\n')
                                    jj = 0
                                    counter = 0
                                else:
                                    counter = counter + 1
                        bar.close()
        print(" File... written!")


# Function for analysing the WW file
def analyse(self, zoneID, factor):
    RATIO_EVA = []

    for p in range(0, self.par):

        ww_neg = []
        ww_neg_pos = []

        ww_noZERO = []
        posBins = []

        for e in range(0, len(self.eb[p])):

            posBins.append(
                len(np.argwhere(self.wwme[p][e] * zoneID > 0))
                / (len(self.wwme[p][e]) * len(np.argwhere(zoneID > 0)))
            )

            ww_neg_pos = np.where(self.wwme[p][e] < 0)

            for item in range(0, len(ww_neg_pos[0])):
                # Appending the position
                # ww_neg.append([ww_neg_pos[0][item],ww_neg_pos[1][item],ww_neg_pos[2][item]])

                # Appending only the number
                ww_neg.append(
                    self.wwme[p][e][
                        ww_neg_pos[0][item], ww_neg_pos[1][item], ww_neg_pos[2][item]
                    ]
                )

            bar = tqdm(unit=" Z slices", desc=" Ratios", total=int(len(self.Z)) - 1)
            # TODO To be improved
            extM = extend_matrix(self.wwme[p][e])
            for k in range(1, (int(len(self.Z)))):
                for j in range(1, (int(len(self.Y)))):
                    for i in range(1, (int(len(self.X)))):
                        if extM[k, j, i] > 0:
                            self.ratio[p][e][k - 1, j - 1, i - 1] = (
                                max(
                                    [
                                        extM[k + 1, j, i],
                                        extM[k - 1, j, i],
                                        extM[k, j + 1, i],
                                        extM[k, j - 1, i],
                                        extM[k, j, i + 1],
                                        extM[k, j, i - 1],
                                    ]
                                )
                                / extM[k, j, i]
                            )
                bar.update()
                # RATIO=[]
            bar.close()

            ### <<<<  it works but it is slower >>>>>>####
            # it   =  np.nditer(self.wwme[p][e], flags=['multi_index'])
            #
            # while not it.finished:
            #     k = it.multi_index[0]
            #     j = it.multi_index[1]
            #     i = it.multi_index[2]
            #
            #     try:
            #         if it[0]>0:
            #
            #             self.ratio[p][e][k,j,i] = max([self.wwme[p][e][k+1,j,i],self.wwme[p][e][k-1,j,i],self.wwme[p][e][k,j+1,i],self.wwme[p][e][k,j-1,i], self.wwme[p][e][k,j,i+1], self.wwme[p][e][k,j,i-1]])/self.wwme[p][e][k,j,i]
            #
            #
            #     except:
            #         self.ratio[p][e][k,j,i] = 1
            #
            #     it.iternext()

        # RATIO_MAX=max([e.max() for e in self.ratio[p]])
        # RATIO_MAX=[e.max() for e in self.ratio[p]]
        # RATIO_MAX=max(RATIO_MAX)
        # print(RATIO_MAX)
        RATIO_EVA.append(
            [[e.max() for e in self.ratio[p]], sum(posBins) / len(self.eb[p]), ww_neg]
        )

        ww_neg = []
        ww_neg_pos = []

        # To create the ratio histogram analysis
    for e in range(0, len(self.ratio[p])):
        font = {
            "family": "serif",
            "color": "darkred",
            "weight": "normal",
            "size": 16,
        }

        x_axis = np.logspace(0, 6, num=21)
        y_axis = []

        for i in range(0, len(x_axis) - 1):
            y_axis.append(
                len(
                    np.where(
                        np.logical_and(
                            self.ratio[p][e] >= x_axis[i],
                            self.ratio[p][e] < x_axis[i + 1],
                        )
                    )[0]
                )
            )

        fig, ax = plt.subplots()
        ax.bar(
            x_axis[:-1], y_axis, width=np.diff(x_axis), log=True, ec="k", align="edge"
        )
        ax.set_xscale("log")
        plt.xlabel("max ratio with nearby cells", fontdict=font)
        plt.ylabel("No.bins", fontdict=font)
        plt.title(
            self.name
            + "_ParNo."
            + str(p + 1)
            + "_"
            + "E"
            + "="
            + str(self.eb[p][e])
            + "MeV",
            fontdict=font,
        )

        # Tweak spacing to prevent clipping of ylabel
        plt.subplots_adjust(left=0.15)
        fig.savefig(
            self.name
            + "_ParNo."
            + str(p + 1)
            + "_"
            + "E"
            + "="
            + str(self.eb[p][e])
            + "MeV"
            + "_Ratio_Analysis.jpg"
        )

    # Print in screen the ww analysis
    print("\n The following WW file has been analysed:  " + self.name)

    for i in range(0, self.par):
        if len(RATIO_EVA[i][2]) > 0:
            flag = "YES"
            flag = (
                flag
                + "["
                + str(len(RATIO_EVA[i][2]))
                + "]"
                + " <<"
                + str(RATIO_EVA[i][2])
                + ">>"
            )
        else:
            flag = "NO"

        title = "Par.No " + str(i + 1)
        print("\n " + title.center(40, "-") + "\n")

        print(" Min Value       : " + str(self.min[i]))
        print(" Max Value       : " + str(self.max[i]))
        print(" Max Ratio       : " + str(RATIO_EVA[i][0]))
        print(" No.Bins>0 [%]   : " + "{:5.2f}".format(RATIO_EVA[i][1] * 100))
        print(" Neg.Value       : " + flag)
        print(" Voxel Dim[X,Y,Z]: " + str(self.dim) + " cm")
        print(" Voxel Vol[cm3]  : " + "{:1.4e}".format(self.vol) + "\n")
        print(" " + "-" * 40 + "\n")

    return self, RATIO_EVA


def zoneDEF(self, degree):
    """
    Define the zone of covering of the weight window mesh.

    Args:
        self:
        degree:

    Returns:

    """
    if degree == "all":  # The ww is completely contained in the model domain
        zoneID = []
        zoneID = np.ones((int(len(self.Y) - 1), int(len(self.X) - 1)))

        factor = 1

    elif degree == "auto":
        zoneID = np.zeros((int(len(self.Y) - 1), int(len(self.X) - 1)))
        non_zero_index = [
            i[3:] for i in np.argwhere(self.wwme)
        ]  # creates a list of indices of j,i if there is a non-zero value
        non_zero_index = np.unique(non_zero_index, axis=0)  # deletes repeated indices
        for j, i in non_zero_index:
            zoneID[j][i] = 1
        zoneID = nd.binary_fill_holes(zoneID).astype(
            int
        )  # Fills all the holes in zoneID
        # plot the ZONE
        cmap = plt.get_cmap("jet", 1064)
        # tell imshow about color map so that only set colors are used
        img = mpl.pyplot.imshow(zoneID, cmap=cmap, norm=colors.Normalize(0, 1))
        factor = sum(sum(zoneID)) / (int(len(self.Y) - 1) * int(len(self.X) - 1))
        print(" zoneID automatically generated!")

    else:  # The ww is partialy contained in the model domain

        # PR - Evaluation of the WW
        zoneID = []
        zoneID = np.zeros((int(len(self.Y) - 1), int(len(self.X) - 1)))

        for j in range(0, int(len(self.Y) - 1)):
            for i in range(0, int(len(self.X) - 1)):
                if np.absolute(np.arctan(self.Y[j] / self.X[i])) < (
                    degree / 2 / 180 * math.pi
                ):
                    zoneID[j, i] = 1

        # plot the ZONE
        cmap = plt.get_cmap("jet", 1064)

        # tell imshow about color map so that only set colors are used
        img = mpl.pyplot.imshow(zoneID, cmap=cmap, norm=colors.Normalize(0, 1))

        # problem here -->> mpl.pyplot.show(block = False)

        # Factor which evaluates the simulation domain
        factor = sum(sum(zoneID)) / (int(len(self.Y)) * int(len(self.X)))

    return zoneID, factor


# Function for "plot" option
def plot(self):
    while True:
        PLANE = input(" Select the plane[X,Y,Z] :")

        if PLANE == "X" or PLANE == "Y" or PLANE == "Z":
            break
        else:
            print(" not expected keyword")

    while True:

        if PLANE == "X":
            INFO_QUOTE = "[X-->" + str(self.X[0]) + ", " + str(self.X[-1]) + "] cm"
        elif PLANE == "Y":
            INFO_QUOTE = "[Y-->" + str(self.Y[0]) + ", " + str(self.Y[-1]) + "] cm"
        elif PLANE == "Z":
            INFO_QUOTE = "[Z-->" + str(self.Z[0]) + ", " + str(self.Z[-1]) + "] cm"

        while True:
            PLANE_QUOTE = input(" Select the quote " + INFO_QUOTE + ":")
            if ISnumber([PLANE_QUOTE]):
                PLANE_QUOTE = float(PLANE_QUOTE)
                break
            else:
                print(" Please insert a numerical value")

        if PLANE == "X":
            if self.X[0] <= PLANE_QUOTE <= self.X[-1]:
                break
            else:
                print(" Value outside the range")
        elif PLANE == "Y":
            if self.Y[0] <= PLANE_QUOTE <= self.Y[-1]:
                break
            else:
                print(" Value outside the range")
        elif PLANE == "Z":
            if self.Z[0] <= PLANE_QUOTE <= self.Z[-1]:
                break
            else:
                print(" Value outside the range")

    if self.par == 1:
        PAR_Select = 0

        if len(self.eb[PAR_Select]) > 1:
            while True:
                ENERGY = input(
                    " Select the energy [0, " + str(self.eb[PAR_Select][-1]) + "]MeV :"
                )

                if 0 < float(ENERGY) <= self.eb[PAR_Select][-1]:
                    break
                else:
                    print(" Value outside the range")
        else:
            ENERGY = self.eb[PAR_Select][0]

    else:
        while True:
            PAR_Select = input(" Select the particle [0,1] :")
            PAR_Select = int(PAR_Select)

            if PAR_Select == 0 or PAR_Select == 1:
                break
            else:
                print(" Wrong value")

        if len(self.eb[PAR_Select]) > 1:
            while True:
                ENERGY = input(
                    " Select the energy [0, " + str(self.eb[PAR_Select][-1]) + "]MeV :"
                )

                if 0 < float(ENERGY) <= self.eb[PAR_Select][-1]:
                    break
                else:
                    print(" Value outside the range")
        else:
            ENERGY = self.eb[PAR_Select][0]

    plot_ww(self, PAR_Select, PLANE, PLANE_QUOTE, ENERGY)


# Function to plot the ww with specific user sets
def plot_ww(self, PAR_Select, PLANE, PLANE_QUOTE, ENERGY):
    if PAR_Select > self.par:
        print(" Error --> No. particle outside range!")
        flag = False

    else:
        flag = True

        WW_P = []

        PE = closest(self.eb[PAR_Select], float(ENERGY))
        PE_STR = str(self.eb[PAR_Select][PE])

        # WW_P=np.array(self.ww[PAR_Select][int(PE*self.bins):int((PE+1)*self.bins)])
        WW_P = np.array(self.wwe[PAR_Select][PE])

        WW_P = WW_P.reshape(len(self.Z) - 1, len(self.Y) - 1, len(self.X) - 1)
        WW_P = WW_P[-1::-1, -1::-1, :]
        # WW_P=np.flipud(WW_P)

        if flag:
            fig = plt.figure()

            if PLANE == "X":
                if (PLANE_QUOTE > self.X[-1]) or (PLANE_QUOTE < self.X[0]):
                    print(" Error --> X dimension outside range!")
                    flag = False
                else:
                    DIM = closest(self.X, PLANE_QUOTE)
                    if DIM == (len(self.X) - 1):
                        DIM = DIM - 1
                    extent = [self.Y[0], self.Y[-1], self.Z[0], self.Z[-1]]
                    vals = WW_P[:, -1::-1, DIM]  # Slice in X
                    plt.xlabel("Y")
                    plt.ylabel("Z")

            elif PLANE == "Y":
                if (PLANE_QUOTE > self.Y[-1]) or (PLANE_QUOTE < self.Y[0]):
                    print(" Error --> Y dimension outside range!")
                    flag = False
                else:
                    DIM = closest(self.Y, PLANE_QUOTE)
                    DIM = len(self.Y) - DIM - 1
                    if DIM == (len(self.Y) - 1):
                        DIM = DIM - 1
                    extent = [self.X[0], self.X[-1], self.Z[0], self.Z[-1]]
                    vals = WW_P[:, DIM, :]  # Slice in Y
                    plt.xlabel("X")
                    plt.ylabel("Z")

            elif PLANE == "Z":
                if (PLANE_QUOTE > self.Z[-1]) or (PLANE_QUOTE < self.Z[0]):
                    print(" Error --> Z dimension outside range!")
                    flag = False
                else:
                    DIM = closest(self.Z, PLANE_QUOTE)
                    # print('1-->'+str(DIM))
                    DIM = len(self.Z) - DIM - 1
                    if DIM == (len(self.Z) - 1):
                        DIM = DIM - 1
                        # print('2-->'+str(DIM))
                    extent = [self.X[0], self.X[-1], self.Y[0], self.Y[-1]]

                    vals = WW_P[DIM, :, :]  # Slice in Z
                    plt.xlabel("X")
                    plt.ylabel("Y")

                    # Plotly solution to be completed
                    # za=[]
                    #
                    # for i in range(0,len(self.Y)-1):
                    #     za.append(vals[i,:])
                    #
                    # fig_plotly = go.Figure(data = go.Contour(z=za,y=self.Y,x=self.X,colorscale='Jet'))
                    # fig_plotly.write_html('first_figure.html', auto_open=True)
            if flag:

                # Using numpy features to find min and max
                f = np.array(vals.tolist())
                vmin = 0

                try:
                    vmin = np.min(f[np.nonzero(f)])
                except:
                    print(
                        " Plot is not exported as only zero are contained in the matrix."
                    )

                if vmin > 0:
                    vmax = np.max(f[np.nonzero(f)])
                    nColors = len(str(int(vmax / vmin))) * 2
                    if vmin > 0:
                        cax = mpl.pyplot.imshow(
                            vals,
                            cmap=plt.get_cmap("jet", nColors),
                            norm=colors.LogNorm(vmin, vmax),
                            extent=extent,
                        )
                    else:
                        cax = mpl.pyplot.imshow(
                            vals,
                            cmap=plt.get_cmap("jet", nColors),
                            vmin=vmin,
                            vmax=vmax,
                            extent=extent,
                        )

                    cbar = fig.colorbar(cax)

                    plt.title(self.name + "@" + PLANE + "=" + str(PLANE_QUOTE) + "cm")

                    if PAR_Select == 0 and self.d["B2_par"] == True:
                        fig.savefig(
                            self.name
                            + str(PAR_Select + 1)
                            + "_"
                            + PLANE
                            + "="
                            + str(PLANE_QUOTE)
                            + "cm"
                            + "_"
                            + "E"
                            + "="
                            + PE_STR
                            + "MeV"
                            + "_ParNo."
                            + ".jpg"
                        )
                    else:
                        fig.savefig(
                            self.name
                            + str(PAR_Select)
                            + "_"
                            + PLANE
                            + "="
                            + str(PLANE_QUOTE)
                            + "cm"
                            + "_"
                            + "E"
                            + "="
                            + PE_STR
                            + "MeV"
                            + ".jpg"
                        )
                        # problem here -->mpl.pyplot.show(block = False)
                    print(" Plot...Done!\n")


def gvr_soft(gvrname):
    """
    Creates a weight window file starting from the datafile imported and using the Global Variance Reduction.

    Args:
        gvrname:

    Returns:

    """
    while True:
        try:
            beta = float(input(" Insert the maximum splitting ratio (" "beta" "): "))
            break
        except:
            print(" Please insert a number!")

    while True:
        try:
            soft = float(input(" Insert the softening factor: "))
            break
        except:
            print(" Please insert a number!")

    while True:
        try:
            fname = input(" Enter the meshtally file to load:")
            with open(fname, "r") as infile:
                if "Mesh Tally Number" in infile.read():  # To improve
                    break
        except:
            print(" Not a valid file")

    mesh = meshtal_module.Meshtal(fname)
    print(" The following tallies have been found:\n")
    for key in mesh.mesh.keys():
        print(" ", key, " \n")
        # mesh.mesh[key].print_info() # Prints the information of all the tallies present

    while True:
        try:
            k = int(input(" Choose the mesh tally to use for the GVR: "))
            m = mesh.mesh[k]
            break
        except:
            print(" Not valid")
    if not m.cart:
        gvr = gvr_soft_cyl(m, mesh, gvrname)

    else:
        X = m.dims[3] + m.origin[3]
        Y = m.dims[2] + m.origin[2]
        Z = m.dims[1] + m.origin[1]
        vec_coarse = [X, Y, Z]
        vec_fine = [
            [1 for i in range(len(vec_coarse[0]) - 1)],
            [1 for i in range(len(vec_coarse[1]) - 1)],
            [1 for i in range(len(vec_coarse[2]) - 1)],
        ]
        nbins = (len(X) - 1) * (len(Y) - 1) * (len(Z) - 1)
        nPar = 1
        eb = [100]
        m.readMCNP(mesh.f)
        ww1 = m.dat.flatten()
        d = {
            "B1_if": 1,
            "B1_iv": 1,
            "B1_ne": ["1"],
            "B1_ni": 1,
            "B1_nr": 10,
            "B2_Xf": m.dims[3][-1] + m.origin[3],
            "B2_Xo": m.origin[3],
            "B2_Yf": m.dims[2][-1] + m.origin[2],
            "B2_Yo": m.origin[2],
            "B2_Zf": m.dims[1][-1] + m.origin[1],
            "B2_Zo": m.origin[1],
            "B2_par": False,
            "vec_coarse": vec_coarse,
            "vec_fine": vec_fine,
        }
        gvr = ww_item(
            gvrname, X, Y, Z, nbins, nPar, ww1, eb, [], [], d
        )  # A ww skeleton is generated with the info from the mesh file
    gvr.info()

    if gvr.coord == "cart":
        while True:
            degree = input(
                " Insert the toroidal coverage of the model [degree, all, auto] for Hole Filling approach or [No]: "
            )
            try:
                degree = float(degree)
                break
            except:
                pass
            if degree in ["all", "No", "auto"]:
                break
        if degree == "No":
            zoneID = []
        else:
            zoneID, factor = zoneDEF(gvr, degree)

        ww_inp = np.zeros(np.shape(gvr.wwme[0][0]))
        fluxinp_max = np.max(gvr.wwme[0][0])
        bar = tqdm(unit=" Z slices", desc=" GVR", total=len(gvr.Z) - 1)
        for k in range(0, len(gvr.Z) - 1):
            for j in range(0, len(gvr.Y) - 1):
                for i in range(0, len(gvr.X) - 1):
                    ww_inp[k, j, i] = np.power(
                        gvr.wwme[0][0][k, j, i] / fluxinp_max * (2 / (beta + 1)), soft
                    )  # Van Vick/A.Davis
            bar.update()
        bar.close()

        if (
            len(zoneID) > 0
        ):  # Hole filling (Super efficient but the values are not averaged or interpolated)
            z = np.tile(zoneID, (len(gvr.Z) - 1, 1, 1))  # A 3d zoneID
            holes = ww_inp == 0
            holes = (
                holes * z
            )  # Only the places that are inside the zoneID and have a value of zero in ww_inp will be filled
            holes = holes == 1
            ww_inp = fill(
                ww_inp, holes
            )  # The hole filling gives to the holes the same value as the nearest non-zero value

        gvr.wwme[0][0] = ww_inp
        # Modification of wwe (denesting list wihth the itertools)
        step1 = gvr.wwme[0][0].tolist()
        step2 = list(chain(*step1))
        gvr.wwe[0][0] = list(chain(*step2))

        # Update of characteristics of weight window set
        emax = [gvr.eb[0][-1]]
        del gvr.eb
        del gvr.min, gvr.max

        gvr.min = [min(gvr.wwe[0][0])]  # As there is only one bin in the gvr matrix
        gvr.max = [max(gvr.wwe[0][0])]  # As there is only one bin in the gvr matrix

        gvr.eb = [emax]

        gvr.par = 1
        gvr.d["B1_ni"] = 1
        gvr.d["B1_ne"] = 1

    else:  # cyl coord
        while True:
            degree = input(" Please insert Yes or No for Hole-filling approach: ")
            if degree == "Yes" or degree == "No":
                break
        if degree == "Yes":
            while True:
                option = input(
                    " Please select an option for the problem geometry [all, auto]: "
                )
                if option == "all" or option == "auto":
                    break
                else:
                    print(" Not a valid option")
            zoneID, factor = zoneDEF_cyl(gvr, option)
            gvr.zoneID = zoneID

        ww_inp = np.zeros(np.shape(gvr.wwme[0][0]))
        fluxinp_max = np.max(gvr.wwme[0][0])
        bar = tqdm(unit=" K slices", desc=" GVR", total=len(gvr.K) - 1)
        for k in range(0, len(gvr.K) - 1):
            for j in range(0, len(gvr.J) - 1):
                for i in range(0, len(gvr.I) - 1):
                    ww_inp[k, j, i] = np.power(
                        gvr.wwme[0][0][k, j, i] / fluxinp_max * (2 / (beta + 1)), soft
                    )  # Van Vick/A.Davis
            bar.update()
        bar.close()

        gvr.wwme[0][0] = ww_inp
        if degree == "Yes":  # Hole filling
            ww_mod = gvr.wwme[0]
            for g in range(int(gvr.d["B1_ne"][0])):
                z = []  # A 3d zoneID
                for i in range(len(zoneID)):
                    z.append([])
                    for m in range(len(gvr.J) - 1):
                        z[i].append(zoneID[i])
                while (
                    True
                ):  # For some unknown reason the fill operation when using cyl coordinates do not clear all the holes at once, several iterations are required
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


# THE WW SHOULD BE ANALYSED BEFORE USING. Function to mitigate long history par. by reducing the values that produce a too high ratio
def mitigate(ww, maxratio):
    for p in range(0, ww.par):
        for e in range(0, len(ww.eb[p])):
            extM = extend_matrix(ww.wwme[p][e])
            while len(np.argwhere(ww.ratio[p][e] >= maxratio)) > 0:
                idxs = np.argwhere(ww.ratio[p][e] >= maxratio)
                print(
                    " Found these many values with a ratio higher than the maximum: ",
                    len(idxs),
                )
                for idx in idxs:
                    neig = [
                        extM[idx[0] + 0, idx[1] + 1, idx[2] + 1],
                        extM[idx[0] + 2, idx[1] + 1, idx[2] + 1],
                        extM[idx[0] + 1, idx[1] + 0, idx[2] + 1],
                        extM[idx[0] + 1, idx[1] + 2, idx[2] + 1],
                        extM[idx[0] + 1, idx[1] + 1, idx[2] + 0],
                        extM[idx[0] + 1, idx[1] + 1, idx[2] + 2],
                    ]
                    neig = [x for x in neig if x > 0]
                    # ww.wwme[p][e][tuple(idx)] = (max(neig)+min(neig))/2.0
                    ww.wwme[p][e][tuple(idx)] = (max(neig)) / (
                        maxratio * 0.9
                    )  # Reduce the ww value to one right below the maxim ratio allowed
                    ww.ratio[p][e][tuple(idx)] = max(neig) / ww.wwme[p][e][tuple(idx)]

    for NoParticle in range(0, ww.par):
        # Modification of wwe (denesting list wihth the itertools)
        for e in range(0, len(ww.eb[NoParticle])):
            step1 = ww.wwme[NoParticle][e].tolist()
            step2 = list(chain(*step1))
            ww.wwe[NoParticle][e] = list(chain(*step2))


# Function for "operate" option
def operate():
    clear_screen()
    print(operate_menu)

    ans, fname = answer_loop("operate")

    if len(wwfiles) == 1:
        index = 0
    else:
        index = selectfile(wwfiles)

    if ans == "soft2":
        while True:
            HF = input(" Hole Filling approach [Yes, No]: ")
            if HF == "Yes" or HF == "No":
                break
        copy = deepcopy(wwdata[index])
        if HF == "Yes":
            if wwdata[index].coord == "cyl":
                while True:
                    option = input(
                        " Please select an option for the problem geometry [all, auto]: "
                    )
                    if option == "all" or option == "auto":
                        break
                    else:
                        print(" Not a valid option")
                zoneID, factor = zoneDEF_cyl(copy, option)

            elif wwdata[index].coord == "cart":
                if not wwdata[index].degree:
                    while True:
                        degree = input(
                            " Insert the toroidal coverage of the model [degree, all, auto]: "
                        )

                        if degree.isdigit():
                            degree = float(degree) / 2
                            break
                        elif degree == "all":
                            break
                        elif degree == "auto":
                            break
                        else:
                            print(" Please insert one of the options!")
                    zoneID, factor = zoneDEF(wwdata[index], degree)
                    wwdata[index].degree.append(zoneID)
                    wwdata[index].degree.append(factor)
                else:
                    zoneID = wwdata[index].degree[0]
                    factor = wwdata[index].degree[1]
        else:
            zoneID = []

        ww_out = copy.biased_soft(zoneID)
        ww_out.name = fname
        flag = True

    elif ans == "soft":
        if wwdata[index].coord == "cyl":
            while True:
                HF = input(" Hole Filling approach [Yes, No]: ")
                if HF == "Yes" or HF == "No":
                    break

            copy = deepcopy(wwdata[index])
            if HF == "Yes":
                while True:
                    option = input(
                        " Please select an option for the problem geometry [all, auto]: "
                    )
                    if option == "all" or option == "auto":
                        break
                    else:
                        print(" Not a valid option")
                zoneID, factor = zoneDEF_cyl(copy, option)
            else:
                zoneID = []

            ww_out = copy.soft_cyl(zoneID)
            ww_out.name = fname
            flag = True

        else:
            flag = True
            while flag:
                HF = input(" Hole Filling approach [Yes, No]: ")
                if HF == "Yes" or HF == "No":
                    flag = False
                    if HF == "Yes":
                        if not wwdata[index].degree:
                            while True:
                                degree = input(
                                    " Insert the toroidal coverage of the model [degree, all, auto]: "
                                )

                                if degree.isdigit():
                                    degree = float(degree) / 2
                                    break
                                elif degree == "all":
                                    break
                                elif degree == "auto":
                                    break
                                else:
                                    print(" Please insert one of the options!")
                            zoneID, factor = zoneDEF(wwdata[index], degree)
                            wwdata[index].degree.append(zoneID)
                            wwdata[index].degree.append(factor)
                        else:
                            zoneID = wwdata[index].degree[0]
                            factor = wwdata[index].degree[1]
                    else:
                        zoneID, factor = [], []
                else:
                    print(" Please insert Yes or No!")
                    flag = True

            copy = deepcopy(wwdata[index])

            ww_out = copy.soft(zoneID)
            ww_out.name = fname
            flag = True

        print(" Softening done!\n")

    elif ans == "add":
        if wwdata[index].par == 2:
            print(
                " Impossible to add a weight window set: already 2 sets are present!\n"
            )
            flag = False
            ww_out = None
        else:
            copy = deepcopy(wwdata[index])
            ww_out = copy.add()
            ww_out.name = fname
            flag = True
            print(" Additional weight window set incorporated!\n")

    elif ans == "mit":
        while True:
            maximum_ratio = input(" Insert maximum ratio allowed: ")
            try:
                maximum_ratio = float(maximum_ratio)
                break
            except:
                print(" Not a valid number")

        copy = deepcopy(wwdata[index])
        if copy.coord == "cyl":
            mitigate_cyl(copy, maximum_ratio)
        else:
            mitigate(copy, maximum_ratio)

        ww_out = copy
        ww_out.name = fname
        flag = True
        print(" Mitigation completed!")

    elif ans == "rem":

        if wwdata[index].par == 1:
            print(" Impossible to remove a weight window set: only 1 set is present!\n")
            flag = False
            ww_out = None
        else:
            copy = deepcopy(wwdata[index])
            ww_out = copy.remove()
            ww_out.name = fname
            flag = True
            print(" Weight window set removed!\n")

    return ww_out, fname, flag


#######################################
###### Menu Supporting Functions ######
#######################################

# Function which defines the main answer loop
def answer_loop(menu):
    pkeys = ["open", "info", "write", "analyse", "plot", "operate", "end", "gvr"]
    wkeys = ["wwinp", "vtk", "end"]
    okeys = ["add", "rem", "soft", "soft2", "mit", "end"]
    menulist = {"principal": pkeys, "write": wkeys, "operate": okeys}
    while True:
        ans = input(" enter action :")
        ans = ans.split()
        ans0 = None
        ans1 = None

        if len(ans) > 0:
            ans0 = ans[0]
        else:
            ans0 = ans
            # if len(ans) > 1 :
            #   ans1 = ans[1]

        if menu == "operate":
            if ans0 not in okeys:
                print(" bad operation keyword")
            elif ans0 == "end":
                sys.exit("\n Thanks for using iWW-GVR! See you soon!")
            else:
                ans1 = input(" Name of the result file:")
                break
        elif menu == "write":
            if ans0 not in wkeys:
                print(" bad operation keyword")
            else:
                # ans1 = input(' Name of the result file:')
                break
        elif ans0 in menulist[menu]:
            break
        else:
            print(" not expected keyword")

    return ans0, ans1


# Function to enter the filename
def enterfilename(name, wwfiles):
    if len(wwfiles) > 0:
        while True:
            fname = input(" enter ww file name:")
            if fname != "":
                if fname in wwfiles:
                    print(" {} already appended! Load a different one.".format(fname))
                elif os.path.isfile(fname) == False:
                    print(" {} not found".format(os.path.abspath(fname)))
                else:
                    break
    else:
        while True:
            fname = input(" enter ww file name:")
            if os.path.isfile(fname) == False:
                print(" {} not found".format(fname))
            else:
                break
    return fname


# Function to select the file
def selectfile(wwfiles):
    print("\n Input files present:")
    counter = 1
    for f in wwfiles:
        print(" - [{}] {}".format(str(counter), f))
        counter = counter + 1

    while True:
        index = input(" enter ww index:")
        if index.isdigit():
            index = int(index) - 1
            if len(wwfiles) > index >= 0:
                break
            else:
                print(" Value not valid")
        else:
            print(" Value not valid")
    return index


#######################################
####      Supporting Functions     ####
#######################################

def closest(list: List, Number):
    """
    Find the index of the closest list value of Number within the given list.

    Args:
        list:
        Number:

    Returns:

    """
    aux = []
    for valor in list:
        aux.append(abs(Number - valor))

    return aux.index(min(aux))


def ISnumber(lis: Iterable) -> bool:
    """
       Check if the list contains only numbers.
    """
    for x in lis:
        try:
            float(x)
            return True
        except ValueError:
            return False


# Function to clean the screen (e.g cls)
def clear_screen():
    if os.name == "nt":
        os.system("cls")
    else:
        os.system("clear")


# Function that returns the same matrix but covered in zeros. m1==m2[1:-1,1:-1] Works for 2d and 3d arrays
def extend_matrix(matrix):
    shape = ()
    for dim in matrix.shape:
        shape = shape + (dim + 2,)
    new_matrix = np.zeros(shape)
    try:
        new_matrix[1:-1, 1:-1] = matrix
    except:
        new_matrix[1:-1, 1:-1, 1:-1] = matrix
    return new_matrix


# Function that returns a matrix with its holes filled. Input(matrix, matrix with True where there is a hole to fix)
def fill(data, invalid):
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid')
    by the value of the nearest valid data cell
    Input:
        data:    numpy array of any dimension
        invalid: matrix with True where there  is a hole to fix
    Output:
        Return a filled array.
    """
    ind = nd.distance_transform_edt(
        invalid, return_distances=False, return_indices=True
    )
    return data[tuple(ind)]


#######################################
####     Cylindrical Functions     ####
#######################################


def load_cyl(InputFile):
    # LIMITATION: Only works with ww that have a single coarse mesh in each direction
    #  This limitation could be overcome.
    #  It may be that the original parser also has this limitation but ignores the fine meshes.
    # LIMITATION 2: Only for one particle weight windows

    # To Import ww file

    # Line counter
    L_COUNTER = 0

    BLOCK_NO = 1  # This parameter define the BLOCK position in the file

    # Variables for BLOCK No.1
    B1_if = 0
    B1_iv = 0
    B1_ni = 0
    B1_nr = 0
    B1_ne = []

    # Variables for BLOCK No.2
    B2_Iints = 0  # Number of bins in I (radius direction)
    B2_Jints = 0  # Number of bins in J (axis direction)
    B2_Kints = 0  # Number of bins in K (polar direction)
    B2_Icn = 0  # Number of coarse meshes in I
    B2_Jcn = 0  # Number of coarse meshes in J
    B2_Kcn = 0  # Number of coarse meshes in K
    B2_Origin = (0, 0, 0)  # Central bottom point of the cylinder
    B2_Final = (0, 0, 0)  # Central top point of the cylinder
    B2_ncx = 0  # Third line of B_2
    B2_ncy = 0  # Third line of B_2
    B2_ncz = 0  # Third line of B_2

    B2_X = False
    B2_Y = False
    B2_3 = False
    vec_coarse = [[], [], []]
    vec_fine = [[], [], []]

    # Variables for BLOCK No.3
    B3_eb1 = []
    B3_eb2 = []
    inValues1 = False
    inValues2 = False

    ww = [[], []]

    nlines = 0  # For the bar progress
    for line in open(InputFile).readlines():
        nlines += 1
    bar = tqdm(unit=" lines read", desc=" Reading file", total=nlines)
    # Function to load WW
    with open(InputFile, "r") as infile:

        for line in infile:
            if BLOCK_NO == 1:
                if L_COUNTER == 0:

                    info = line[50:]
                    line = line[:50]

                    split = line.split()

                    B1_if = int(split[0])
                    B1_iv = int(split[1])
                    B1_ni = int(split[2])
                    B1_nr = int(split[3])
                    L_COUNTER += 1

                elif L_COUNTER == 1:
                    split = line.split()

                    for item in split:
                        B1_ne.append(item)

                    # Modification for only neutron WW created by Advantge
                    if (B1_ni == 2) and (int(B1_ne[1]) == 0):
                        B2_par = False
                        B1_ni = 1
                        B1_ne = B1_ne[:1]
                        # Modification for only photon WW
                    elif (B1_ni == 2) and (int(B1_ne[0]) == 0):
                        B2_par = True  # ww2 set imposed in the ww1 position *** only photon case ***
                        B1_ni = 1  # As if only set was contained
                        B1_ne[0] = B1_ne[1]
                        del B1_ne[1]
                    else:
                        B2_par = False  # ww2 set imposed in the ww2 position

                    L_COUNTER += 1
                    BLOCK_NO = 2  # TURN ON SWITCH FOR BLOCK No. 2

            elif BLOCK_NO == 2:
                if L_COUNTER == 2:
                    split = line.split()
                    B2_Iints = int(float(split[0]))
                    B2_Jints = int(float(split[1]))
                    B2_Kints = int(float(split[2]))
                    B2_Origin = (float(split[3]), float(split[4]), float(split[5]))
                    L_COUNTER += 1

                elif L_COUNTER == 3:
                    split = line.split()
                    B2_Icn = int(float(split[0]))
                    B2_Jcn = int(float(split[1]))
                    B2_Kcn = int(float(split[2]))
                    B2_Final = (float(split[3]), float(split[4]), float(split[5]))
                    L_COUNTER += 1

                elif L_COUNTER == 4:
                    split = line.split()
                    B2_ncx = float(split[0])
                    B2_ncy = float(split[1])
                    B2_ncz = float(split[2])
                    vec = (
                        B2_ncx - B2_Origin[0],
                        B2_ncy - B2_Origin[1],
                        B2_ncz - B2_Origin[2],
                    )
                    vec = vec / np.linalg.norm(vec)
                    L_COUNTER += 1

                    B2_X = True

                    # Now we are be in the region of the coarse and fine meshes specification
                    # For now I will consider only files with one coarse mesh

                elif B2_X:
                    split = line.split()
                    split = [float(i) for i in split]
                    if len(split) == 4:
                        if vec_coarse[0] == []:
                            vec_coarse[0].append(split[0])
                        vec_fine[0].append(split[1])
                        vec_coarse[0].append(split[2])
                    if len(split) == 6:
                        if vec_coarse[0] == []:
                            vec_coarse[0].append(split[0])
                        vec_fine[0].append(split[1])
                        vec_coarse[0].append(split[2])
                        vec_fine[0].append(split[4])
                        vec_coarse[0].append(split[5])
                    if split[-1] == 1.0000 and len(split) != 6:
                        B2_X = False
                        B2_Y = True
                elif B2_Y:
                    split = line.split()
                    split = [float(i) for i in split]
                    if len(split) == 4:
                        if vec_coarse[1] == []:
                            vec_coarse[1].append(split[0])
                        vec_fine[1].append(split[1])
                        vec_coarse[1].append(split[2])
                    if len(split) == 6:
                        if vec_coarse[1] == []:
                            vec_coarse[1].append(split[0])
                        vec_fine[1].append(split[1])
                        vec_coarse[1].append(split[2])
                        vec_fine[1].append(split[4])
                        vec_coarse[1].append(split[5])
                    if split[-1] == 1.0000 and len(split) != 6:
                        B2_Y = False
                        B2_Z = True

                elif B2_Z:
                    split = line.split()
                    split = [float(i) for i in split]
                    if len(split) == 4:
                        if vec_coarse[2] == []:
                            vec_coarse[2].append(split[0])
                        vec_fine[2].append(split[1])
                        vec_coarse[2].append(split[2])
                    if len(split) == 6:
                        if vec_coarse[2] == []:
                            vec_coarse[2].append(split[0])
                        vec_fine[2].append(split[1])
                        vec_coarse[2].append(split[2])
                        vec_fine[2].append(split[4])
                        vec_coarse[2].append(split[5])
                    if split[-1] == 1.0000 and len(split) != 6:
                        B2_Z = False
                        BLOCK_NO = 3  # TURN ON SWITCH FOR BLOCK No. 3

                    nbins = float(B2_Iints) * float(B2_Jints) * float(B2_Kints)
                    X = [vec_coarse[0][0]]
                    for i in range(1, len(vec_coarse[0])):
                        X = np.concatenate(
                            (
                                X,
                                np.linspace(
                                    X[-1], vec_coarse[0][i], int(vec_fine[0][i - 1] + 1)
                                )[1:],
                            )
                        )
                    B2_Xf = X[-1]

                    Y = [vec_coarse[1][0]]
                    for i in range(1, len(vec_coarse[1])):
                        Y = np.concatenate(
                            (
                                Y,
                                np.linspace(
                                    Y[-1], vec_coarse[1][i], int(vec_fine[1][i - 1] + 1)
                                )[1:],
                            )
                        )
                    B2_Yf = Y[-1]

                    Z = [vec_coarse[2][0]]
                    for i in range(1, len(vec_coarse[2])):
                        Z = np.concatenate(
                            (
                                Z,
                                np.linspace(
                                    Z[-1], vec_coarse[2][i], int(vec_fine[2][i - 1] + 1)
                                )[1:],
                            )
                        )
                    B2_Zf = Z[-1]

            elif BLOCK_NO == 3:
                if inValues1:
                    for item in line.split():
                        ww[0].append(float(item))
                    if len(ww[0]) == nbins * int(B1_ne[0]):
                        inValues1 = False
                elif inValues2:
                    for item in line.split():
                        ww[1].append(float(item))
                else:
                    if len(ww[0]) == 0:
                        if B3_eb1 == []:
                            B3_eb1 = line.split()
                        else:
                            B3_eb1 = B3_eb1 + line.split()
                        if len(B3_eb1) == int(B1_ne[0]):
                            inValues1 = True
                    else:
                        if B3_eb2 == []:
                            B3_eb2 = line.split()
                        else:
                            B3_eb2 = B3_eb2 + line.split()
                        if len(B3_eb2) == int(B1_ne[1]):
                            inValues2 = True

            bar.update()

        bar.close()

    axis = np.array(B2_Final) - np.array(B2_Origin)
    rotAxis = np.cross(axis, [0, 0, 1])
    if np.linalg.norm(rotAxis) == 0:
        rotM = R.from_rotvec([0, 0, 0])
    else:
        rotAxis = rotAxis / np.linalg.norm(rotAxis)
        ang = -np.arccos(np.dot(axis, [0, 0, 1]))
        rotM = R.from_rotvec(rotAxis * ang)

    dict = {
        "B1_if": B1_if,
        "B1_iv": B1_iv,
        "B1_ni": B1_ni,
        "B1_nr": B1_nr,
        "B1_ne": B1_ne,
        "I": X,
        "J": Y,
        "K": Z,
        "B2_Origin": B2_Origin,
        "B2_Final": B2_Final,
        "B2_Iints": B2_Iints,
        "B2_Jints": B2_Jints,
        "B2_Kints": B2_Kints,
        "vec": vec,
        "rotM": rotM,
        "vec_coarse": vec_coarse,
        "vec_fine": vec_fine,
    }

    ww = ww_item_cyl(InputFile, X, Y, Z, nbins, ww, B3_eb1, B3_eb2, dict)
    return ww


class ww_item_cyl:
    """
    Cylinder mesh
    """

    def __init__(self, filename, I, J, K, nbins, ww, B3_eb1, B3_eb2, dict):
        # >>> ww_item properties
        # - self.d    : dictionary
        # - self.I    : I discretization vector cyl radius
        # - self.J    : J discretization vector cyl axis
        # - self.K    : K discretization vector cyl angle
        # - self.name : filename
        # - self.bins : No. of voxel per particle
        # - self.eb1   : ww energy bin list         [[]|ParNo1
        # --> @@@ Nested list of numpy array@@@
        # - self.wwme : ww set numpy array         [[[k,j,i]e_i,[k,j,i]e_i+1, ....,,[k,j,i]e_n]|ParNo1,[[k,j,i]e_i,[k,j,i]e_i+1, ....,,[k,j,i]e_n ]|ParNo2]
        # - self.ratio: max ratio of voxel with nearby values (shape as self.wwme)
        # - self.coord: states 'cyl' for cylindrical coordinates
        # - self.zoneID: zoneID
        # - self.par: number of particles

        self.zoneID = []
        self.d = dict

        self.coord = "cyl"

        self.I = I
        self.J = J
        self.K = K

        self.name = filename

        self.bins = nbins

        self.ww = ww
        self.eb1 = B3_eb1
        self.eb2 = B3_eb2
        self.eb12 = [self.eb1, self.eb2]

        if len(self.eb1) == 1:
            self.wwme = np.array(
                [
                    np.array(ww[0]).reshape(
                        self.d["B2_Kints"], self.d["B2_Jints"], self.d["B2_Iints"]
                    )
                ]
            )
        else:
            self.wwme = np.array(ww[0]).reshape(
                len(self.eb1),
                self.d["B2_Kints"],
                self.d["B2_Jints"],
                self.d["B2_Iints"],
            )

        if len(self.eb2) == 1:
            self.par = 2
            self.wwme = [
                self.wwme,
                np.array(
                    [
                        np.array(ww[1]).reshape(
                            self.d["B2_Kints"], self.d["B2_Jints"], self.d["B2_Iints"]
                        )
                    ]
                ),
            ]
        elif len(self.eb2) > 1:
            self.par = 2
            self.wwme = [
                self.wwme,
                np.array(ww[1]).reshape(
                    len(self.eb2),
                    self.d["B2_Kints"],
                    self.d["B2_Jints"],
                    self.d["B2_Iints"],
                ),
            ]
        else:
            self.wwme = [self.wwme, []]
            self.par = 1
        self.ratio = []
        for p in range(len(self.d["B1_ne"])):
            self.ratio.append([])
            for e in range(len(self.eb1)):
                self.ratio[p].append(
                    np.ones(
                        (self.d["B2_Kints"], self.d["B2_Jints"], self.d["B2_Iints"])
                    )
                )

    # Function to print the information of the ww
    def info(self):

        print("\n The following WW file has been analysed:  " + self.name + "\n")

        Part_A = "From"
        Part_B = "To"
        Part_C = "No. Bins"

        print(
            "{:>10}".format("")
            + "\t"
            + Part_A.center(15, "-")
            + "\t"
            + Part_B.center(15, "-")
            + "\t"
            + Part_C.center(15, "-")
        )

        line_X = (
            "{:>10}".format(" Radius -->")
            + "\t"
            + "{:^15}".format("{:8.2f}".format(self.I[0]))
            + "\t"
            + "{:^15}".format("{:8.2f}".format(self.I[-1]))
            + "\t"
            + "{:^15}".format(len(self.I) - 1)
        )
        line_Y = (
            "{:>10}".format(" Height -->")
            + "\t"
            + "{:^15}".format("{:8.2f}".format(self.J[0]))
            + "\t"
            + "{:^15}".format("{:8.2f}".format(self.J[-1]))
            + "\t"
            + "{:^15}".format(len(self.J) - 1)
        )
        line_Z = (
            "{:>10}".format(" Theta  -->")
            + "\t"
            + "{:^15}".format("{:8.2f}".format(self.K[0]))
            + "\t"
            + "{:^15}".format("{:8.2f}".format(self.K[-1]))
            + "\t"
            + "{:^15}".format(len(self.K) - 1)
        )

        print(line_X)
        print(line_Y)
        print(line_Z)
        print("\n The mesh coordinates are cylindrical.")
        print(
            "\n The file contain {0} particle/s and {1} voxels!".format(
                len(self.d["B1_ne"]), int(self.bins) * len(self.d["B1_ne"])
            )
        )

        print("\n ***** Particle No.1 ****")
        print(" Energy[{0}]: {1}\n\n".format(len(self.eb1), self.eb1))
        if len(self.d["B1_ne"]) == 2:
            print("\n ***** Particle No.2 ****")
            print(" Energy[{0}]: {1}\n\n".format(len(self.eb2), self.eb2))

            # This function is repeated for both types of coordinates

    def biased_soft(self, zoneID):
        """
        This function is analogue to the soft function but it modifies the value
        of the softening of each voxel depending on the square of the distance to
        a focus point.
        """
        if self.par == 2:
            flag = True
            while flag:
                NoParticle = input(" Insert the No.Particle to modify[0,1]: ")
                if NoParticle == "0" or NoParticle == "1":
                    NoParticle = int(NoParticle)
                    flag = False
                else:
                    print(" Please insert 0 or 1!")
                    flag = True
        else:
            NoParticle = int(0)
        flag = True
        while flag:
            soft_min = input(" Insert the minimum softening factor: ")
            if ISnumber(soft_min):
                soft_min = float(soft_min)
                flag = False
            else:
                print(" Please insert a number!")
                flag = True
        flag = True
        while flag:
            soft_max = input(" Insert the maximum softening factor: ")
            if ISnumber(soft_max):
                soft_max = float(soft_max)
                flag = False
            else:
                print(" Please insert a number!")
                flag = True
        flag = True
        while flag:
            focus = input(" Insert the focus point(ex: 2 32.2 12): ")
            try:
                focus = [float(x) for x in focus.split()]
                flag = False
            except:
                print(" Please insert the point!")
                flag = True

        dist_matrix = []  # This has the same shape as wwme k,j,i
        # Vectors containing the mid position for each interval are generated
        if self.coord == "cart":
            Z = [(self.Z[i] + self.Z[i + 1]) / 2 for i in range(len(self.Z) - 1)]
            Y = [(self.Y[i] + self.Y[i + 1]) / 2 for i in range(len(self.Y) - 1)]
            X = [(self.X[i] + self.X[i + 1]) / 2 for i in range(len(self.X) - 1)]
        if self.coord == "cyl":
            Z = [(self.K[i] + self.K[i + 1]) / 2 for i in range(len(self.K) - 1)]
            Y = [(self.J[i] + self.J[i + 1]) / 2 for i in range(len(self.J) - 1)]
            X = [(self.I[i] + self.I[i + 1]) / 2 for i in range(len(self.I) - 1)]

        # Now the matrix of distances is calculated. Each voxel has a distance value
        # from its center to the focus point.
        for k in Z:
            J_vec = []
            rotM = R.from_rotvec([0, 0, k])  # useful only for cyl coord
            for j in Y:
                I_vec = []
                for i in X:
                    if (
                        self.coord == "cart"
                    ):  # The distance in calculated differently depending if the coordinates are cart or cyl
                        dist = (
                            (i - focus[0]) ** 2
                            + (j - focus[1]) ** 2
                            + (k - focus[2]) ** 2
                        ) ** 0.5
                    else:
                        point = rotM.apply([i, 0, j])
                        dist = (
                            (point[0] - focus[0]) ** 2
                            + (point[1] - focus[1]) ** 2
                            + (point[2] - focus[2]) ** 2
                        ) ** 0.5
                    I_vec.append(dist)
                J_vec.append(I_vec)
            dist_matrix.append(J_vec)

        dist_matrix = np.array(dist_matrix)
        # Eq: A*dist**2 +B = soft
        dist_min = dist_matrix.min() ** 2
        dist_max = dist_matrix.max() ** 2

        B = (soft_min * dist_min - soft_max * dist_max) / (dist_min - dist_max)
        A = (soft_min - B) / dist_max
        # Now the distances matrix is used to build the softening factors
        # matrix. Every voxel has its specific softening factor
        softs = []  # Same shape as dist_matrix
        for k in dist_matrix:
            J_vec = []
            for j in k:
                I_vec = []
                for i in j:
                    factor = A * (i ** 2) + B
                    I_vec.append(factor)
                J_vec.append(I_vec)
            softs.append(J_vec)
        softs = np.array(softs)

        # The following part of the code is analogue to the original soft and
        # soft_cyl functions. It includes the hole filling which is different
        # depending on the type of coordinates.
        ww_mod = self.wwme
        if self.coord == "cart":
            if len(zoneID) > 1:  # Hole-filling
                for g in range(0, len(self.eb[NoParticle])):
                    z = np.tile(zoneID, (len(self.Z) - 1, 1, 1))  # A 3d zoneID
                    holes = ww_mod[NoParticle][g] == 0
                    holes = holes * z
                    holes = holes == 1
                    ww_mod[NoParticle][g] = fill(ww_mod[NoParticle][g], holes)
        if self.coord == "cyl":
            if len(zoneID) > 0:  # Hole-filling
                for g in range(int(self.d["B1_ne"][NoParticle])):
                    z = []  # A 3d zoneID
                    for i in range(len(zoneID)):
                        z.append([])
                        for m in range(len(self.J) - 1):
                            z[i].append(zoneID[i])
                    while (
                        True
                    ):  # For some unknown reason the fill operation when using cyl coordinates do not clear all the holes at once, several iterations are required
                        holes = ww_mod[g] == 0
                        holes = holes * z
                        holes = holes == 1
                        if len(np.argwhere(holes)) == 0:
                            break
                        ww_mod[g] = fill(ww_mod[g], holes)

        # Modification of wwme
        for e in range(len(self.wwme[NoParticle])):
            for k in range(len(self.wwme[NoParticle][e])):
                for j in range(len(self.wwme[NoParticle][e][k])):
                    for i in range(len(self.wwme[NoParticle][e][k][j])):
                        self.wwme[NoParticle][e][k][j][i] = np.power(
                            ww_mod[NoParticle][e][k][j][i], softs[k][j][i]
                        )

        # Modification of wwe (denesting list wihth the itertools)
        if self.coord == "cart":
            for e in range(len(self.wwme[NoParticle])):
                step1 = self.wwme[NoParticle][e].tolist()
                step2 = list(chain(*step1))
                self.wwe[NoParticle][e] = list(chain(*step2))
        if self.coord == "cyl":
            self.ww[NoParticle] = self.wwme[NoParticle].flatten()
        return self

    # Function that applies a soft and a norm factor. It also does a hole-filling if there is a zoneID
    def soft_cyl(self, zoneID):
        flag = True
        while flag:
            soft = input(" Insert the softening factor: ")
            if ISnumber(soft):
                soft = float(soft)
                flag = False
            else:
                print(" Please insert a number!")
                flag = True

        flag = True
        while flag:
            norm = input(" Insert the normalization factor: ")
            if ISnumber(norm):
                norm = float(norm)
                flag = False
            else:
                print(" Please insert a number!")
                flag = True

        for p in range(len(self.d["B1_ne"])):
            ww_mod = self.wwme[p]
            if len(zoneID) > 0:  # Hole-filling
                for g in range(int(self.d["B1_ne"][p])):
                    z = []  # A 3d zoneID
                    for i in range(len(zoneID)):
                        z.append([])
                        for m in range(len(self.J) - 1):
                            z[i].append(zoneID[i])
                    while (
                        True
                    ):  # For some unknown reason the fill operation when using cyl coordinates do not clear all the holes at once, several iterations are required
                        holes = ww_mod[g] == 0
                        holes = holes * z
                        holes = holes == 1
                        if len(np.argwhere(holes)) == 0:
                            break
                        ww_mod[g] = fill(ww_mod[g], holes)

            # Modification of wwme
            for e in range(len(ww_mod)):
                self.wwme[p][e] = np.power(ww_mod[e] * norm, soft)

            # Modification of ww
            self.ww[p] = self.wwme[p].flatten()
        return self

    def add(self) -> "ww_item_cyl":
        """
        Add a ww set to the ww

        Returns:
            self
        """
        flag = True
        while flag:
            soft = input(" Insert the softening factor: ")
            if ISnumber(soft):
                soft = float(soft)
                flag = False
            else:
                print(" Please insert a number!")
                flag = True

        flag = True
        while flag:
            norm = input(" Insert the normalization factor: ")
            if ISnumber(norm):
                norm = float(norm)
                flag = False
            else:
                print(" Please insert a number!")
                flag = True

        ww_mod = self.wwme[0]
        # Modification of wwme
        for e in range(len(ww_mod)):
            self.wwme[1].append(np.power(ww_mod[e] * norm, soft))
        self.wwme[1] = np.array(self.wwme[1])

        # Modification of ww
        self.ww[1] = np.array(self.wwme[1]).flatten()

        self.eb2 = self.eb1
        self.eb12 = [self.eb1, self.eb2]
        self.par = 2

        self.d["B1_ne"] = [self.d["B1_ne"][0], self.d["B1_ne"][0]]

        self.ratio.append(deepcopy(self.ratio[0]))

        return self

    def remove(self) -> "ww_item_cyl":
        """
        Remove a ww set to the ww.
        """
        flag = True
        while flag:
            NoParticle = input(" Insert the weight windows set to remove[0,1]: ")
            if ISnumber(NoParticle):
                NoParticle = int(NoParticle)
                flag = False
            else:
                print(" Please insert a number!")
                flag = True

        self.ww = [self.ww[NoParticle - 1], []]
        self.wwme = [self.wwme[NoParticle - 1], []]
        self.eb12 = [self.eb12[NoParticle - 1], []]
        self.eb1 = self.eb12[0]
        self.eb2 = []
        self.par = 1

        self.par = 1
        self.d["B1_ne"] = [self.d["B1_ne"][NoParticle - 1]]

        self.ratio = [self.ratio[NoParticle - 1]]

        return self


# On works
def plot_cyl(self, Jslice):
    z = []
    for k in range(len(self.K) - 1):
        m = []
        for i in range(len(self.I) - 1):
            m.append(float(self.wwme[0][k][Jslice][i]))
        z.append(m)
    fig = plt.figure()
    # ax = Axes3D(fig)
    rad = self.I
    azm = [a * 2 * np.pi for a in self.K]
    r, th = np.meshgrid(rad, azm)
    plt.subplot(projection="polar")
    plt.pcolormesh(th, r, z)
    plt.plot(azm, r, color="k", ls="none")
    plt.thetagrids([x * 360 / (2 * np.pi) for x in azm])
    plt.rgrids(rad)
    plt.grid()
    plt.show()


def zoneDEF_cyl(
    self, option, flag=False
):  # Set flag to True to produce a vtk file with the zoneID
    if option == "all":
        zoneID = np.ones((len(self.K) - 1, len(self.I) - 1))
    if option == "auto":
        zoneID = np.zeros((len(self.K) - 1, len(self.I) - 1))
        non_zero_index = []
        for p in range(len(self.d["B1_ne"])):
            nz = [
                (i[1], i[3]) for i in np.argwhere(self.wwme[p])
            ]  # creates a list of indices of j,i if there is a non-zero value
            non_zero_index = non_zero_index + nz
        non_zero_index = np.unique(non_zero_index, axis=0)  # deletes repeated indices

        for k, i in non_zero_index:
            zoneID[k][i] = 1
        for (
            Ivec
        ) in (
            zoneID
        ):  # Hole filling in I direction: it fills all space between 1s in the I direction of each K slice
            pos0, pos1 = np.argwhere(Ivec > 0)[0], np.argwhere(Ivec > 0)[-1]
            for i in range(pos0[0], pos1[0]):
                Ivec[i] = 1
    factor = sum(sum(zoneID)) / (
        int(self.wwme[0][0].shape[0] * self.wwme[0][0].shape[2])
    )
    if flag:
        copy = deepcopy(self)
        copy.name = copy.name + "ZONEID"
        z = []  # A 3d zoneID
        for i in range(len(zoneID)):
            z.append([])
            for m in range(len(self.J) - 1):
                z[i].append(zoneID[i])
        copy.wwme = []
        for p in range(len(self.d["B1_ne"])):
            copy.wwme.append([])
            for e in range(int(self.d["B1_ne"][p])):
                copy.wwme[p].append(z)
        writeVTK_cyl(copy)
    return zoneID, factor


# Modifies self.ratio having into account a cylindrical coordinate system, k[-1] is next to k[0].
def analyse_cyl(self, zoneID):
    RATIO_EVA = []

    z = []  # A 3d zoneID
    for i in range(len(zoneID)):
        z.append([])
        for m in range(len(self.J) - 1):
            z[i].append(zoneID[i])
    z = np.array(z)

    for p in range(len(self.d["B1_ne"])):
        posBins = []
        ww_neg = False
        for e in range(int(self.d["B1_ne"][p])):
            posBins.append(
                len(np.argwhere(self.wwme[p][e] * z > 0)) / len(np.argwhere(z > 0))
            )
            if np.argwhere(self.wwme[p] < 0).size > 0:
                ww_neg = True
            extM = extend_matrix_cyl(self.wwme[p][e])
            bar = tqdm(unit=" K slices", desc=" Analysing", total=len(self.K) - 1)
            for k in range(len(self.K) - 1):
                for j in range(1, len(self.J)):
                    for i in range(1, len(self.I)):
                        if extM[k, j, i] > 0:
                            neig = [
                                x
                                for x in (
                                    [
                                        extM[k + 1, j, i],
                                        extM[k - 1, j, i],
                                        extM[k, j + 1, i],
                                        extM[k, j - 1, i],
                                        extM[k, j, i + 1],
                                        extM[k, j, i - 1],
                                    ]
                                )
                                if x > 0
                            ]
                            if len(neig) > 0:
                                self.ratio[p][e][k, j - 1, i - 1] = (
                                    max(neig) / extM[k, j, i]
                                )
                bar.update()
            bar.close()

        RATIO_EVA.append(
            [
                [e.max() for e in self.ratio[p]],
                sum(posBins) / int(self.d["B1_ne"][p]),
                ww_neg,
            ]
        )

        # To create the ratio histogram analysis
        for e in range(int(self.d["B1_ne"][p])):
            font = {
                "family": "serif",
                "color": "darkred",
                "weight": "normal",
                "size": 16,
            }
            x_axis = np.logspace(0, 6, num=21)
            y_axis = []

            for i in range(0, len(x_axis) - 1):
                y_axis.append(
                    len(
                        np.where(
                            np.logical_and(
                                self.ratio[p][e] >= x_axis[i],
                                self.ratio[p][e] < x_axis[i + 1],
                            )
                        )[0]
                    )
                )

            fig, ax = plt.subplots()
            ax.bar(
                x_axis[:-1],
                y_axis,
                width=np.diff(x_axis),
                log=True,
                ec="k",
                align="edge",
            )
            ax.set_xscale("log")
            plt.xlabel("max ratio with nearby cells", fontdict=font)
            plt.ylabel("No.bins", fontdict=font)
            plt.title(
                self.name
                + "_ParNo."
                + str(p + 1)
                + "_"
                + "E"
                + "="
                + str(self.eb12[p][e])
                + "MeV",
                fontdict=font,
            )
            # Tweak spacing to prevent clipping of ylabel
            plt.subplots_adjust(left=0.15)
            fig.savefig(
                self.name
                + "_ParNo."
                + str(p + 1)
                + "_"
                + "E"
                + "="
                + str(self.eb12[p][e])
                + "MeV"
                + "_Ratio_Analysis.jpg"
            )

        # Print in screen the ww analysis
        print("\n The following WW file has been analysed:  " + self.name)
        title = "Par.No " + str(p + 1)
        print("\n " + title.center(40, "-") + "\n")
        print(" Min Value       : " + str(min(self.ww[p])))
        print(" Max Value       : " + str(max(self.ww[p])))
        print(" Max Ratio       : " + str(max(RATIO_EVA[p][0])))
        print(" No.Bins>0 [%]   : " + str(RATIO_EVA[p][1] * 100))
        print(" Neg.Value       : " + str(RATIO_EVA[p][2]))
        print(" " + "-" * 40 + "\n")


def extend_matrix_cyl(
    matrix,
):  # extends a matrix in both senses for J and I and only in one sense in K
    newM = np.zeros((matrix.shape[0] + 1, matrix.shape[1] + 2, matrix.shape[2] + 2))
    newM[
        :-1, 1:-1, 1:-1
    ] = matrix  # This way the K[-1] will be k[0] and not a zero value
    newM[-1] = newM[-2]
    return newM


def makeVTKarray(self):
    # The array should have the order: k,i,j
    arrays = []
    for p in range(len(self.d["B1_ne"])):
        for e in range(int(self.d["B1_ne"][p])):
            arr = []
            ww = self.wwme[p][e]
            for j in range(len(self.J) - 1):
                for i in range(len(self.I) - 1):
                    for k in range(len(self.K) - 1):
                        arr.append(ww[k][j][i])
            vtkarr = numpy_support.numpy_to_vtk(
                arr, deep=True, array_type=vtk.VTK_DOUBLE
            )
            vtkarr.SetName("Par " + str(p + 1) + " - " + str(self.eb12[p][e]) + "MeV")
            arrays.append(vtkarr)

    if (
        max([e.max() for e in self.ratio[0]]) != 1
    ):  # To be improved just to check if matrix is all one.
        for p in range(len(self.ratio)):
            maxratio = np.ones(np.shape(self.ratio[0][0]))
            oratio = np.array(self.ratio)
            for k in range(len(self.K) - 1):
                for j in range(len(self.J) - 1):
                    for i in range(len(self.I) - 1):
                        maxratio[k][j][i] = max(oratio[p, ..., k, j, i])
            arr = []
            for j in range(len(self.J) - 1):
                for i in range(len(self.I) - 1):
                    for k in range(len(self.K) - 1):
                        arr.append(maxratio[k][j][i])
            vtkarr = numpy_support.numpy_to_vtk(
                arr, deep=True, array_type=vtk.VTK_DOUBLE
            )
            vtkarr.SetName("Par " + str(p + 1) + " - " + "RATIO")
            arrays.append(vtkarr)
    return arrays


# Limitation: the vtk will be written as if the cylinder were had a vertical axis (0 0 1)
# If the cylinder has an arbitrary axis the plot will be correct but rotated to a vertical axis
# Surely to overcome this limitation a rotation matrix should be calculated from orgin and final points
# and this matrix should be applied every time a point is written to the vtk array.
def writeVTK_cyl(self):
    colors = vtk.vtkNamedColors()
    dims = [len(self.K), len(self.I), len(self.J)]
    # Create the structured grid.
    sgrid = vtk.vtkStructuredGrid()
    sgrid.SetDimensions(dims)
    points = vtk.vtkPoints()
    points.Allocate(dims[0] * dims[1] * dims[2])
    p = [0.0, 0.0, 0.0]
    vec = self.d["B2_Origin"]
    ps = []
    bar = tqdm(unit=" J slices", desc=" Writing", total=len(self.J))
    for j in self.J:
        for i in self.I:
            for k in self.K:
                theta = k * 2 * np.pi
                p[0] = i * np.cos(theta)
                p[1] = i * np.sin(theta)
                p[2] = j
                ps.append((i * np.cos(theta), i * np.sin(theta), j))

        bar.update()
    bar.close()
    ps = self.d["rotM"].apply(ps)
    ps = np.array(ps) + vec
    for p in range(len(ps)):
        points.InsertPoint(p, ps[p])

    sgrid.SetPoints(points)
    cellData = sgrid.GetCellData()

    for vtkarr in makeVTKarray(self):
        try:
            cellData.AddArray(vtkarr)
        except:
            print(" cannot input array")

    writer = vtk.vtkXMLStructuredGridWriter()
    writer.SetInputData(sgrid)
    writer.SetFileName(self.name + ".vts")
    writer.SetDataModeToAscii()
    writer.Update()


def writeWWINP_cyl(ww):
    outputFile = ww.name + "_2write"
    with open(outputFile, "w") as outfile:
        line_A = "{:>10}".format("{:.0f}".format(ww.d["B1_if"]))
        line_B = "{:>10}".format("{:.0f}".format(ww.d["B1_iv"]))
        line_C = "{:>10}".format("{:.0f}".format(ww.d["B1_ni"]))
        line_D = "{:>10}".format("{:.0f}".format(ww.d["B1_nr"]))
        outfile.write(line_A + line_B + line_C + line_D + "\n")

        line_A = "{:>10}".format("{:.0f}".format(float(ww.d["B1_ne"][0])))
        if len(ww.d["B1_ne"]) == 2:
            line_A = line_A + "{:>10}".format("{:.0f}".format(float(ww.d["B1_ne"][1])))
        outfile.write(line_A + "\n")

        line_A = "{:>9}".format("{:.2f}".format(len(ww.I) - 1))
        line_B = "{:>13}".format("{:.2f}".format(len(ww.J) - 1))
        line_C = "{:>13}".format("{:.2f}".format(len(ww.K) - 1))
        line_D = "{:>13}".format("{:.2f}".format(ww.d["B2_Origin"][0]))
        line_E = "{:>13}".format("{:.2f}".format(ww.d["B2_Origin"][1]))
        line_F = "{:>12}".format("{:.2f}".format(ww.d["B2_Origin"][2]))
        outfile.write(line_A + line_B + line_C + line_D + line_E + line_F + "    \n")

        line_A = "{:>9}".format("{:.2f}".format(len(ww.d["vec_coarse"][0]) - 1))
        line_B = "{:>13}".format("{:.2f}".format(len(ww.d["vec_coarse"][1]) - 1))
        line_C = "{:>13}".format("{:.2f}".format(len(ww.d["vec_coarse"][2]) - 1))
        line_D = "{:>13}".format("{:.2f}".format(ww.d["B2_Final"][0]))
        line_E = "{:>13}".format("{:.2f}".format(ww.d["B2_Final"][1]))
        line_F = "{:>12}".format("{:.2f}".format(ww.d["B2_Final"][2]))
        outfile.write(line_A + line_B + line_C + line_D + line_E + line_F + "    \n")

        line_A = "{:>9}".format(
            "{:.2f}".format(ww.d["B2_Origin"][0] + ww.I[-1] * ww.d["vec"][0])
        )
        line_B = "{:>13}".format(
            "{:.2f}".format(ww.d["B2_Origin"][1] + ww.I[-1] * ww.d["vec"][1])
        )
        line_C = "{:>13}".format(
            "{:.2f}".format(ww.d["B2_Origin"][2] + ww.I[-1] * ww.d["vec"][2])
        )
        line_D = "{:>13}".format("{:.2f}".format(2))
        outfile.write(line_A + line_B + line_C + line_D + "    \n")

        l = []
        for i in range(len(ww.d["vec_coarse"][0])):
            l.append(ww.d["vec_coarse"][0][i])
            try:
                l.append(ww.d["vec_fine"][0][i])
            except:
                pass
        s = ""

        for i in l:
            s = s + " {: 1.5e}".format(i)
            if len(s.split()) == 6:
                outfile.write(s + "\n")
                s = " {: 1.5e}".format(1)
            if len(s.split()) == 3:
                s = s + " {: 1.5e}".format(1)
        outfile.write(s + "\n")

        l = []
        for i in range(len(ww.d["vec_coarse"][1])):
            l.append(ww.d["vec_coarse"][1][i])
            try:
                l.append(ww.d["vec_fine"][1][i])
            except:
                pass
        s = ""

        for i in l:
            s = s + " {: 1.5e}".format(i)
            if len(s.split()) == 6:
                outfile.write(s + "\n")
                s = " {: 1.5e}".format(1)
            if len(s.split()) == 3:
                s = s + " {: 1.5e}".format(1)
        outfile.write(s + "\n")

        l = []
        for i in range(len(ww.d["vec_coarse"][2])):
            l.append(ww.d["vec_coarse"][2][i])
            try:
                l.append(ww.d["vec_fine"][2][i])
            except:
                pass
        s = ""

        for i in l:
            s = s + " {: 1.5e}".format(i)
            if len(s.split()) == 6:
                outfile.write(s + "\n")
                s = " {: 1.5e}".format(1)
            if len(s.split()) == 3:
                s = s + " {: 1.5e}".format(1)
        outfile.write(s + "\n")

        s = ""
        for e in ww.eb1:
            s = s + " {: 1.5e}".format(float(e))
            if len(s.split()) == 6:
                outfile.write(s + "    \n")
                s = ""
        if len(s) > 0:
            outfile.write(s + "    \n")

        bar = tqdm(unit=" words", desc=" Writing p1", total=len(ww.ww[0]))
        s = ""
        for i in ww.ww[0]:
            bar.update()
            value = float(i)

            # To avoid MCNP fatal error dealing with WW bins with more than 2 exponentials
            if value >= 1e100:
                value = 9.99e99
                print("***warning: WW value >= 1e+100 reduced to 9.99e+99!***")

            s = s + " {: 1.5e}".format(value)
            if len(s.split()) == 6:
                outfile.write(s + "    \n")
                s = ""
        bar.close()
        if len(s) > 0:
            outfile.write(s + "    \n")

        if len(ww.d["B1_ne"]) == 2:
            s = ""
            for e in ww.eb2:
                s = s + " {: 1.5e}".format(float(e))
                if len(s.split()) == 6:
                    outfile.write(s + "    \n")
                    s = ""
            if len(s) > 0:
                outfile.write(s + "    \n")

            bar = tqdm(unit=" words", desc=" Writing p2", total=len(ww.ww[1]))
            s = ""
            for i in ww.ww[1]:
                bar.update()
                value = float(i)

                # To avoid MCNP fatal error dealing with WW bins with more than 2 exponentials
                if value >= 1e100:
                    value = 9.99e99
                    print("***warning: WW value >= 1e+100 reduced to 9.99e+99!***")

                s = s + " {: 1.5e}".format(value)
                if len(s.split()) == 6:
                    outfile.write(s + "    \n")
                    s = ""
            bar.close()
            if len(s) > 0:
                outfile.write(s + "    \n")


def gvr_soft_cyl(m, mesh, gvrname):
    K = m.dims[1]
    J = m.dims[2]
    I = m.dims[3]
    vec_coarse = [
        I,
        J,
        K,
    ]  # We build the vec_coarse and fine as if there are no fine meshes, all meshes are coarse with only 1 interval
    vec_fine = [
        [1 for i in range(len(vec_coarse[0]) - 1)],
        [1 for i in range(len(vec_coarse[1]) - 1)],
        [1 for i in range(len(vec_coarse[2]) - 1)],
    ]
    nbins = (len(I) - 1) * (len(J) - 1) * (len(K) - 1)
    nPar = 1
    eb = [100]
    m.readMCNP(mesh.f)
    ww = [m.dat.flatten(), []]

    origin = [m.origin[3], m.origin[2], m.origin[1]]
    height = m.dims[2][-1]
    vec = m.axis * height
    B2_Final = origin + vec

    rotAxis = np.cross(vec, [0, 0, 1])
    if np.linalg.norm(rotAxis) == 0:
        rotM = R.from_rotvec([0, 0, 0])
    else:
        rotAxis = rotAxis / np.linalg.norm(rotAxis)
        ang = -np.arccos(np.dot(vec, [0, 0, 1]))
        rotM = R.from_rotvec(rotAxis * ang)

    B3_eb1 = m.dims[0][1:]
    B3_eb2 = []

    if m.vec is None:  # For MCNP5 Meshtals there is no info about vec
        if m.axis[0] != 1:
            m.vec = [1, 0, 0]  # Default VEC in MCNP
        else:
            m.vec = [0, 1, 0]

    dict = {
        "B1_if": 1,
        "B1_iv": 1,
        "B1_ni": 1,
        "B1_nr": 16,
        "B1_ne": ["1"],
        "I": I,
        "J": J,
        "K": K,
        "B2_Origin": origin,
        "B2_Final": B2_Final,
        "B2_Iints": len(m.dims[3]) - 1,
        "B2_Jints": len(m.dims[2]) - 1,
        "B2_Kints": len(m.dims[1]) - 1,
        "rotM": rotM,
        "vec": m.vec,
        "vec_coarse": vec_coarse,
        "vec_fine": vec_fine,
    }
    gvr = ww_item_cyl(
        gvrname, I, J, K, nbins, ww, B3_eb1, B3_eb2, dict
    )  # A ww skeleton is generated with the info from the mesh file
    return gvr


# Mitigates long histories problems
def mitigate_cyl(self, maxratio):
    for p in range(len(self.d["B1_ne"])):
        for e in range(int(self.d["B1_ne"][p])):
            iterations = 0
            while len(np.argwhere(self.ratio[p][e] >= maxratio)) > 0:
                iterations += 1
                idxs = np.argwhere(self.ratio[p][e] >= maxratio)
                print(
                    " Found these many values with a ratio higher than the maximum: ",
                    len(idxs),
                )
                for idx in idxs:
                    extM = extend_matrix_cyl(self.wwme[p][e])
                    neig = [
                        extM[idx[0] - 1, idx[1] + 1, idx[2] + 1],
                        extM[idx[0] + 1, idx[1] + 1, idx[2] + 1],
                        extM[idx[0] + 0, idx[1] + 0, idx[2] + 1],
                        extM[idx[0] + 0, idx[1] + 2, idx[2] + 1],
                        extM[idx[0] + 0, idx[1] + 1, idx[2] + 0],
                        extM[idx[0] + 0, idx[1] + 1, idx[2] + 2],
                    ]
                    neig = [x for x in neig if x > 0]
                    self.wwme[p][e][tuple(idx)] = (max(neig)) / (
                        maxratio * 0.9
                    )  # Reduce the ww value to one right below the maxim ratio allowed
                    self.ratio[p][e][tuple(idx)] = (
                        max(neig) / self.wwme[p][e][tuple(idx)]
                    )
                if iterations > 5:
                    print(" Maximum number of iterations reached")
                    print(
                        " The difference of value between certain voxels is too high to reduce the ratio below the specified max ratio"
                    )
                    break
        # Modification of ww
        self.ww[p] = self.wwme[p].flatten()


#######################################
####     Selections Menu           ####
#######################################

principal_menu = """
 ***********************************************
        Weight window manipulator and GVR
 ***********************************************

 * Open weight window file   (open)   
 * Display ww information    (info)   
 * Write                     (write)
 * Analyse                   (analyse)
 * Plot                      (plot)   
 * Weight window operation   (operate)
 * GVR generation            (gvr)
 * Exit                      (end)    
"""
write_menu = """               
 * Write to wwinp            (wwinp)
 * Write to VTK              (vtk)
 * Exit                      (end)
"""
operate_menu = """             
 * Softening and normalize   (soft)
 * Focused softening         (soft2)
 * Mitigate long histories   (mit)
 * Add                       (add)
 * Remove                    (rem)
 * Exit                      (end)
"""


#######################################
####     Operational Code          ####
#######################################


def main():
    clear_screen()
    print(principal_menu)
    ans, optname = answer_loop("principal")
    while True:
        # Load the ww
        if ans == "open":
            if len(wwfiles) == 0:
                fname = enterfilename(optname, wwfiles)
                with open(fname, "r") as infile:
                    split = infile.readline().split()
                if split[3] == "16":
                    ww_out = load_cyl(fname)
                else:
                    ww_out = load(fname)

                wwfiles.append(fname)
                wwdata.append(ww_out)
            else:
                fname = enterfilename(optname, wwfiles)
                with open(fname, "r") as infile:
                    split = infile.readline().split()
                if split[3] == "16":
                    ww_out = load_cyl(fname)
                else:
                    ww_out = load(wwfiles[-1])

                wwfiles.append(fname)
                wwdata.append(ww_out)
                # print(wwfiles)

        # Print ww information
        elif ans == "info":
            if len(wwfiles) == 0:
                print(" No weight window file")
            else:
                if len(wwfiles) == 1:
                    wwdata[-1].info()
                else:
                    index = selectfile(wwfiles)
                    wwdata[index].info()

            # print(wwfiles)

        # Plot the ww
        elif ans == "plot":
            if len(wwfiles) == 1:
                if wwdata[-1].coord == "cyl":
                    print(" Plot for cylindrical coordinates not available")
                else:
                    plot(wwdata[-1])
            else:
                index = selectfile(wwfiles)
                if wwdata[index].coord == "cyl":
                    print(" Plot for cylindrical coordinates not available")
                else:
                    plot(wwdata[index])

            # print(wwfiles)

        # Analyse the ww
        elif ans == "analyse":
            if len(wwfiles) == 1:
                index = 0
            else:
                index = selectfile(wwfiles)
            if wwdata[index].coord == "cart":
                if not wwdata[index].degree:
                    while True:
                        degree = input(
                            " Insert the toroidal coverage of the model [degree, all, auto]: "
                        )

                        if degree.isdigit():
                            degree = float(degree) / 2
                            break
                        elif degree == "all":
                            break
                        elif degree == "auto":
                            break
                        else:
                            print(" Please insert one of the options!")

                    zoneID, factor = zoneDEF(wwdata[index], degree)

                    wwdata[index].degree.append(zoneID)
                    wwdata[index].degree.append(factor)
                else:
                    zoneID = wwdata[index].degree[0]
                    factor = wwdata[index].degree[1]

                wwdata[index], RATIO_EVA = analyse(wwdata[index], zoneID, factor)
            # analyse(wwdata[index],zoneID, factor)

            elif wwdata[index].coord == "cyl":
                if wwdata[index].zoneID == []:
                    while True:
                        option = input(
                            " Please select an option for the problem geometry [all, auto]: "
                        )
                        if option == "all" or option == "auto":
                            break
                        else:
                            print(" Not a valid option")
                    zoneID, factor = zoneDEF_cyl(wwdata[index], option)
                    wwdata[index].zoneID = zoneID
                analyse_cyl(wwdata[index], wwdata[index].zoneID)

        # Export the ww
        elif ans == "write":
            clear_screen()
            if len(wwfiles) == 1:
                index = 0
            else:
                index = selectfile(wwfiles)
            write(wwdata, wwfiles, index)

        # Generate GVR
        elif ans == "gvr":
            if len(wwfiles) == 0:
                fname = input(" Please write the name of the resulting GVR: ")

                ww_out = gvr_soft(fname)

                wwfiles.append(fname)
                wwdata.append(ww_out)
            else:
                fname = input(" Please write the name of the resulting GVR: ")
                wwfiles.append(fname)

                ww_out = gvr_soft(fname)
                wwdata.append(ww_out)

        # Modify the ww
        elif ans == "operate":

            ww_out, fname, flag = operate()

            if flag:
                wwfiles.append(fname)
                wwdata.append(ww_out)

        elif ans == "end":
            sys.exit("\n Thanks for using iWW-GVR! See you soon!")
        else:
            break

        if ans == "operate" or ans == "write":
            print(principal_menu)
        ans, optname = answer_loop("principal")
        clear_screen()
        print(principal_menu)


wwfiles = []  # list storing filename
wwdata = []  # list storing weight window class objects

if __name__ == "__main__":
    main()
