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
Author: Alvaro Cubi
"""
from __future__ import annotations

from copy import deepcopy
from typing import Optional, TYPE_CHECKING

import numpy as np
import pyvista as pv

if TYPE_CHECKING:
    from iww_gvr.weight_window import WW


class Plotter:
    def __init__(self, ww: WW):
        self.ww = ww
        self.mesh: Optional[pv.PolyData] = None
        self.bar_args = dict(interactive=True,  # Log bar for plots
                             title_font_size=20,
                             label_font_size=16,
                             shadow=True,
                             n_labels=11,  # Because n_colors of the plot is 10
                             italic=True,
                             fmt="%.e",
                             font_family="arial")
        self.args = dict(scalar_bar_args=self.bar_args,
                         cmap='jet',
                         log_scale=True,
                         n_colors=10, )
        return

    def plot(self, particle: str, energy: float):
        self.load_data(particle, energy)
        plotter = pv.Plotter()
        plotter.add_mesh(self.mesh, **self.args)
        plotter.show()
        return

    def plot_interactive_plane(self, particle: str, energy: float):
        self.load_data(particle, energy)
        plotter = pv.Plotter()
        plotter.add_mesh_clip_plane(self.mesh, **self.args)
        plotter.show()
        return

    def load_data(self, particle: str, energy: float):
        if self.mesh is None:
            self.load_mesh()
        data = self.ww.values[particle][energy]
        # The cyl grid has a different formatting
        if self.ww.coordinates == 'cyl':
            data = data.swapaxes(0, 1)
        data = data.flatten()
        self.mesh['data'] = data
        # Update the range of colors to a reasonable range
        exp_max = int(np.log10(data.max()))
        data_max = 10 ** exp_max
        data_min = 10 ** (exp_max - 10)  # data_min is 10 decades lower than data_max
        # noinspection PyTypeChecker
        self.args['clim'] = [data_min, data_max]
        return

    def load_mesh(self):
        if self.ww.coordinates == 'cart':
            # TODO: Implement cart coordinates for the PyVista plotter
            raise NotImplementedError
        mesh = pv.CylinderStructured(radius=self.ww.vector_i,
                                     height=self.ww.vector_j[-1],
                                     z_resolution=len(self.ww.vector_j),
                                     theta_resolution=len(self.ww.vector_k),
                                     direction=self.ww.director_vector)
        self.mesh = mesh
        return

    def plot_ratio(self, particle: str, log_scale=False):
        self.load_data_ratio(particle)
        mod_args = deepcopy(self.args)
        mod_args['log_scale'] = log_scale
        plotter = pv.Plotter()
        plotter.add_mesh(self.mesh, **mod_args)
        plotter.show()
        return

    def plot_ratio_interactive_plane(self, particle: str, log_scale=False):
        self.load_data_ratio(particle)
        plotter = pv.Plotter()
        mod_args = deepcopy(self.args)
        mod_args['log_scale'] = log_scale
        plotter.add_mesh_clip_plane(self.mesh, **mod_args)
        plotter.show()
        return

    def load_data_ratio(self, particle):
        if self.mesh is None:
            self.load_mesh()
        if self.ww.ratios is None:
            self.ww.calculate_ratios()
        data = self.ww.ratios_total_max[particle]
        # The cyl grid has a different formatting
        if self.ww.coordinates == 'cyl':
            data = data.swapaxes(0, 1)
        data = data.flatten()
        self.mesh['data'] = data
        # Update the range of colors so the min is 1, the zero values in the array mess up the range
        # noinspection PyTypeChecker
        self.args['clim'] = [1., data.max()]
        return


if __name__ == '__main__':
    from iww_gvr.weight_window import WW

    _ww = WW.read_from_ww_file('../tests/GVR_cubi_v1')
    _ww.plotter.plot('n', 50.0)
    _ww.plotter.plot_interactive_plane('n', 50.0,)
