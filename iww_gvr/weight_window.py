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

This file contains the WW class that contains all the information and methods relevant for WW analysis and
 manipulation.

TODO: The rotation of meshes, either cartesian or cylindrical is not yet implemented
"""
from copy import deepcopy
from typing import Dict, Optional, List

import numpy as np
from pyevtk.hl import gridToVTK
from tqdm import tqdm
from vtkmodules import vtkIOXML, vtkCommonDataModel, vtkCommonCore
from vtkmodules.util.numpy_support import numpy_to_vtk

from iww_gvr.plotter import Plotter
from iww_gvr.ww_parser import read_ww, write_ww, load_meshtally_file, read_meshtally, compose_b2_vector


class WW:
    """
    Represents a single weight-window mesh. Can be instantiated with WW.read_from_ww_file() or
    read_from_meshtally_file().

    Properties
    ----------
    particles: List[str]
        Either ['n'] or ['n', 'p']
    energies: dict
        self.energies['n'] => [1., 2., 14.1]
    vector_i, vector_j, vector_k: np.ndarray
        Vectors representing the binning of the mesh. Different behaviour depending on the coordinates. In cartesian
        coordinates the vectors represent the X, Y, Z points of the mesh. In cylindrical coordinates they represent
        the radial, axial and angular coordinates. The angular coordinates are in revolutions units, 0 to 1 being a
        360 degrees revolution. In cylindrical coordinates these vectors act as the cylinder is on the Z axis and its
        bottom center is place in (0, 0, 0), in order to have the real values the user should consider the origin and
        rotation values.
    info: str
        Information about the ww
    info_analyse: str
        Detailed information of the values like maximum ratio and minimum value

    Attributes
    ----------
    filename: str
        The name of the file or the name given to the weight-window
    coordinates: str
        'cart' or 'cyl' for cartesian or cylindrical coordinates
    values: dict
        This stores all the weight-window values. Many properties of the class are produced by exploration of this
        attribute. The data is accessed first by a key representing the particle and then by a key representing the
        energy bin. Example: self.values['n'][14.] => [[[1, 324, ...]]] k, j, i
    ratios: dict
        Same format as ww.values. Each value represents the maximum ratio found in the index in ww. values. The ratio
        is the value increase when going from or towards a neighbouring cell.
    ratios_total_max: dict
        Same as ratios but with a single array for particle, all energy bins considered

    Methods
    -------
    read_from_ww_file(filename)
    read_from_meshtally_file(filename, tally_id, maximum_splitting_ratio=5.)
    write_ww_file(filename)
    export_to_vtk(filename)
    calculate_ratios()
    apply_normalization(factor)
    apply_softening(factor)
    add_particle(norm, soft)
    remove_particle()
    filter_ratios(max_ratio)
    """
    # Public attributes
    filename: str
    plotter: Plotter
    coordinates: str  # Either 'cart' or 'cyl'
    values: Dict[str, Dict]  # self.values['n'][14.] => [[[1, 324, ...]]] k, j, i
    ratios: Optional[Dict[str, Dict]]  # Same format as values, shows the maximum ratio between voxels
    ratios_total_max: Optional[Dict]  # Same as ratios but with a single array for particle, all energy bins considered

    # Public properties
    particles: List[str]  # ['n'] or ['n', 'p']
    energies: Dict[str, np.ndarray]  # self.energies['n'] => [1., 2., 14.1]
    vector_i: np.ndarray  # The i vector constructed from the coarse vector and fine ints
    vector_j: np.ndarray  # The j vector constructed from the coarse vector and fine ints
    vector_k: np.ndarray  # The k vector constructed from the coarse vector and fine ints
    origin: np.ndarray  # The bottom left corner in cart. and center bottom in cyl
    info: str  # Information about the ww
    info_analyse: str  # Detailed information of the values like maximum ratio and minimum value

    # Private attributes
    # Vectors with the positions of the coarse mesh points
    _coarse_vector_i: List[float]
    _coarse_vector_j: List[float]
    _coarse_vector_k: List[float]

    # Lists with the amount of regular ints between each _coarse_vector points
    _fine_ints_list_i: List[float]
    _fine_ints_list_j: List[float]
    _fine_ints_list_k: List[float]

    # If the coordinates are cart the director vectors are None
    _director_1: Optional[List[float]]
    _director_2: Optional[List[float]]

    def __init__(self, data: Dict, filename: str):
        self.filename = filename
        self.plotter = Plotter(self)

        if data['nr'] == 10:
            self.coordinates = 'cart'
            self._director_1 = None
            self._director_2 = None
        else:
            self.coordinates = 'cyl'
            self._director_1 = data['director_1']
            self._director_2 = data['director_2']

        self._origin = np.array(data['origin'])
        self._coarse_vector_i, self._fine_ints_list_i = self._decompose_b2_vector(data['b2_vector_i'])
        self._coarse_vector_j, self._fine_ints_list_j = self._decompose_b2_vector(data['b2_vector_j'])
        self._coarse_vector_k, self._fine_ints_list_k = self._decompose_b2_vector(data['b2_vector_k'])

        # Build the values dictionary
        self.values: Dict[str, Dict[float, np.ndarray]] = {}
        i_ints = int(sum(self._fine_ints_list_i))
        j_ints = int(sum(self._fine_ints_list_j))
        k_ints = int(sum(self._fine_ints_list_k))
        particles = ['n']
        if len(data['values']) > 1:
            particles += ['p']
        for i, particle in enumerate(particles):
            self.values[particle] = {}
            vector = np.array(data['values'][i]).reshape([len(data['energies'][i]), k_ints, j_ints, i_ints])
            for j, energy in enumerate(data['energies'][i]):
                self.values[particle][energy] = vector[j]

        self.ratios = None  # Will not be populated until calculate_ratios() is called
        # Same as ratios but showing the maximum value across all the energies
        self.ratios_total_max = None  # {'n': [[[]]], 'p': [[[]]]}
        return

    @property
    def particles(self):
        """Either ['n'] or ['n', 'p']"""
        return list(self.values.keys())

    @property
    def energies(self):
        """self.energies['n'] => [1., 2., 14.1]"""
        e = {}
        for particle in self.particles:
            energies_par = list(self.values[particle].keys())
            energies_par.sort()
            e[particle] = energies_par
        return e

    @property
    def vector_i(self):
        vector = [self._coarse_vector_i[0]]
        for i in range(len(self._coarse_vector_i) - 1):
            vector += np.linspace(self._coarse_vector_i[i],
                                  self._coarse_vector_i[i + 1],
                                  int(self._fine_ints_list_i[0]) + 1).tolist()[1:]
        return np.array(vector)

    @property
    def vector_j(self):
        vector = [self._coarse_vector_j[0]]
        for i in range(len(self._coarse_vector_j) - 1):
            vector += np.linspace(self._coarse_vector_j[i],
                                  self._coarse_vector_j[i + 1],
                                  int(self._fine_ints_list_j[0]) + 1).tolist()[1:]
        return np.array(vector)

    @property
    def vector_k(self):
        vector = [self._coarse_vector_k[0]]
        for i in range(len(self._coarse_vector_k) - 1):
            vector += np.linspace(self._coarse_vector_k[i],
                                  self._coarse_vector_k[i + 1],
                                  int(self._fine_ints_list_k[0]) + 1).tolist()[1:]
        return np.array(vector)

    @property
    def origin(self):
        """Bottom left corner in cart, bottom center in cyl coordinates"""
        return self._origin

    @property
    def director_vector(self):
        """If cyl coordinates, director vector of the cyl axis"""
        if self.coordinates == 'cart':
            return None
        vector = self._director_1 + self.origin
        vector /= np.linalg.norm(vector)
        return vector

    @property
    def info(self):
        """Information about the ww"""
        text = f"{self.filename} weight window:\n"

        ints_i = int(sum(self._fine_ints_list_i))
        ints_j = int(sum(self._fine_ints_list_j))
        ints_k = int(sum(self._fine_ints_list_k))

        text += "               -----From----- -----To----- ---No. Bins---\n"
        text += f" I --> {' ':12}{self._coarse_vector_i[0]:.2f}{' ':9}{self._coarse_vector_i[-1]:.2f}{' ':9}{ints_i}\n"
        text += f" J --> {' ':12}{self._coarse_vector_j[0]:.2f}{' ':9}{self._coarse_vector_j[-1]:.2f}{' ':9}{ints_j}\n"
        text += f" K --> {' ':12}{self._coarse_vector_k[0]:.2f}{' ':9}{self._coarse_vector_k[-1]:.2f}{' ':9}{ints_k}\n"

        coordinates = 'cartesian' if self.coordinates == 'cart' else 'cylindrical'
        text += f"\n The mesh coordinates are {coordinates}\n"

        voxel_amount = ints_i * ints_j * ints_k
        text += f"\n The weight window contains {len(self.particles)} particle/s of {voxel_amount} voxels.\n"

        for particle in self.particles:
            text += f"\n Energy bins of particle {particle}:\n"
            text += f" {self.energies[particle]}"

        return text

    @property
    def info_analyse(self):
        """Detailed information of the values like maximum ratio and minimum value"""
        self.calculate_ratios()

        text = f"The following weight window has been analysed: {self.filename}\n"
        for particle in self.particles:
            text += f'---------------Par. {particle.capitalize()}---------\n'
            min_value = min([np.min(self.values[particle][energy]) for energy in self.energies[particle]])
            max_value = max([np.max(self.values[particle][energy]) for energy in self.energies[particle]])
            amount_positive_values = sum([self.values[particle][energy][self.values[particle][energy] > 0].shape[0] for
                                          energy in self.energies[particle]])
            amount_all_values = np.prod(self.values[particle][self.energies[particle][0]].shape) * len(
                self.energies[particle])
            percent_positive_values = 100. * amount_positive_values / amount_all_values
            max_ratio = self.ratios_total_max[particle].max()

            # Formatting of the information
            text += f'Min Value       : {min_value:.5f}\n' \
                    f'Max Value       : {max_value:.5f}\n' \
                    f'Max Ratio       : {max_ratio:.5f}\n' \
                    f'No.Bins > 0 [%] : {percent_positive_values:.2f}\n' \
                    f'Neg. Value      : {"YES" if min_value < 0 else "NO"}\n \n'

        text += f'Coordinates     : {self.coordinates}\n \n'
        if self.coordinates == 'cart':
            dimensions = [self.vector_i[1] - self.vector_i[0],
                          self.vector_j[1] - self.vector_j[0],
                          self.vector_k[1] - self.vector_k[0]]
            text += f"Voxel dimensions [x, y, z]: {dimensions[0]:.2f}, {dimensions[1]:.2f}, {dimensions[2]:.2f}\n"
            text += f"Voxel volume: {np.prod(dimensions):.2f} cm3\n"

        return text

    @staticmethod
    def read_from_ww_file(filename: str):
        """Returns a WW instance from a WW file"""
        data = read_ww(filename)
        return WW(data, filename=filename)

    @staticmethod
    def read_from_meshtally_file(filename: str, tally_id: int, maximum_splitting_ratio=5.):
        """Returns a WW instance in the form of a GVR generated from a meshtally filename. The meshtally used for the
        generation will be the one corresponding to tally_id."""
        meshtal_file = load_meshtally_file(filename)
        data = read_meshtally(meshtal_file, tally_id=tally_id)
        gvr = WW(data, filename=filename)

        # The values of this WW are simply the flux values, they need to be converted to weight values
        flux_max = 0
        for energy in gvr.energies['n']:
            flux_max = max(flux_max, gvr.values['n'][energy].max())
        for energy in gvr.energies['n']:
            # Van Vick / Andrew Davis / Magic algorithm
            gvr.values['n'][energy] /= flux_max
            gvr.values['n'][energy] *= (2/(maximum_splitting_ratio + 1))
        return gvr

    def write_ww_file(self, filename: str):
        """Writes the WW instance to a WW file with the name filename."""
        energies_flat = []
        for particle in self.particles:
            energies_flat.append(self.energies[particle])
        values_flat = []
        for particle in self.particles:
            values_particle_flat = []
            for energy in self.energies[particle]:
                values_particle_flat += self.values[particle][energy].flatten().tolist()
            values_flat.append(values_particle_flat)
        data = {'if_': 1,  # File type. Always 1, unused.
                'iv': 1,  # Time-dependent windows flag. 1/2 = no/yes
                'ni': len(self.particles),  # Number of particle types
                'nr': 10 if self.coordinates == 'cart' else 16,  # 10/16/16 = cartesian/cylindrical/spherical coord
                'probid': 'Generated with iww_gvr',  # Unused information of the file like datetime
                'ne': [len(x) for x in self.energies.values()],  # List of numbers, each number represents the amount
                # of energy bins for each particle
                'nfx': sum(self._fine_ints_list_i),  # Number of fine meshes in i
                'nfy': sum(self._fine_ints_list_j),  # Number of fine meshes in j
                'nfz': sum(self._fine_ints_list_k),  # Number of fine meshes in k
                'origin': self.origin,  # Corner of cartesian geometry or bottom center of cylindrical geometry
                'ncx': len(self._coarse_vector_i) - 1,  # Number of coarse meshes in i
                'ncy': len(self._coarse_vector_j) - 1,  # Number of coarse meshes in j
                'ncz': len(self._coarse_vector_k) - 1,  # Number of coarse meshes in k
                'b2_vector_i': compose_b2_vector(self._coarse_vector_i, self._fine_ints_list_i),
                'b2_vector_j': compose_b2_vector(self._coarse_vector_j, self._fine_ints_list_j),
                'b2_vector_k': compose_b2_vector(self._coarse_vector_k, self._fine_ints_list_k),
                'energies': energies_flat,  # [[energies p0]] or [[energies p0], [energies p1]]
                'values': values_flat,  # [[values p0]] or [[values p0], [values p1]]}
                }
        if self.coordinates == 'cyl':
            data['director_1'] = self._director_1
            data['director_2'] = self._director_2
        write_ww(filename, data)
        return

    def export_to_vtk(self, filename: str):
        """Writes the WW instance in VTK format with the name filename. Works for both types of coordinates.
        The implementation is different depending on the type of coordinates"""
        if self.coordinates == 'cyl':
            self._export_to_vtk_cyl(filename)
            return

        # VTK writing for cart coordinates
        # Create and fill the "cellData" dictionary
        cell_data: dict = {}  # Key: name of the data_array, Value: np.array of values in 3D dimensions [[[]]]
        for particle in self.particles:
            for energy in self.energies[particle]:
                array_name = f'{particle}_{energy:.3e}MeV'
                cell_data[array_name] = np.swapaxes(self.values[particle][energy], 0, 2)

        # If the ratios are calculated add them to the data arrays
        if self.ratios is not None:
            for particle in self.particles:
                for energy in self.energies[particle]:
                    array_name = f'Max_ratio_{particle}_{energy:.3e}MeV'
                    cell_data[array_name] = self.ratios[particle][energy]
                cell_data[f'Total_max_ratio_{particle}'] = self.ratios_total_max[particle]

        # Export to VTR format
        gridToVTK(f'./{filename}', self.vector_i, self.vector_j, self.vector_k, cellData=cell_data)
        return

    def _export_to_vtk_cyl(self, filename: str):
        """Does not work well if the Cylinder has only one angular int"""

        # Export to VTS format
        struct_grid = vtkCommonDataModel.vtkStructuredGrid()
        dimensions = [len(self.vector_k), len(self.vector_i), len(self.vector_j)]
        struct_grid.SetDimensions(*dimensions)
        vtk_points = vtkCommonCore.vtkPoints()
        vtk_points.Allocate(dimensions[0] * dimensions[1] * dimensions[2], 0)
        points = []
        bar = tqdm(unit=' J slices', desc=' Writing', total=len(self.vector_j))
        for j in self.vector_j:
            for i in self.vector_i:
                for k in self.vector_k:
                    theta = k * 2 * np.pi
                    point = [i * np.cos(theta), i * np.sin(theta), j]
                    points.append(point)
            bar.update()
        bar.close()

        # TODO: Rotate the cylinder according to the mesh axis
        points = np.array(points) + self.origin  # Displace the points to the origin

        for i in range(len(points)):
            vtk_points.InsertPoint(i, points[i])
        struct_grid.SetPoints(vtk_points)

        # Add the data arrays
        cell_data: vtkCommonDataModel.vtkCellData = struct_grid.GetCellData()
        for particle in self.particles:
            for energy in self.energies[particle]:
                # The array should have the order: k,i,j instead of  k, j, i
                array_name = f'{particle}_{energy:.3e}MeV'
                val_array = self.values[particle][energy]
                val_array = val_array.swapaxes(1, 2)  # Swap the i axis with the j axis => k,i,j
                # Now we swap the axes again so in the flatten operation we would be looping like j,i,k instead of k,i,j
                val_array = val_array.swapaxes(0, 2)  # Swap the k axis with the j axis => j,i,k
                val_array = val_array.flatten()
                vtk_array = numpy_to_vtk(val_array, deep=True, array_type=vtkCommonCore.VTK_DOUBLE)
                vtk_array.SetName(array_name)
                cell_data.AddArray(vtk_array)

        # If the ratios are calculated add them to the data arrays
        if self.ratios is not None:
            for particle in self.particles:
                for energy in self.energies[particle]:
                    # The array should have the order: k,i,j instead of  k, j, i
                    array_name = f'Max_ratio_{particle}_{energy:.3e}MeV'
                    val_array = self.ratios[particle][energy]
                    val_array = val_array.swapaxes(1, 2)  # Swap the i axis with the j axis => k,i,j
                    # Now we swap the axes again so in the flatten operation we would be looping like j,i,k
                    val_array = val_array.swapaxes(0, 2)  # Swap the k axis with the j axis => j,i,k
                    val_array = val_array.flatten()
                    vtk_array = numpy_to_vtk(val_array, deep=True, array_type=vtkCommonCore.VTK_DOUBLE)
                    vtk_array.SetName(array_name)
                    cell_data.AddArray(vtk_array)

                # The array should have the order: k,i,j instead of  k, j, i
                array_name = f'Total_max_ratio_{particle}'
                val_array = self.ratios_total_max[particle]
                val_array = val_array.swapaxes(1, 2)  # Swap the i axis with the j axis => k,i,j
                # Now we swap the axes again so in the flatten operation we would be looping like j,i,k
                val_array = val_array.swapaxes(0, 2)  # Swap the k axis with the j axis => j,i,k
                val_array = val_array.flatten()
                vtk_array = numpy_to_vtk(val_array, deep=True, array_type=vtkCommonCore.VTK_DOUBLE)
                vtk_array.SetName(array_name)
                cell_data.AddArray(vtk_array)

        writer = vtkIOXML.vtkXMLStructuredGridWriter()
        writer.SetInputData(struct_grid)
        writer.SetFileName(filename + '.vts')
        writer.Write()
        return

    def calculate_ratios(self):
        """Populates self.ratios the maximum ratio between voxels with the same shape as ww.values"""
        self.ratios = {}
        for particle in self.particles:
            self.ratios[particle] = {}
            for energy in self.energies[particle]:
                self.ratios[particle][energy] = self._calculate_array_ratio(self.values[particle][energy])

        # Calculate ratios_total_max
        self.ratios_total_max = {}
        for particle in self.particles:
            self.ratios_total_max[particle] = self.ratios[particle][self.energies[particle][0]]
            for energy in self.energies[particle]:
                self.ratios_total_max[particle] = np.maximum(self.ratios_total_max[particle],
                                                             self.ratios[particle][energy])
        return

    def apply_normalization(self, factor: float):
        """Multiplies all the weight-window values by a factor. It is applied to all particles and energy bins."""
        for particle in self.particles:
            for energy in self.energies[particle]:
                self.values[particle][energy] *= factor
        return

    def apply_softening(self, factor: float):
        """Elevates to the power of factor all the weight-window values. It is applied to all particles and energy
        bins."""
        for particle in self.particles:
            for energy in self.energies[particle]:
                self.values[particle][energy] **= factor
        self.calculate_ratios()
        return

    def add_particle(self, norm: float, soft: float):
        """If the WW is neutrons only, it adds a second particle type (photons) to the WW by copying the values of the
        first particle. It applies a normalization norm and a softening soft to the new particle only. It is
        recommended to reduce the values for the photons with respect to the neutrons by applying a norm like 1/10."""
        if len(self.particles) == 2:
            raise ValueError('The WW already has 2 particles')
        self.values['p'] = {}
        for energy in self.energies['n']:
            self.values['p'][energy] = deepcopy(self.values['n'][energy])
            self.values['p'][energy] *= norm
            self.values['p'][energy] **= soft
        return

    def remove_particle(self):
        """Removes the p particle if the WW has 2 particle types"""
        if len(self.particles) == 1:
            raise ValueError('The WW already has only 1 particle type')
        del self.values['p']
        return

    def filter_ratios(self, max_ratio: float):
        """Turns off the WW in the voxels that have a ratio greater than max_ratio."""
        if self.ratios is None:
            self.calculate_ratios()
        for particle in self.particles:
            for energy in self.energies[particle]:
                filter_ratio = np.where(self.ratios[particle][energy] > max_ratio)
                self.values[particle][energy][filter_ratio] = 0.0
        self.calculate_ratios()
        return

    @staticmethod
    def _calculate_array_ratio(array: np.ndarray):
        """
        Calculate the maximum ratio for every index of a 3D array.
        The ratio is calculated by the step between an index and its neighbours.
        """
        ratio = np.zeros_like(array)
        # The way we create an array of the maximum ratio found for every index relative to its contiguous
        # neighbours is by dividing the original array by other arrays where one of their indexes is shifted
        # by one in one direction.
        for axis in [0, 1, 2]:  # [k, j, i]
            for direction in [-1, 1]:
                # Add zeros in the specified axis and direction, this increase the size of the array but it will
                # be later sliced effectively moving all the values up or down.
                pad_width = [(0, 0), (0, 0), (0, 0)]
                pad_width[axis] = (1, 0) if direction == -1 else (0, 1)
                shifted_array = np.pad(array, pad_width, mode='constant', constant_values=0)[:, :, :]

                # Right now shifted_array has increased its size relative to array_p_e shape due to the
                # inclusion of zeros with the np.pad operation. We slice the array to make it recover the
                # expected shape. When slicing we should keep the zeros added in the operation and remove the
                # index that is furthest away from the zeros.
                if axis == 0 and direction == 1:
                    shifted_array = shifted_array[1:, :, :]
                elif axis == 0 and direction == -1:
                    shifted_array = shifted_array[:-1, :, :]
                elif axis == 1 and direction == 1:
                    shifted_array = shifted_array[:, 1:, :]
                    #  TODO: ensure that in the case of cyl coordinates the last and first index are
                    #   neighbours in the angular coordinate
                    # if self.coordinates == ' cyl':  # If it is a cyl the first and last angular bin are neighbours
                    #     shifted_array[:, -1, :] = array[:, 0, :]
                elif axis == 1 and direction == -1:
                    shifted_array = shifted_array[:, :-1, :]
                    # if self.coordinates == ' cyl':  # If it is a cyl the first and last angular bin are neighbours
                    #     shifted_array[:, 0, :] = array[:, -1, :]
                elif axis == 2 and direction == 1:
                    shifted_array = shifted_array[:, :, 1:]
                elif axis == 2 and direction == -1:
                    shifted_array = shifted_array[:, :, :-1]

                ratio_direction_axis = np.divide(array,
                                                 shifted_array,
                                                 out=np.zeros_like(array),  # In case of division by 0
                                                 where=shifted_array != 0)  # the ratio will be 0

                # To get the maximum step always, e.g. 0.5 => 2
                inverse_ratio_direction_axis = np.divide(np.ones_like(ratio_direction_axis),
                                                         ratio_direction_axis,
                                                         out=np.zeros_like(array),  # In case of division by 0
                                                         where=ratio_direction_axis != 0)  # the ratio will be 0
                ratio_direction_axis = np.maximum(ratio_direction_axis, inverse_ratio_direction_axis)

                # Modify the ratio array only where there is a new bigger value
                ratio = np.maximum(ratio, ratio_direction_axis)

        return ratio

    @staticmethod
    def _decompose_b2_vector(b2_vector):
        """
        Takes a b2_vector with format:
         [mesh_position, intervals, mesh_position, 1.0000, intervals, mesh_position, 1.0000, ...]
        And returns _coarse_vector and _fine_ints_list of WW.
        """
        # To extract coarse and fine vectors introduce an extra number in vector
        b2_vector.insert(1, 1.0000)
        coarse = [b2_vector[j] for j in range(len(b2_vector)) if j % 3 == 0]
        fine_ints = [int(b2_vector[j + 2]) for j in range(len(b2_vector)) if j % 3 == 0 and j + 2 < len(b2_vector)]
        return coarse, fine_ints

    def __repr__(self):
        return self.info


if __name__ == '__main__':
    _ww = WW.read_from_meshtally_file('../tests/meshtal_cyl.msh', 14)
    help(_ww)
