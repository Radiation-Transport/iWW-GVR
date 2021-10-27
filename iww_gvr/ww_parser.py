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

This file contains the reader and writer of WW files.
The reader returns a dictionary with the information extracted from the file.
The writer needs as input a dictionary of the same format as the reader output.
"""
from typing import Dict, List, TextIO

import numpy as np

from iww_gvr.meshtal_module import Meshtal


def _read_block2_vector(infile: TextIO, number_coarse_intervals) -> List[float]:
    # The amount of words to define the mesh vector depends on the amount of coarse meshes
    # [mesh_position, intervals, mesh_position, 1.0000, intervals, mesh_position, 1.0000, ...]
    vector = []
    words_in_vector = 3 * number_coarse_intervals + 1
    while len(vector) < words_in_vector:
        line = infile.readline()
        split = [float(word) for word in line.split()]
        vector += split
    return vector


def compose_b2_vector(coarse_vector, fine_ints_list):
    """
    The opposite of the method _decompose_b2_vector. It takes as input the coarse vector and fine ints list
     and returns a vector with the Block 2 format of the WW parser:
     [mesh_position, intervals, mesh_position, 1.0000, intervals, mesh_position, 1.0000, ...]
    """
    b2_vector = []
    for i in range(len(coarse_vector) - 1):
        b2_vector.append(coarse_vector[i])
        b2_vector.append(1.0)
        b2_vector.append(fine_ints_list[i])
    b2_vector.append(coarse_vector[-1])
    b2_vector.append(1)
    b2_vector.remove(1)
    return b2_vector


def _format_str_5_digits(number) -> str:
    """The amount of decimal digits depends on how big the number is, in total only 5 digits are allowed.
    If negative the - sign does not add to the count. If positive there is an empty space at the beginning.
    """
    result = f" {number:.8f}"
    if result[1] == '-':
        result = result[1:8]
    else:
        result = result[0:7]
    return result


def _format_str_6_digits_scientific(number) -> str:
    """Formatting for the ww values"""
    result = f"{number:.5E}"
    return result


def read_ww(filename: str) -> Dict:
    with open(filename, 'r') as infile:
        # Parse Block 1
        line = infile.readline()
        split = line.split()

        if_: int = int(split[0])  # File type. Always 1, unused.
        iv: int = int(split[1])  # Time-dependent windows flag. 1/2 = no/yes
        ni: int = int(split[2])  # Number of particle types
        nr: int = int(split[3])  # 10/16/16 = cartesian/cylindrical/spherical coordinates
        probid: str = ' '.join(split[4:])  # Unused information of the file like datetime

        line = infile.readline()
        # List of numbers, each number represents the amount of energy bins for each particle
        ne: List[int] = [int(word) for word in line.split()]

        line = infile.readline()
        split = [float(word) for word in line.split()]
        nfx: float = split[0]  # Number of fine meshes in i
        nfy: float = split[1]  # Number of fine meshes in j
        nfz: float = split[2]  # Number of fine meshes in k
        origin: List[float] = split[3:]  # Corner of cartesian geometry or bottom center of cylindrical geometry

        line = infile.readline()
        split = [float(word) for word in line.split()]
        ncx: float = split[0]  # Number of coarse meshes in i
        ncy: float = split[1]  # Number of coarse meshes in j
        ncz: float = split[2]  # Number of coarse meshes in k

        if nr == 10:  # Cartesian coordinates
            _nwg: float = split[3]  # Geometry type, here must be 1 == cartesian geometry
        elif nr == 16:  # Cylindrical coordinates
            director_1: List[float] = split[3:]
            line = infile.readline()
            split = [float(word) for word in line.split()]
            director_2: List[float] = split[0:3]
            _nwg: float = split[3]
        else:
            raise Exception('The nr value is not 10 or 16...')

        # Block 2
        # b2 vectors combine the info of the coarse and fine meshes, they give the position of each coarse mesh
        #  point and the amount of fine intervals between them. The b2 vector have the next format:
        #  [mesh_position, intervals, mesh_position, 1.0000, intervals, mesh_position, 1.0000, ...]
        b2_vector_i: List[float] = _read_block2_vector(infile, ncx)
        b2_vector_j: List[float] = _read_block2_vector(infile, ncy)
        b2_vector_k: List[float] = _read_block2_vector(infile, ncz)

        # Block 3
        energies: List[List[float]] = []
        values: List[List[float]] = []

        # Same process for each particle type
        for particle_index in range(ni):

            # Read the energy bins for this particle
            energies_current_particle: List[float] = []
            while len(energies_current_particle) < ne[particle_index]:
                line = infile.readline()
                split = [float(word) for word in line.split()]
                energies_current_particle += split

            # Read the values for this particle, all the energy bins at once
            words_in_particle = nfx * nfy * nfz * ne[particle_index]
            values_current_particle: List[float] = []
            while len(values_current_particle) < words_in_particle:
                line = infile.readline()
                split = [float(word) for word in line.split()]
                values_current_particle += split

            # Populate energies and values
            energies.append(energies_current_particle)
            values.append(values_current_particle)

    # Return all the data
    data = {
        'if_': if_,  # File type. Always 1, unused.
        'iv': iv,  # Time-dependent windows flag. 1/2 = no/yes
        'ni': ni,  # Number of particle types
        'nr': nr,  # 10/16/16 = cartesian/cylindrical/spherical coordinates
        'probid': probid,  # Unused information of the file like datetime
        'ne': ne,  # List of numbers, each number represents the amount of energy bins for each particle
        'nfx': nfx,  # Number of fine meshes in i
        'nfy': nfy,  # Number of fine meshes in j
        'nfz': nfz,  # Number of fine meshes in k
        'origin': origin,  # Corner of cartesian geometry or bottom center of cylindrical geometry
        'ncx': ncx,  # Number of coarse meshes in i
        'ncy': ncy,  # Number of coarse meshes in j
        'ncz': ncz,  # Number of coarse meshes in k
        'b2_vector_i': b2_vector_i,
        'b2_vector_j': b2_vector_j,
        'b2_vector_k': b2_vector_k,
        'energies': energies,  # [[energies p0]] or [[energies p0], [energies p1]]
        'values': values,  # [[values p0]] or [[values p0], [values p1]]
    }
    if nr == 16:  # Cylindrical coordinates
        data['director_1'] = director_1  # From origen to this: polar axis
        data['director_2'] = director_2  # From origen to this: azimuthal axis
    return data


def write_ww(filename: str, data: Dict):
    """data is a dictionary with the same format as the output of read_ww()"""
    with open(filename, 'w') as infile:
        # Block 1
        infile.write(f"         {data['if_']}         {data['iv']}         {data['ni']}        {data['nr']}")
        infile.write(f"                     {data['probid']} \n")

        energy_bins = [str(word) for word in data['ne']]
        infile.write(f"         {' '.join(energy_bins)}\n")

        inputs = [_format_str_5_digits(word) for word in [data['nfx'], data['nfy'], data['nfz'],
                                                          data['origin'][0], data['origin'][1], data['origin'][2]]]
        infile.write(f"  " + '      '.join(inputs) + '    \n')

        inputs = [_format_str_5_digits(word) for word in [data['ncx'], data['ncy'], data['ncz']]]
        if data['nr'] == 10:  # Cartesian coordinates
            inputs.append(_format_str_5_digits(1))
            infile.write(f"  " + '      '.join(inputs) + '    \n')
        else:  # Cylindrical coordinates
            inputs += [_format_str_5_digits(word) for word in [data['director_1'][0], data['director_1'][1],
                                                               data['director_1'][2]]]
            infile.write(f"  " + '      '.join(inputs) + '    \n')
            inputs = [_format_str_5_digits(word) for word in [data['director_2'][0], data['director_2'][1],
                                                              data['director_2'][2]]]
            inputs.append(_format_str_5_digits(2))
            infile.write(f"  " + '      '.join(inputs) + '    \n')

        # Block 2
        for vector in [data['b2_vector_i'], data['b2_vector_j'], data['b2_vector_k']]:
            # Make the vector a list of lists where each list have 6 items or less
            vector = [vector[i:i + 6] for i in range(0, len(vector), 6)]
            for words in vector:
                inputs = [_format_str_5_digits(word) for word in words]
                infile.write(f"  " + '      '.join(inputs) + '    \n')

        # Block 3
        for particle_index in range(len(data['energies'])):
            energy_bin = data['energies'][particle_index]
            # Make the vector a list of lists where each list have 6 items or less
            energy_bin = [energy_bin[i:i + 6] for i in range(0, len(energy_bin), 6)]
            for words in energy_bin:
                inputs = [_format_str_5_digits(word) for word in words]
                infile.write(f"  " + '      '.join(inputs) + '    \n')

            values = data['values'][particle_index]
            values = [values[i:i + 6] for i in range(0, len(values), 6)]
            for words in values:
                inputs = [_format_str_6_digits_scientific(word) for word in words]
                infile.write(f"  " + '  '.join(inputs) + '  \n')
    return


def load_meshtally_file(filename: str) -> Meshtal:
    mesh_file = Meshtal(filename)
    return mesh_file


def read_meshtally(mesh_file: Meshtal, tally_id: int) -> Dict:
    mesh = mesh_file.mesh[tally_id]
    mesh.readMCNP(mesh_file.f)

    nr = 10 if mesh.cart else 16
    # When building a GVR we consider that there are no regular intervals between points, that is all the vectors
    #  are coarse vectors with only one int between each point of the coarse vector. This means that the length of
    #  a coarse vector is equal to the total amount of fine ints of that coordinate.
    nfx = len(mesh.dims[3])
    nfy = len(mesh.dims[2])
    nfz = len(mesh.dims[1])
    origin = np.array([mesh.origin[3], mesh.origin[2], mesh.origin[1]])
    ncx = len(mesh.dims[3])
    ncy = len(mesh.dims[2])
    ncz = len(mesh.dims[1])
    b2_vector_i = compose_b2_vector(mesh.dims[3], [1 for _ in range(len(mesh.dims[3]) - 1)])
    b2_vector_j = compose_b2_vector(mesh.dims[2], [1 for _ in range(len(mesh.dims[2]) - 1)])
    b2_vector_k = compose_b2_vector(mesh.dims[1], [1 for _ in range(len(mesh.dims[1]) - 1)])
    values = [mesh.dat.flatten()]

    data = {'if_': 1,  # File type. Always 1, unused.
            'iv': 1,  # Time-dependent windows flag. 1/2 = no/yes
            'ni': 1,  # Only one particle type is considered when creating a GVR
            'nr': nr,  # 10/16/16 = cartesian/cylindrical/spherical coordinates
            'probid': mesh.meshtal.probid,  # Unused information of the file like datetime
            'ne': [1],  # List of numbers, each number represents the amount of energy bins for each particle
            'nfx': nfx,  # Number of fine meshes in i
            'nfy': nfy,  # Number of fine meshes in j
            'nfz': nfz,  # Number of fine meshes in k
            'origin': origin,  # Corner of cartesian geometry or bottom center of cylindrical geometry
            'ncx': ncx,  # Number of coarse meshes in i
            'ncy': ncy,  # Number of coarse meshes in j
            'ncz': ncz,  # Number of coarse meshes in k
            'b2_vector_i': b2_vector_i,
            'b2_vector_j': b2_vector_j,
            'b2_vector_k': b2_vector_k,
            'energies': [[100.]],  # Only one energy bin is considered when creating a GVR
            'values': values,  # [[values p0]] or [[values p0], [values p1]]
            }  # Only one energy bin is considered when creating a GVR

    if nr == 16:  # cylindrical coordinates
        if mesh.vec is None:  # For MCNP5 Meshtal there is no info about vec
            if mesh.axis[0] != 1:
                mesh.vec = [1, 0, 0]  # Default VEC in MCNP
            else:
                mesh.vec = [0, 1, 0]
        height = mesh.dims[2][-1]
        data['director_1'] = origin + mesh.axis * height  # Top center of the cylinder
        # Obtain the azimuthal vector
        radius = mesh.dims[3][-1]
        vector_1 = mesh.axis / np.linalg.norm(mesh.axis)
        if np.array_equal(vector_1, np.array([0, -1, 0])):
            vector_2 = [1, 0, 0]
        else:
            vector_2 = np.cross(vector_1, [0, -1, 0])
        data['director_2'] = origin + vector_2 * radius
    return data


if __name__ == '__main__':
    d = read_ww('../tests/GVR_cubi_v1')
    write_ww('delete', d)
    pass
