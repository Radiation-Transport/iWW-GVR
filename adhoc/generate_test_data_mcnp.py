"""
The script generates MCNP models, which will be used to produce WW and meshtal files for the iww-gvr package testing.

The WW files will be produced using MCNP, advantg or other programs.
The test data will be produced by iww-gvr interactive mode using "soft", "soft2", "mit", "gvr" operation
and with various scenarios and parameters.

Both rectilinear and cylinder WW meshes are to be produced for "soft" and "mit" operations testing..
Both rectilinear and cylinder meshtal meshes are to be produced for GVR.

Number of voxels in the meshes (both WW and tally) should not be too large to keep testing times in reasonable range.

We are going also to implement benchmark tests. For these tests the meshes can be rather large,
to be equivalent to meshes from real problems.

"""
from typing import TextIO

import sys

import numpy as np

WATER = """
         1001.31c 6.32384E-004
         1002.31c 7.27325E-008
         8016.31c 3.15460E-004
         8017.31c 1.20167E-007
         8018.31c 6.48268E-007
""".rstrip()

CONTROLS = """
rand gen=2 seed=12653953025 stride=152917 hist=1
ctme  2
nps 1e+12
prdmp -90 -90  1 1 1e6
lost 10000 10000
mode n p
phys:n  20 1e-08  0
phys:p  20  4j
cut:n   j  1e-11 -0.5  -0.25  j
cut:p   j  1e-3  -0.5  -0.25  j
print
"""[
    1:-1
]


"""
c Generate weight window options:
wwg     4 0 0.4 4j 0
mesh    ref=391.10050705   -8.93454908   52.8 $ the left hfsr antenna tip
        geom=cyl
        axs=0 0 2000.0 vec=1680.4 -645.6 -1500.0
        origin=0 0 -1500.
        imesh=300. 1000. 1800.0      iints=10 70 25
        jmesh=920. 1300. 1500. 2100.  jints=30 38 15 30
        kmesh=0.11667  1.0            kints=30 1
wwge:n  1.e-6 0.1 20.
wwge:p  100.


c use weight window options (defaults + use external wwinp):
wwp:n   4j -1
c
c Generate weight window options:
wwg     6 0 0.4 4j 0
mesh    ref=641 0 67  $ standard source magnetic axis
        geom=cyl
        axs=0 0 1
        vec=1 0 0
        origin=0 0 -1400
        imesh=220 308 359 418 830 1000 1700
        iints=2   4   5   6   10  5    2
        jmesh=1000 1150.0 1351 1554 1809 1954 2167 3200 
        jints=2    4      20   20   10   10   5    2
        kmesh=0.055555555556 0.944444444444  1
        kints=20             1               20
wwge:n  1e-6 1e-3 0.01 0.1 1 10 20
wwge:p  1e-2 0.1 1 10 20 

"""


def create_cube_with_source_at_corner(
    *,
    cell_size: float = 5.0,
    cell_bins: int = 4,
    ww_bins: int = 4,
    mt_bins: int = 4,
    material: str = WATER,
    density: float = 0.0,
    stream: TextIO = sys.stdout,
) -> None:
    """
    Create a simple MCNP model with the neutron source in the corner and the target tally in the center.
    """
    print(f"test model: cube with source at corner", file=stream)
    print("c\nc   Parameters:\nc")
    print(f"c    cell_size={cell_size}", file=stream)
    print(f"c    cell_bins={cell_bins}", file=stream)
    print(f"c    density={density}", file=stream)
    print("c")
    cells_numbers = np.arange(1, cell_bins ** 3 + 1, dtype=np.int32).reshape(
        (cell_bins, cell_bins, cell_bins)
    )
    edge_size = cell_bins * cell_size
    half_edge = 0.5 * edge_size
    cell_bins_coords = np.linspace(-half_edge, half_edge, cell_bins + 1)
    surface_numbers = np.arange(1, cell_bins_coords.size * 3 + 1).reshape(
        (3, cell_bins_coords.size)
    )
    # p print(surface_numbers, file=stream)
    xsn, ysn, zsn = surface_numbers[:]

    print("c", "-" * 30, "  Cells", file=stream)
    for k, j, i in np.ndindex(cells_numbers.shape):
        surfaces = [xsn[k], -xsn[k + 1], ysn[j], -ysn[j + 1], zsn[i], -zsn[i + 1]]
        str_surfaces = " ".join(map(str, surfaces))
        print(
            cells_numbers[k, j, i],
            f"1 {density} {str_surfaces} imp:n=1 imp:p=1",
            file=stream,
        )

    outer_cell = cells_numbers[-1, -1, -1] + 1
    outer_surfaces = [-xsn[0], xsn[-1], -ysn[0], ysn[-1], -zsn[0], zsn[-1]]
    str_outer_surfaces = ":".join(map(str, outer_surfaces))
    #  print("c\nc Outer space\nc", file=stream)
    print(f"{outer_cell} 0 {str_outer_surfaces} imp:n=0 imp:p=0", file=stream)

    print(file=stream)
    print("c", "-" * 30, "  Surfaces", file=stream)
    for i, dimension in enumerate("xyz"):
        for j, coordinate in enumerate(cell_bins_coords):
            print(surface_numbers[i, j], f"p{dimension}", coordinate, file=stream)

    print(file=stream)
    print("c", "-" * 30, "  Control", file=stream)
    print("m1", material)
    print("sdef cel=1 par=1 x=d1 y=d1 z=d1")
    print(f"si1 H {cell_bins_coords[0]} {cell_bins_coords[1]}")
    print("sp1 D 0.0 1.0")
    print("fc4 --- Tally at the cube center")
    center_cell_index = cell_bins // 2
    center_cell = cells_numbers[center_cell_index, center_cell_index, center_cell_index]
    print(f"f4:n {center_cell}")
    print(CONTROLS)


def main():
    create_cube_with_source_at_corner(
        cell_size=10.0, cell_bins=3, density=-1.0, material=WATER
    )


if __name__ == "__main__":
    main()
