"""
The script generates MCNP models, which will be used to produce WW files for the iww-gvr package testing.

The WW files will be produced using MCNP, advantg or other programs.
The test data will be produced by iww-gvr interactive mode using "soft", "soft2", "mit", "gvr" operation
and with various scenarios and parameters.

Both rectilinear and cylinder WW meshes are to be produced.
Both rectilinear and cylinder tally meshes are to be produced for GVR.

Number of voxels in the meshes (both WW and tally) should not be too large to keep testing times in reasonable range.

We are going also to implement benchmark tests. For these tests the meshes can be rather large,
to be equivalent to meshes from real problems.

"""

import itertools

import numpy as np


def create_cube_with_source_at_corner(
    edge_size: float = 100.0,
    cell_bins: int = 10,
    ww_bins: int = 10,
    mt_bins: int = 10,
    material: str = "H",
    density: float = None,
) -> None:
    """
    Create MCNP simple MCNP model with source in the corner and target tally in the center
    """
    half_edge = 0.5 * edge_size
    cell_bins_coords = np.linspace(-half_edge, half_edge, cell_bins + 1)

    cells_numbers = np.arange(1, cell_bins ** 3 + 1, dtype=np.int32).reshape(
        cell_bins, cell_bins, cell_bins
    )

    for k, j, i in np.ndindex(cells_numbers.shape):
        print(i, j, k, ":", cells_numbers[k, j, i], "0")


def main():
    create_cube_with_source_at_corner()


if __name__ == "__main__":
    main()
