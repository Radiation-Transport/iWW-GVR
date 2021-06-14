import numpy as np

from numpy import array
from numpy.testing import assert_array_equal

import iww_gvr.main as m
import pytest

from iww_gvr.main import ISnumber, extend_matrix, load
from iww_gvr.utils.resource import path_resolver

data_path = path_resolver("tests")


def a(*elements, dtype=float):
    """Typing saver"""
    return np.array(elements, dtype=dtype)


@pytest.mark.parametrize(
    "inp, expected",
    [
        ("1.21 1e-7 8", True),
        ("a 0", False),
    ],
)
def test_is_number(inp, expected):
    actual = ISnumber(inp.split())
    assert expected == actual


@pytest.mark.parametrize(
    "msg, inp, expected",
    [
        (
            "1D array",
            [1, 2],
            [0, 1, 2, 0],
        ),
        (
            "2D array",
            [[1, 2], [3, 4]],
            [
                [0, 0, 0, 0],
                [0, 1, 2, 0],
                [0, 3, 4, 0],
                [0, 0, 0, 0],
            ],
        ),
        (
            "3D array",
            [
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]],
            ],
            [
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0],
                    [0, 1, 2, 0],
                    [0, 3, 4, 0],
                    [0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0],
                    [0, 5, 6, 0],
                    [0, 7, 8, 0],
                    [0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
            ],
        ),
    ],
)
def test_extend_matrix(msg, inp, expected):
    actual = extend_matrix(array(inp, dtype=float))
    assert_array_equal(array(expected), actual, err_msg=msg)


# TODO dvp: any constructor is to be trivial, and this is not the case now.
#           To me, the current design is not acceptable.
@pytest.mark.xfail(reason="The constructor is absolutely not trivial.")
def test_cartesian_constructor():
    filename = "dummy.wwinp"
    x = a(1, 2, 3)
    y = a(4, 5, 6)
    z = a(7, 8, 9)
    nbins = x.size * y.size * z.size  # including bin from origin
    particles_qty = 1
    ww1 = np.linspace(1.0, 8.0, 8, endpoint=True, dtype=float)
    eb1 = a(0, 20)
    ww2 = None
    eb2 = None
    extra = {
        "B1_if": 1,
        "B1_iv": 1,
        "B1_ne": ["1"],
        "B1_ni": 1,
        "B1_nr": 10,
        "B2_Xf": x.size,
        "B2_Xo": 0.0,
        "B2_Yf": y.size,
        "B2_Yo": 0.0,
        "B2_Zf": z.size,
        "B2_Zo": 0.0,
        "B2_par": False,
        # "vec_coarse": vec_coarse,
        # "vec_fine": vec_fine,
    }
    cartesian = m.ww_item(
        filename, x, y, z, nbins, particles_qty, ww1, eb1, ww2, eb2, extra
    )
    assert_array_equal(cartesian.X, x)
    assert_array_equal(cartesian.Y, y)
    assert_array_equal(cartesian.Z, z)


@pytest.mark.skip(
    reason="Segmentation fault in matplotlib destroy? What involves matplotlib on reading? Check."
)
def test_read_simple_cartesian():
    path = data_path("data/simple_cartesian.wwinp")
    assert path.exists()
    mesh: m.ww_item = load(str(path))
    assert_array_equal(mesh.X, np.linspace(0, 3, 4, dtype=float))
    assert_array_equal(mesh.Y, np.linspace(0, 3, 4, dtype=float))
    assert_array_equal(mesh.Z, np.linspace(0, 3, 4, dtype=float))
    assert 1 == len(mesh.wwe), "Only one energy bin was specified"
    assert 27 == len(
        mesh.wwe[0][0]
    ), "Number of bins for the only energy bin and the only particle"
    wwme = mesh.wwme
    assert 1 == len(wwme)
    ww = wwme[0][0]
    assert 27 == ww.size
    assert (3, 3, 3) == ww.shape


# TODO dvp: see below
@pytest.mark.skip(reason="Segmentation fault in matplotlib? Check.")
@pytest.mark.slow
def test_read_simple_cartesian_extended():
    path = data_path("data/simple_cartesian.wwinp")
    assert path.exists()
    mesh: m.ww_item = load(str(path))
    # TODO dvp: identify assertions in the following
    mesh.info()
    for degree in ["all", "auto", 20.0]:
        zone_id, factor = m.zoneDEF(mesh, degree)
        m.analyse(mesh, zone_id, factor)


if __name__ == "__main__":
    pytest.main()
