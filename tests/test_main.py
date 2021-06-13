import pytest
from iww_gvr.main import ISnumber, extend_matrix
from numpy import array
from numpy.testing import assert_array_equal


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
            ],
        ),
    ],
)
def test_extend_matrix(msg, inp, expected):
    actual = extend_matrix(array(inp, dtype=float))
    assert_array_equal(array(expected), actual, err_msg=msg)


if __name__ == "__main__":
    pytest.main()
