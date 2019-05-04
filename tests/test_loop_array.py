import numpy as np

from SplineTK.lib import _loop_array


def test_loop_knots():
    t = [0, 1, 2, 3, 4, 5, 5, 6]
    p = 3

    computed_c = _loop_array(t, p)
    expected_c = [0, 1, 2, 3, 4, 5, 5, 6, 0, 1, 2]
    np.testing.assert_almost_equal(computed_c, expected_c)


def test_loop_array_coefficient():
    c = [0, 1, 2, 3, 4]
    p = 2

    computed_c = _loop_array(c, p)
    expected_c = [0, 1, 2, 3, 4, 0, 1]
    np.testing.assert_almost_equal(computed_c, expected_c)


def test_loop_array_parametric_coefficients():
    c = [
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 5]
    ]
    p = 2

    computed_c = _loop_array(c, p)
    expected_c = [
        [0, 1, 2, 3, 4, 0, 1],
        [0, 1, 2, 3, 5, 0, 1]
    ]

    np.testing.assert_almost_equal(computed_c, expected_c)
