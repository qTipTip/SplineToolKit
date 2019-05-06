import numpy as np

from SplineTK.lib import _loop_coefficients, _loop_uniform_knots


def test_loop_knots():
    t = (0, 1, 2, 3, 4)
    p = 2
    dt = 1

    computed_t = _loop_uniform_knots(t, p, dt)
    expected_t = (0, 1, 2, 3, 4, 5, 6)

    np.testing.assert_almost_equal(computed_t, expected_t)


def test_loop_array_coefficient():
    c = [0, 1, 2, 3, 4]
    p = 2

    computed_c = _loop_coefficients(c, p)
    expected_c = [0, 1, 2, 3, 4, 0, 1]
    np.testing.assert_almost_equal(computed_c, expected_c)


def test_loop_array_parametric_coefficients():
    c = [
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 5]
    ]
    p = 2

    computed_c = _loop_coefficients(c, p)
    expected_c = [
        [0, 1, 2, 3, 4, 0, 1],
        [0, 1, 2, 3, 5, 0, 1]
    ]

    np.testing.assert_almost_equal(computed_c, expected_c)
