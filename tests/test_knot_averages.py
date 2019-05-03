import numpy as np

from SplineTK.lib import knot_averages


def test_knot_averages_linear():
    t = [0, 1, 2, 3, 3, 4]
    p = 1

    computed_averages = knot_averages(t, p)
    expected_averages = [3, 5, 6, 7]

    np.testing.assert_almost_equal(computed_averages, expected_averages,
                                   err_msg=f'computed knot averages {computed_averages} do not agree with expected averages {expected_averages}')


def test_knot_averages_quadratic():
    t = [0, 1, 2, 3, 3, 4]
    p = 2

    computed_averages = knot_averages(t, p)
    expected_averages = [3, 4, 5]

    np.testing.assert_almost_equal(computed_averages, expected_averages,
                                   err_msg=f'computed knot averages {computed_averages} do not agree with expected averages {expected_averages}')
