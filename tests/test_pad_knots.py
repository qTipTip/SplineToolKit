import numpy as np

from SplineTK.lib import pad_knots


def test_pad_knots():
    t = [0, 1, 2, 3, 4]
    p = 3
    expected_array = [-1, -1, -1, -1, 0, 1, 2, 3, 4, 5, 5, 5, 5]
    computed_array = pad_knots(t, p)

    np.testing.assert_almost_equal(computed_array, expected_array,
                                   err_msg=f'computed_array {computed_array} does not match expected array {expected_array}')
