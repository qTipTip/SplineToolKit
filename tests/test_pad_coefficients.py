import numpy as np

from SplineTK.lib import pad_and_reshape_coefficients


def test_pad_coefficient_1d():
    c = [0, 1, 2, 3, 4]
    p = 2
    expected_result = np.array([[0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0]])
    computed_result = pad_and_reshape_coefficients(c, p)

    np.testing.assert_almost_equal(computed_result, expected_result,
                                   err_msg=f'padded coefficients {computed_result} does not equal expected results {expected_result}')


def test_pad_coefficient_2d():
    c = [
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 5]
    ]
    p = 2
    expected_result = np.array([[0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0],
                                [0, 0, 0, 1, 2, 3, 4, 5, 0, 0, 0]])
    computed_result = pad_and_reshape_coefficients(c, p)

    np.testing.assert_almost_equal(computed_result, expected_result,
                                   err_msg=f'padded coefficients {computed_result} does not equal expected results {expected_result}')
