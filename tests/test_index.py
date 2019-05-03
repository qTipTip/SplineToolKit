import numpy as np
import pytest

from SplineTK.lib import index


@pytest.mark.parametrize(["x", "expected_i"],
                         [
                             (0, 0),
                             (0.5, 0),
                             (1, 1),
                             (5, 6),
                             (3.5, 3)
                         ])
def test_index_interior(x, expected_i):
    t = [0, 1, 2, 3, 4, 5, 5, 6]
    computed_i = index(x, t)
    np.testing.assert_equal(computed_i, expected_i,
                            err_msg=f'computed index {computed_i} does not match expected index {expected_i}')


@pytest.mark.parametrize(["x", "expected_i"],
                         [
                             (0, 1),
                             (5, 5),
                         ])
def test_index_edge(x, expected_i):
    t = [0, 0, 1, 2, 3, 4, 5, 5]
    computed_i = index(x, t)
    np.testing.assert_equal(computed_i, expected_i,
                            err_msg=f'computed index {computed_i} does not match expected index {expected_i}')
