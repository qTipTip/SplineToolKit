import numpy as np

from SplineTK.lib import evaluate_spline_vectorized
from SplineTK.spline import Spline


def test_single_linear_basis_function():
    t = [0, 1, 2]
    x = np.linspace(0, 2, 11)
    c = [1]
    p = 1

    computed_y = evaluate_spline_vectorized(x, t, c, p)
    expected_y = list(map(lambda x: x if 0 <= x < 1 else 2 - x, x))

    np.testing.assert_almost_equal(computed_y, expected_y,
                                   err_msg=f'computed values {computed_y} does not agree with expected values {expected_y}')


def test_single_linear_basis_function_parametric():
    t = [0, 1, 2]
    x = np.linspace(0, 2, 11)
    c = [[1], [0]]
    p = 1

    computed_y = evaluate_spline_vectorized(x, t, c, p)
    expected_y = np.atleast_2d(list(map(lambda x: x if 0 <= x < 1 else 2 - x, x)))
    expected_y = np.concatenate((expected_y, np.zeros_like(expected_y)), axis=0)
    np.testing.assert_almost_equal(computed_y, expected_y,
                                   err_msg=f'computed values {computed_y} does not agree with expected values {expected_y}')


def test_single_quadratic_basis_function():
    t = [0, 1, 2, 3]
    x = np.linspace(0, 3, 11)
    c = [1]
    p = 2

    computed_y = evaluate_spline_vectorized(x, t, c, p)
    expected_y = exact_quadratic(x)

    np.testing.assert_almost_equal(computed_y, expected_y,
                                   err_msg=f'computed values {computed_y} does not agree with expected values {expected_y}')


def test_quadratic_parametric_spline():
    t = [0, 1, 2, 3]
    x = np.linspace(0, 3, 20)
    c = [
        [1], [1]
    ]
    p = 2

    f = Spline(p, t, c)

    computed_y = f(x)
    expected_y = [
        c[0] * exact_quadratic(x),
        c[1] * exact_quadratic(x)
    ]

    np.testing.assert_almost_equal(computed_y, expected_y,
                                   err_msg=f'computed values {computed_y} does not agree with expected values {expected_y}')


def test_quadratic_spline_nonzero_at_endpoint():
    t = (0, 0, 0, 1, 2, 3, 4, 5, 5, 5)
    p = 2
    c = (-1, 1, -1, 1, -1, 1, -1)

    f = Spline(p, t, c)
    x = np.linspace(0, 5, num=60)
    y = f(x)

    assert np.not_equal(y[-1],
                        0), f'the spline is evaluated to be {y[-1]} at the endpoint, even though we expect {c[-1]}'


@np.vectorize
def exact_quadratic(x):
    if 0 <= x < 1:
        return x ** 2 / 2
    elif 1 <= x < 2:
        return (x * (2 - x) + (3 - x) * (x - 1)) / 2
    elif 2 <= x < 3:
        return (3 - x) ** 2 / 2
    else:
        return 0
