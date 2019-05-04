import numpy as np


def index(x, t, eps=np.spacing(1)):
    """
    Given an array t and a value x, return the index i such that
    t_i <= x < t_i+1. If x is equal to the last element of t, return the second to last index. If no
    such element is found, return -1. 
    :param eps: floating point value comparison tolerance.
    :param x: value
    :param t: array
    :return: index i
    """

    if x < t[0] or x > t[-1]:
        return -1
    if t[-1] - x < eps:
        for i in range(len(t) - 1, 0, -1):
            if t[i] < x:
                return i
    for i in range(len(t) - 1):
        if t[i] <= x < t[i + 1]:
            return i


def pad_knots(t, p):
    """
    Given an array t and a polynomial degree p, return the array with p + 1 values added to either end of the array.
    :param t: array
    :param p: int
    :return: padded array
    """

    return np.pad(t, pad_width=p + 1, mode='constant', constant_values=(t[0] - 1, t[-1] + 1))


def pad_and_reshape_coefficients(c, p):
    """
    Given a 1D / 2D array c and a polynomial degree p, make c 2D and pad each dimension with p + 1 values.
    :param c: coefficient vector
    :param p: polynomial degree
    :return: padded and reshaped c
    """

    c = np.atleast_2d(c)
    return np.pad(c, pad_width=((0, 0), (p + 1, p + 1)), mode='constant', constant_values=0)


def _evaluate_non_zero_b_splines(x, t, i, p):
    """
    Evaluates the non-zero degree p bsplines at x defined over the knot vector t.
    Never call this directly!

    :param x: point of evaluation
    :param t: knot vector
    :param i: index of x w.r.t t
    :param p: polynomial degree
    :return: array of p+1 values
    """

    b = 1
    for k in range(1, p + 1):
        t1 = t[i - k + 1: i + 1]
        t2 = t[i + 1: i + k + 1]
        omega = np.divide((x - t1), (t2 - t1), out=np.zeros_like(t1), where=((t2 - t1) != 0))
        b = np.append((1 - omega) * b, 0) + np.insert((omega * b), 0, 0)

    return b


def _evaluate_spline(x, t, c, p):
    """
    Evalutes the degree p spline given by the knot vector t, the coefficients c at the point x.
    Never call this directly!
    :param x: point of evaluation
    :param t: knot vector
    :param c: coefficient vector
    :param p: polynomial degree
    :return: f(x)
    """
    i = index(x, t) + p + 1
    t = pad_knots(t, p).astype(np.float64)
    c = pad_and_reshape_coefficients(c, p)

    if i == -1:
        return np.zeros((c.shape[0], 1))

    nonzero_b = _evaluate_non_zero_b_splines(x, t, i, p)
    nonzero_c = c[:, i - p:i + 1]

    return np.sum(np.multiply(nonzero_c, nonzero_b), axis=1, keepdims=True)


def evaluate_spline_vectorized(x, t, c, p):
    """
    Calls `evaluate_spline` for each point of the array x.
        :param x: points of evaluation
        :param t: knot vector
        :param c: coefficient vector
        :param p: polynomial degree
        :return: [f(x1), ..., f(xn)]
    """
    x = np.atleast_1d(x)
    y = np.array([_evaluate_spline(X, t, c, p) for X in x])

    return y.squeeze(axis=2).T.squeeze()


def knot_averages(t, p):
    """
    return the knot averages (greville abcissae) of t with respect to p.
    :param t: knot vector
    :param p: degree p
    :return: array
    """

    return np.convolve(a=t[1:-1], v=np.ones(p), mode='valid') / p
