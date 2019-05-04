import numpy as np

from SplineTK.lib import evaluate_spline_vectorized, knot_averages, _loop_coefficients, _loop_uniform_knots


class Spline(object):
    """
    Represents a spline function or a parametric spline curve.
    """

    def __init__(self, p, t, c):
        """
        Create a callable spline function.
        :param p: polynomial degree
        :param t: knot vector
        :param c: array of spline coefficients
        """

        self.p = p
        self.t = np.array(t, dtype=np.float64)
        self.c = np.atleast_2d(c).astype(dtype=np.float64)

    def __call__(self, x):
        """
        Evaluates the spline at the point(s) x.
        :param x: points of evaluation
        :return: f(x)
        """
        return evaluate_spline_vectorized(x, self.t, self.c, self.p)

    @property
    def control_polygon(self):
        if self.c.shape[0] == 1:
            return [knot_averages(self.t, self.p), self.c[0]]
        else:
            return self.c


class NonUniformKnotsError(ValueError):
    pass


class ClosedSpline(Spline):
    """
    Represents a closed parametric spline curve or a periodic spline function.
    """

    def __init__(self, p, t, c):
        # enforce uniform knot vector?

        dt = np.diff(t)
        if not np.all(dt == dt[0]):
            raise NonUniformKnotsError()

        super().__init__(p, t, c)
        self.t = _loop_uniform_knots(self.t, self.p, dt[0])
        self.c = _loop_coefficients(self.c, self.p)

    @property
    def domain_of_definition(self):
        return [self.t[self.p], self.t[-self.p - 1]]
