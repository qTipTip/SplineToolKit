from SplineTK.lib import evaluate_spline_vectorized


class Spline(object):

    def __init__(self, p, t, c):
        """
        Create a callable spline function.
        :param p: polynomial degree
        :param t: knot vector
        :param c: array of spline coefficients
        """

        self.p = p
        self.t = t
        self.c = c

    def __call__(self, x):
        """
        Evaluates the spline at the point(s) x.
        :param x: points of evaluation
        :return: f(x)
        """
        return evaluate_spline_vectorized(x, self.t, self.c, self.p)
