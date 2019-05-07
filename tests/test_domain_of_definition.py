from SplineTK import Spline, np


def test_domain_of_definition_endpoint_interpolation():
    t = (0, 0, 0, 1, 2, 3, 4, 4, 4)
    p = 2
    c = (0, 1, 2, 3, 4, 5)

    s = Spline(p, t, c)

    expected_domain_of_definition = (0, 4)
    computed_domain_of_definition = s.domain_of_definition

    np.testing.assert_equal(computed_domain_of_definition, expected_domain_of_definition)


def test_domain_of_definition_endpoint_multiplicity():
    t = (0, 0, 1, 2, 3, 4, 4, 4)
    p = 2
    c = (0, 1, 2, 3, 4, 5)

    s = Spline(p, t, c)

    expected_domain_of_definition = (1, 4)
    computed_domain_of_definition = s.domain_of_definition

    np.testing.assert_equal(computed_domain_of_definition, expected_domain_of_definition)


def test_domain_of_definition_uniform_knots():
    t = (0, 1, 2, 3, 4, 5, 6, 7, 8)
    p = 2
    c = (0, 1, 2, 3, 4, 5)

    s = Spline(p, t, c)

    expected_domain_of_definition = (2, 6)
    computed_domain_of_definition = s.domain_of_definition

    np.testing.assert_almost_equal(computed_domain_of_definition, expected_domain_of_definition)
