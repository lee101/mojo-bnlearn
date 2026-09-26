"""Parity of the incomplete-gamma p-value kernel against scipy.

`chi2_contingency` turns its statistic into a p-value with `chi2.sf(stat, dof)`,
which is a regularised upper incomplete gamma. The kernel implements that with
the usual series/continued-fraction split, so the reference is
`scipy.stats.chi2.sf` and `scipy.special.gammaincc` directly.
"""

import numpy as np
import pytest
from scipy.special import gammainc, gammaincc
from scipy.stats import chi2

from mojo_bnlearn import _lib

# The regularised incomplete gamma carries an exponential prefactor
# `exp(-x + a*log(x) - lgamma(a))`. An error of `d` in `log` becomes an absolute
# error of `a * d * |log x|` in the exponent, so the relative error of the result
# grows roughly linearly in `a = df / 2`. Mojo's `log` differs from glibc's by
# up to 3.2e-9, and the measured worst case is 4.9e-9 at df = 1 rising to
# 1.0e-7 at df = 199; 1e-8 * df sits above the measurement at every df tested.
def rtol_for_df(df):
    return 1e-8 * df


def rtol_for_a(a):
    return 1e-8 * a


@pytest.mark.parametrize("df", [1, 2, 3, 5, 8, 13, 40, 199])
def test_chi2_sf_matches_scipy_across_the_whole_range(df):
    x = np.concatenate(
        [
            np.linspace(1e-6, 5.0, 200),
            np.linspace(5.0, 400.0, 200),
            np.array([1e-8, 1e-3, 1.0, 37.5, 250.0, 1000.0]),
        ]
    )
    got = _lib.chi2_sf_array(x, df)
    want = chi2.sf(x, df)
    np.testing.assert_allclose(got, want, rtol=rtol_for_df(df))


def test_series_and_continued_fraction_agree_across_the_switch():
    """The two branches meet at x ~ a + 1; a bug in either shows up as a jump."""
    a = 5.0
    xs = np.linspace(a - 3.0, a + 4.0, 400)
    got = np.array([_lib.gammaq(a, float(x)) for x in xs])
    want = gammaincc(a, xs)
    np.testing.assert_allclose(got, want, rtol=rtol_for_a(a))

    jump = np.abs(np.diff(got))
    assert jump.max() < 5e-3, "the series and continued fraction do not meet"


def test_gammaq_matches_gammaincc():
    rng = np.random.default_rng(2)
    for a in (0.25, 1.0, 2.5, 7.0, 20.0, 100.0):
        xs = np.concatenate(
            [np.linspace(1e-6, 2.0, 60), np.linspace(2.0, 4 * a + 50, 60)]
        )
        got = np.array([_lib.gammaq(a, float(x)) for x in xs])
        np.testing.assert_allclose(got, gammaincc(a, xs), rtol=rtol_for_a(a))


def test_gammaq_boundary_values():
    assert _lib.gammaq(3.0, 0.0) == 1.0
    # Q(a, x) is the tail, so it must fall monotonically from 1 to 0.
    xs = np.linspace(0.0, 200.0, 500)
    vals = np.array([_lib.gammaq(4.0, float(x)) for x in xs])
    assert vals[0] == pytest.approx(1.0)
    assert (np.diff(vals) <= 0).all()
    assert vals[-1] < 1e-60


def test_chi2_sf_survives_extreme_tails():
    """Deep in the tail the series underflows to zero; that is the right answer."""
    assert _lib.chi2_sf(1e5, 2) == 0.0
    assert _lib.chi2_sf(1e3, 1) < 1e-200
    assert _lib.chi2_sf(0.0, 4) == 1.0
    assert np.isnan(_lib.chi2_sf(float("nan"), 4))


def test_chi2_sf_is_the_complement_of_the_cdf():
    """cdf + sf == 1, checked where the cdf is not within rounding of one.

    Close to x = 0 the cdf is 1 to within 1e-12, and forming the complement of a
    number that rounds to 1 has no significant digits left to compare; Mojo's
    `log` also differs from glibc's by 3e-9, which is what bounds the accuracy
    of the series in that region. Away from it the identity is tight.
    """
    rng = np.random.default_rng(3)
    for df in (1, 4, 11, 97):
        x = rng.uniform(0.01, 60.0, 50)
        keep = chi2.cdf(x, df) < 0.99
        x = x[keep]
        total = chi2.cdf(x, df) + _lib.chi2_sf_array(x, df)
        np.testing.assert_allclose(total, np.ones_like(x), rtol=rtol_for_df(df))


def test_chi2_sf_array_matches_the_scalar_entry_point():
    rng = np.random.default_rng(4)
    x = rng.uniform(0.0, 100.0, 97)
    vector = _lib.chi2_sf_array(x, 6)
    scalar = np.array([_lib.chi2_sf(float(v), 6) for v in x])
    np.testing.assert_array_equal(vector, scalar)


def test_incomplete_gamma_survives_its_own_pitfall():
    """`P(a, x) + Q(a, x) == 1` is the identity that catches a sign error.

    The series branch returns `1 - P`, so it is only checked where the
    complement leaves significant digits. Where it does not, the kernel is held
    against scipy instead, with a tolerance set by Mojo's `log`.
    """
    rng = np.random.default_rng(5)
    for a in (0.5, 3.0, 11.0):
        x = rng.uniform(0.05, 80.0, 200)
        q = np.array([_lib.gammaq(a, float(v)) for v in x])
        keep = q > 1e-3
        total = gammainc(a, x[keep]) + q[keep]
        np.testing.assert_allclose(total, np.ones(int(keep.sum())), rtol=rtol_for_a(a))
        np.testing.assert_allclose(q[~keep], gammaincc(a, x[~keep]), rtol=rtol_for_a(a))
