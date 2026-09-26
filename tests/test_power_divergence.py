"""Parity of the power-divergence kernel against scipy and against bnlearn.

The statistic is a sum of per-cell terms involving `log` and `pow`, and Mojo
emits FMA, so the comparison uses a tolerance. What *is* exact is the integer
counting underneath, the expected-frequency table, and the degrees of freedom.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2_contingency

import mojo_bnlearn as mbn
from mojo_bnlearn import _lib

# Mojo's `log` differs from glibc's by up to 3.2e-9 relative (measured over
# 20000 points in [0.5, 50]), and every lambda except pearson builds on it, so a
# bound is set a few times above that: freeman-tukey divides by -0.125, so a
# relative error in the inner ratio is amplified eightfold before the sum.
# Pearson is pure arithmetic on the observed and expected counts and is held far
# tighter.
LOG_RTOL = 1e-7

# A relative error d in the statistic lands as an absolute error of about
# (chi / 2) * d in log p, so for a statistic in the tens the p-value inherits a
# relative error of order 1e-7. That is the bound the p-value comparisons use.
P_RTOL = 1e-6

LAMBDAS = [
    "pearson",
    "log-likelihood",
    "mod-log-likelihood",
    "neyman",
    "freeman-tukey",
    "cressie-read",
]


def random_table(rng, r, c, lo=1, hi=40):
    return rng.integers(lo, hi, size=(r, c)).astype(np.float64)


def drop_empty(table):
    """The sparse table scipy would have been handed, with empty margins cut."""
    rows = table.sum(1) > 0
    cols = table.sum(0) > 0
    return table[np.ix_(rows, cols)]


@pytest.mark.parametrize("lam", LAMBDAS)
def test_matches_scipy_on_a_dense_table(lam):
    rng = np.random.default_rng(7)
    obs = random_table(rng, 4, 5)
    stat, dof, expected, p = _lib.power_divergence(obs, lam, True)
    ref = chi2_contingency(obs, lambda_=lam)
    rtol = 1e-12 if lam == "pearson" else LOG_RTOL
    assert int(dof[0]) == ref.dof
    np.testing.assert_allclose(stat[0], ref.statistic, rtol=rtol)
    # The expected frequencies are products and quotients of the observed
    # counts, with no transcendental in the way, so they must match tightly.
    np.testing.assert_allclose(expected[0], ref.expected_freq, rtol=1e-13)
    assert p[0] == pytest.approx(ref.pvalue, rel=100 * rtol)


@pytest.mark.parametrize("lam", LAMBDAS)
def test_matches_scipy_on_a_square_table(lam):
    """A square table catches a swapped marginal; a 4x5 table can hide it."""
    rng = np.random.default_rng(8)
    obs = random_table(rng, 3, 3)
    stat, dof, expected, p = _lib.power_divergence(obs, lam, True)
    ref = chi2_contingency(obs, lambda_=lam)
    rtol = 1e-12 if lam == "pearson" else LOG_RTOL
    assert int(dof[0]) == ref.dof
    np.testing.assert_allclose(stat[0], ref.statistic, rtol=rtol)
    # The expected frequencies are products and quotients of the observed
    # counts, with no transcendental in the way, so they must match tightly.
    np.testing.assert_allclose(expected[0], ref.expected_freq, rtol=1e-13)
    assert p[0] == pytest.approx(ref.pvalue, rel=100 * rtol)


@pytest.mark.parametrize("lam", LAMBDAS)
def test_yates_correction_only_at_one_degree_of_freedom(lam):
    rng = np.random.default_rng(9)
    two_by_two = random_table(rng, 2, 2)
    stat_on, dof, _e, _p = _lib.power_divergence(two_by_two, lam, True)
    stat_off, _d, _e, _p = _lib.power_divergence(two_by_two, lam, False)
    assert int(dof[0]) == 1
    rtol = 1e-12 if lam == "pearson" else LOG_RTOL
    np.testing.assert_allclose(stat_on[0], chi2_contingency(two_by_two, lambda_=lam).statistic, rtol=rtol)
    np.testing.assert_allclose(stat_off[0], chi2_contingency(two_by_two, correction=False, lambda_=lam).statistic, rtol=rtol)
    # The correction must actually change the answer, or it is not applied.
    assert not np.isclose(stat_on[0], stat_off[0], rtol=1e-4)

    two_by_three = random_table(rng, 2, 3)
    s_on, d_on, _e, _p = _lib.power_divergence(two_by_three, lam, True)
    s_off, d_off, _e, _p = _lib.power_divergence(two_by_three, lam, False)
    assert int(d_on[0]) == 2
    np.testing.assert_allclose(s_on[0], s_off[0], rtol=0, atol=0)
    assert int(d_on[0]) == int(d_off[0])


def test_zero_observed_cell_reproduces_numpy_non_finite_terms():
    """A cell with zero observed but positive expected breaks two lambdas.

    scipy computes `0 * inf` for neyman and freeman-tukey (nan) and
    `e * log(e / 0)` for mod-log-likelihood (inf). The kernel reports those
    through a status flag rather than producing them itself, so this test is
    what keeps the flag honest.
    """
    obs = np.array([[0.0, 10.0, 20.0], [20.0, 20.0, 20.0]])
    expectations = {
        "pearson": 15.0,
        "log-likelihood": 20.92992575058192,
        "mod-log-likelihood": np.inf,
        "neyman": np.nan,
        "freeman-tukey": np.nan,
        "cressie-read": 16.064035431571913,
    }
    for lam, want in expectations.items():
        stat, dof, _e, p = _lib.power_divergence(obs, lam, True)
        assert int(dof[0]) == 2
        if np.isnan(want):
            assert np.isnan(stat[0]), lam
            assert np.isnan(p[0]), lam
        else:
            rtol = 1e-12 if lam == "pearson" else LOG_RTOL
            assert stat[0] == pytest.approx(want, rel=100 * rtol), lam
            assert p[0] == pytest.approx(chi2_contingency(obs, lambda_=lam).pvalue, rel=1000 * rtol)


def test_degenerate_rows_and_columns_are_dropped_from_the_dof():
    """scipy refuses a table with a zero expected cell; the sparse table is
    what bnlearn's `unstack` actually produces, so the kernel drops the empty
    row/column and the dof must follow the reduced shape."""
    obs = np.array([[0.0, 12.0, 3.0], [0.0, 14.0, 5.0]])
    reduced = drop_empty(obs)
    assert reduced.shape == (2, 2)
    with pytest.raises(ValueError, match="zero element"):
        chi2_contingency(obs, correction=False)
    for lam in LAMBDAS:
        stat, dof, expected, _p = _lib.power_divergence(obs, lam, True)
        ref = chi2_contingency(reduced, lambda_=lam)
        assert int(dof[0]) == ref.dof == 1
        assert expected[0][:, 0].sum() == 0.0
        assert expected[0].sum() == pytest.approx(reduced.sum())
        rtol = 1e-12 if lam == "pearson" else LOG_RTOL
        assert stat[0] == pytest.approx(ref.statistic, rel=100 * rtol, nan_ok=True)


def test_balanced_table_is_not_called_dependent():
    """A near-uniform table has a small statistic and a large p-value."""
    obs = np.array(
        [[25.0, 25.0, 25.0, 25.0], [25.0, 25.0, 25.0, 25.0], [25.0, 25.0, 25.0, 25.0]]
    )
    stat, dof, expected, p = _lib.power_divergence(obs, "pearson", True)
    assert stat[0] == 0.0
    assert int(dof[0]) == 6
    assert p[0] == 1.0
    assert expected[0].min() == 25.0
    rng = np.random.default_rng(10)
    noisy = rng.integers(20, 40, size=(3, 4)).astype(np.float64)
    stat, _dof, _e, p = _lib.power_divergence(noisy, "pearson", True)
    assert p[0] > 0.5
    assert p[0] == pytest.approx(chi2_contingency(noisy).pvalue, rel=1e-8)


def test_table_stack_is_independent_per_table():
    """Batching many tables must not let one table's margins leak into another."""
    rng = np.random.default_rng(21)
    tables = np.stack([random_table(rng, 3, 4) for _ in range(6)])
    batch = _lib.power_divergence(tables, "cressie-read", True)
    for t in range(tables.shape[0]):
        one = _lib.power_divergence(tables[t], "cressie-read", True)
        assert int(batch[1][t]) == int(one[1][0])
        assert batch[0][t] == pytest.approx(one[0][0], rel=1e-12)
        np.testing.assert_allclose(batch[2][t], one[2][0], rtol=1e-13)
        assert batch[3][t] == pytest.approx(one[3][0], rel=1e-9)


def test_threaded_stack_matches_a_table_at_a_time():
    """Above the crossover the stack is split across threads.

    Each table must land in the same slot with the same margins whether it ran
    on a thread or alone, so the split is invisible in the results.
    """
    rng = np.random.default_rng(22)
    ntab, qx, qy = 97, 12, 9
    tables = rng.integers(0, 30, size=(ntab, qx, qy)).astype(np.float64)
    batch = _lib.power_divergence(tables, "log-likelihood", False)
    assert _lib._workers(ntab, qx * qy) > 1, "this fixture is meant to fan out"
    for t in range(ntab):
        one = _lib.power_divergence(tables[t], "log-likelihood", False)
        assert int(batch[1][t]) == int(one[1][0])
        assert batch[0][t] == pytest.approx(one[0][0], rel=1e-13)
        np.testing.assert_array_equal(batch[2][t], one[2][0])
        assert batch[3][t] == pytest.approx(one[3][0], rel=1e-9)


def test_small_stack_stays_serial():
    assert _lib._workers(4, 100) == 1
    assert _lib._workers(100, 4) == 1
    assert _lib._workers(100, 400) > 1


def test_one_empty_table_in_a_stack_does_not_disturb_the_others():
    tables = np.array(
        [
            [[10.0, 20.0], [20.0, 30.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            [[4.0, 9.0], [11.0, 3.0]],
        ]
    )
    stat, dof, _e, p = _lib.power_divergence(tables, "pearson", True)
    assert int(dof[1]) == 0
    assert stat[1] == 0.0
    assert p[1] == 1.0
    assert stat[0] > 0 and stat[2] > 0


def random_categorical(n, cards, seed):
    rng = np.random.default_rng(seed)
    codes = np.column_stack([rng.integers(0, c, size=n) for c in cards])
    columns = [chr(ord("A") + i) for i in range(codes.shape[1])]
    return pd.DataFrame(codes, columns=columns)


BN_TESTS = [
    ("chi_square", mbn.chi_square),
    ("g_sq", mbn.g_sq),
    ("log_likelihood", mbn.log_likelihood),
    ("freeman_tuckey", mbn.freeman_tuckey),
    ("modified_log_likelihood", mbn.modified_log_likelihood),
    ("neyman", mbn.neyman),
    ("cressie_read", mbn.cressie_read),
]


@pytest.mark.parametrize("name,ours", BN_TESTS)
def test_matches_bnlearn_citests_unconditional(name, ours):
    bn = pytest.importorskip("bnlearn.CITests")
    data = random_categorical(4000, [3, 4], seed=31)
    theirs = getattr(bn, name)(X="A", Y="B", Z=[], data=data, boolean=False)
    got = ours("A", "B", [], data, boolean=False)
    rtol = 1e-12 if name == "chi_square" else LOG_RTOL
    assert got[2] == theirs[2]
    assert got[0] == pytest.approx(theirs[0], rel=100 * rtol)
    assert got[1] == pytest.approx(theirs[1], rel=P_RTOL)


@pytest.mark.parametrize("name,ours", BN_TESTS)
def test_matches_bnlearn_citests_conditional(name, ours):
    bn = pytest.importorskip("bnlearn.CITests")
    data = random_categorical(6000, [3, 4, 2, 3], seed=32)
    theirs = getattr(bn, name)(X="A", Y="B", Z=["C", "D"], data=data, boolean=False)
    got = ours("A", "B", ["C", "D"], data, boolean=False)
    rtol = 1e-12 if name == "chi_square" else LOG_RTOL
    assert got[2] == theirs[2]
    assert got[0] == pytest.approx(theirs[0], rel=1000 * rtol)
    assert got[1] == pytest.approx(theirs[1], rel=P_RTOL)


def test_boolean_mode_agrees_with_bnlearn():
    bn = pytest.importorskip("bnlearn.CITests")
    rng = np.random.default_rng(33)
    data = pd.DataFrame({"A": rng.integers(0, 2, 3000), "B": rng.integers(0, 3, 3000)})
    theirs = bool(bn.chi_square(
        X="A", Y="B", Z=[], data=data, boolean=True, significance_level=0.05
    ))
    ours = bool(mbn.chi_square("A", "B", [], data, boolean=True, significance_level=0.05))
    assert ours == theirs


def test_strong_dependence_is_detected_and_independence_is_not():
    """A deliberately dependent pair must be rejected, an independent one kept."""
    bn = pytest.importorskip("bnlearn.CITests")
    rng = np.random.default_rng(34)
    a = rng.integers(0, 3, 5000)
    dependent = pd.DataFrame({"A": a, "B": (a + rng.integers(0, 2, 5000)) % 3})
    assert not mbn.chi_square("A", "B", [], dependent, boolean=True)
    assert not bn.chi_square(
        X="A", Y="B", Z=[], data=dependent, boolean=True, significance_level=0.05
    )

    independent = pd.DataFrame(
        {"A": rng.integers(0, 3, 5000), "B": rng.integers(0, 3, 5000)}
    )
    assert mbn.chi_square("A", "B", [], independent, boolean=True)
    assert bn.chi_square(
        X="A", Y="B", Z=[], data=independent, boolean=True, significance_level=0.05
    )


def test_x_in_z_is_rejected():
    data = random_categorical(100, [2, 2, 2], seed=35)
    with pytest.raises(ValueError, match="cannot also be in Z"):
        mbn.chi_square("A", "B", ["A"], data)
