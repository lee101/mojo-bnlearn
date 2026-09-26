"""Parity of the least-squares residual kernel with bnlearn's Gaussian score.

`bnlearn.structure_learning.LogLikelihoodGauss` is bnlearn's own scorer, not
pgmpy's: it fits each node on its parents with `numpy.linalg.lstsq` and scores
the fit by residual variance. The Mojo kernel does the same fit through the
normal equations, so these tests compare it against the real class and against
a direct least-squares reference.

FMA makes the two least-squares paths differ in the last bits, and the normal
equations square the condition number, so the tolerance is relative and stated
rather than exact.
"""

import numpy as np
import pandas as pd
import pytest

import mojo_bnlearn as mbn
from mojo_bnlearn import _lib

# Well-conditioned fixtures: a random normal design with up to 5 columns has a
# Gram matrix condition number of order 5, so the normal equations lose about
# one digit and one step of iterative refinement takes it back.
RTOL = 1e-9


def lstsq_rss(y, X):
    design = np.column_stack((np.ones(y.size), X)) if X.shape[1] else np.ones((y.size, 1))
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    return float(resid @ resid), beta


@pytest.mark.parametrize("p", [0, 1, 2, 5])
def test_residuals_match_a_direct_least_squares_fit(p):
    rng = np.random.default_rng(100 + p)
    n = 2000
    y = rng.standard_normal(n) + 3.0
    X = rng.standard_normal((n, p))
    got, _beta = _lib.ols_rss(y, X)
    want, _ = lstsq_rss(y, X)
    assert got == pytest.approx(want, rel=RTOL)


def test_coefficients_match_linalg_lstsq():
    rng = np.random.default_rng(110)
    n, p = 800, 4
    X = rng.standard_normal((n, p))
    beta_true = np.array([1.5, -2.0, 0.25, 3.0, -0.5])
    y = beta_true[0] + X @ beta_true[1:] + rng.standard_normal(n) * 0.3
    _rss, beta = _lib.ols_rss(y, X)
    np.testing.assert_allclose(beta, beta_true, rtol=2e-2)
    _want, want_beta = lstsq_rss(y, X)
    np.testing.assert_allclose(beta, want_beta, rtol=1e-7)


def test_intercept_only_fit_is_the_mean():
    rng = np.random.default_rng(120)
    y = rng.standard_normal(500) * 3.0 + 7.0
    rss, beta = _lib.ols_rss(y, np.zeros((500, 0)))
    centered = y - y.mean()
    assert rss == pytest.approx(float(centered @ centered), rel=RTOL)
    assert beta[0] == pytest.approx(y.mean(), rel=RTOL)


def test_a_perfect_fit_has_almost_no_residual():
    """A degenerate case for the normal equations: exactly zero residual.

    If the right-hand side or the elimination were wrong, this is the first
    thing to break, and the residual would be O(n) instead of O(eps).
    """
    rng = np.random.default_rng(130)
    n, p = 300, 3
    X = rng.standard_normal((n, p))
    beta_true = np.array([2.0, 1.0, -1.0, 0.5])
    y = beta_true[0] + X @ beta_true[1:]
    rss, _beta = _lib.ols_rss(y, X)
    assert rss == pytest.approx(0.0, abs=1e-18)


def test_residual_is_invariant_to_column_scaling():
    """OLS is equivariant under a rescaling of the predictors, so the rss is too."""
    rng = np.random.default_rng(140)
    n = 600
    X = rng.standard_normal((n, 2))
    y = rng.standard_normal(n)
    base, _ = _lib.ols_rss(y, X)
    scaled, _ = _lib.ols_rss(y, X * np.array([1000.0, 0.001]))
    assert scaled == pytest.approx(base, rel=1e-8)


def test_rank_deficient_design_is_reported_not_silently_solved():
    """A duplicated column makes the Gram matrix singular; the kernel must say so."""
    rng = np.random.default_rng(150)
    n = 200
    col = rng.standard_normal(n)
    y = rng.standard_normal(n)
    with pytest.raises(np.linalg.LinAlgError, match="rank deficient"):
        _lib.ols_rss(y, np.column_stack([col, col]))


def test_local_score_matches_bnlearn_loglikelihood_gauss():
    bn = pytest.importorskip("bnlearn.structure_learning")
    rng = np.random.default_rng(160)
    n = 3000
    frame = pd.DataFrame(
        {
            "target": rng.standard_normal(n) * 2.0 + 1.0,
            "p1": rng.standard_normal(n),
            "p2": rng.standard_normal(n),
            "p3": rng.standard_normal(n),
        }
    )
    scorer = bn.LogLikelihoodGauss(frame)
    y = frame["target"].to_numpy()
    for parents in (["p1"], ["p1", "p2"], ["p1", "p2", "p3"]):
        X = frame[parents].to_numpy()
        ours = mbn.gauss_local_score(y, X)
        theirs = scorer.local_score("target", parents)
        assert ours == pytest.approx(theirs, rel=RTOL)


def test_local_score_without_parents_matches_bnlearn():
    bn = pytest.importorskip("bnlearn.structure_learning")
    rng = np.random.default_rng(161)
    n = 1500
    frame = pd.DataFrame({"target": rng.standard_normal(n) + 4.0})
    scorer = bn.LogLikelihoodGauss(frame)
    ours = mbn.gauss_local_score(frame["target"].to_numpy(), np.zeros((n, 0)))
    assert ours == pytest.approx(scorer.local_score("target", []), rel=RTOL)


def test_gaussian_scores_rank_a_planted_relationship_first():
    """A real signal must score above noise; a scorer that is flat cannot."""
    rng = np.random.default_rng(170)
    n = 4000
    noise = rng.standard_normal((n, 3))
    signal = noise[:, 0] * 2.0 + rng.standard_normal(n)
    scores = [mbn.gauss_local_score(signal, noise[:, [i]]) for i in range(3)]
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]
    # Adding the true parent must improve the fit.
    with_true = mbn.gauss_local_score(signal, noise)
    assert with_true > mbn.gauss_local_score(signal, noise[:, 1:])


def test_aic_and_bic_offsets_follow_bnlearn():
    """bnlearn's AICGauss / BICGauss are the log-likelihood plus a penalty."""
    bn = pytest.importorskip("bnlearn.structure_learning")
    rng = np.random.default_rng(171)
    n = 1000
    frame = pd.DataFrame(
        {
            "target": rng.standard_normal(n),
            "p1": rng.standard_normal(n),
            "p2": rng.standard_normal(n),
        }
    )
    y = frame["target"].to_numpy()
    parents = ["p1", "p2"]
    X = frame[parents].to_numpy()
    base = mbn.gauss_local_score(y, X)
    assert bn.AICGauss(frame).local_score("target", parents) == pytest.approx(
        base - (len(parents) + 2), rel=RTOL
    )
    assert bn.BICGauss(frame).local_score("target", parents) == pytest.approx(
        base - 0.5 * (len(parents) + 2) * np.log(n), rel=RTOL
    )


def test_variance_floor_engages_for_a_saturated_fit():
    """More parents than samples cannot fit, so the floor has to take over."""
    bn = pytest.importorskip("bnlearn.structure_learning")
    rng = np.random.default_rng(180)
    n = 12
    X = rng.standard_normal((n, 6))
    y = rng.standard_normal(n)
    rss, _beta = _lib.ols_rss(y, X)
    assert rss >= 0.0
    floored = mbn.gauss_local_score(y, X, variance_floor=1e-12)
    exact = -0.5 * n * (np.log(2 * np.pi) + 1.0 + np.log(rss / n))
    assert floored == pytest.approx(exact, rel=RTOL)
