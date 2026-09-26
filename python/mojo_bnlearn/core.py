"""bnlearn's conditional-independence tests and Gaussian score, on Mojo kernels.

The names here deliberately mirror `bnlearn.CITests` and
`bnlearn.structure_learning` so that a caller can swap one for the other:

    bn.CITests.chi_square(X='A', Y='B', Z=[], data=df, boolean=False)
    mbn.power_divergence('A', 'B', [], df, ci_test='pearson', boolean=False)

Both return the same triple. The difference is where the arithmetic happens:
bnlearn builds the contingency table with a pandas `groupby(...).size().unstack`
and hands it to scipy, while this module counts the states in one compiled pass
and evaluates the expected-frequency table and the statistic in a second.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from . import _lib

# bnlearn's CI-test aliases, mapped onto the Cressie-Read family.
CI_TESTS = {
    "chi_square": "pearson",
    "g_sq": "log-likelihood",
    "log_likelihood": "log-likelihood",
    "freeman_tuckey": "freeman-tukey",
    "modified_log_likelihood": "mod-log-likelihood",
    "neyman": "neyman",
    "cressie_read": "cressie-read",
    "power_divergence": "cressie-read",
}


def encode(data, columns: Sequence[str] | None = None):
    """Factorise the requested columns into contiguous integer state codes.

    bnlearn works on the raw values and lets pandas group them; here the
    grouping has to be explicit because the kernel indexes states directly.
    Returns `(codes, cardinalities)` where `codes` is an (nrows, ncols) int32
    array in the column order given and `cardinalities` is the matching tuple.
    """
    if columns is None:
        columns = list(data.columns)
    codes, cards, _states = _encode(data, columns)
    return codes, cards


def _encode(data, columns):
    """`encode` plus the sorted state values of each column."""
    columns = list(columns)
    blocks = []
    cards = []
    states = []
    for name in columns:
        codes, uniques = _factorise(data[name])
        blocks.append(codes)
        cards.append(len(uniques))
        states.append(uniques)
    if not blocks:
        return np.zeros((len(data), 0), dtype=np.int32), (), []
    return np.ascontiguousarray(np.stack(blocks, axis=1), dtype=np.int32), tuple(cards), states


def _factorise(values):
    """Map a column's distinct values onto 0..k-1 in ascending value order.

    Ascending order is the convention `pandas.groupby` uses when it sorts the
    group keys, so state `i` here means the same thing as the `i`-th row of
    bnlearn's `unstack` table. Ordering by first appearance instead would make
    the result depend on row order, which is not a property a statistical
    routine should have.
    """
    if hasattr(values, "to_numpy"):
        arr = values.to_numpy()
    else:
        arr = np.asarray(values)
    if arr.size == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=arr.dtype)
    if arr.dtype.kind in "biufc":
        if arr.dtype.kind == "f" and not np.isfinite(arr).all():
            raise ValueError("NaN or infinite values are not valid states")
    elif any(v is None or (isinstance(v, float) and v != v) for v in arr.ravel().tolist()):
        # Object columns are the slow path anyway: numpy cannot test `v != v`
        # across a Python object array without a Python loop, and a missing
        # value would otherwise become a state of its own.
        raise ValueError("missing values are not valid states")
    # `np.unique` with `return_inverse` does the sort and the lookup in one
    # compiled pass; a Python dict over every row would dominate the runtime of
    # anything the kernel is meant to speed up.
    uniques, inverse = np.unique(arr, return_inverse=True)
    return np.ascontiguousarray(inverse.reshape(arr.shape), dtype=np.int32), uniques


def contingency(data, X: str, Y: str, Z: Sequence[str] = (), return_labels: bool = False):
    """Stratified X x Y contingency counts, shape (rz, qx, qy).

    This is the table `bnlearn.CITests` builds with
    `data.groupby(Z).apply(lambda df: df.groupby([X, Y]).size().unstack(Y, fill_value=0))`,
    but on a fixed dense grid so the kernel can index it directly. States are
    numbered in ascending value order, strata in lexicographic order of Z, so
    the axes line up with pandas' own grouping. A stratum in which X or Y takes
    fewer states than it does overall comes back with zero rows or columns,
    which the statistic drops.

    With `return_labels=True` the sorted state values of X, Y and each
    conditioning column are returned alongside the counts, so a caller can
    label the axes.
    """
    Z = list(Z)
    if X in Z or Y in Z:
        raise ValueError(f"X or Y cannot also be in Z; got X={X!r} Z={Z}")
    columns = [X, Y] + Z
    codes, cards, states = _encode(data, columns)
    qx, qy = cards[0], cards[1]
    zcard = np.array(cards[2:], dtype=np.int32)
    rz = int(np.prod(zcard)) if len(zcard) else 1
    counts = _lib.contingency_counts(
        codes, 0, 1, np.array([i + 2 for i in range(len(Z))], dtype=np.int32),
        zcard, qx, qy, rz,
    )
    if return_labels:
        return counts, (states[0], states[1], states[2:])
    return counts



def power_divergence(
    X: str,
    Y: str,
    Z: Sequence[str],
    data,
    boolean: bool = True,
    ci_test: str = "cressie-read",
    significance_level: float = 0.05,
):
    """Conditional independence test of X and Y given Z.

    Mirrors `bnlearn.CITests.power_divergence`. The statistic and the degrees of
    freedom are summed over the strata of Z, exactly as bnlearn does, and the
    p-value is `chi2.sf(chi, dof)` computed by the Mojo incomplete-gamma
    kernel rather than by `scipy.stats.chi2.cdf`.

    `boolean=True` returns the independence verdict, `p >= significance_level`,
    which is the convention bnlearn uses for every one of its CI tests.
    """
    if ci_test not in CI_TESTS:
        raise ValueError(
            f"unknown ci_test {ci_test!r}; expected one of {sorted(CI_TESTS)}"
        )
    counts = contingency(data, X, Y, Z).astype(np.float64)
    stat, dof, _expected, _p = _lib.power_divergence(counts, CI_TESTS[ci_test], True)
    chi = float(np.sum(stat))
    total_dof = int(np.sum(dof))
    p_value = _lib.chi2_sf(chi, total_dof)
    if boolean:
        return p_value >= significance_level
    return chi, p_value, total_dof


def chi_square(X, Y, Z, data, boolean=True, significance_level=0.05, **kwargs):
    return power_divergence(X, Y, Z, data, boolean, "chi_square", significance_level)


def g_sq(X, Y, Z, data, boolean=True, significance_level=0.05, **kwargs):
    return power_divergence(X, Y, Z, data, boolean, "g_sq", significance_level)


def log_likelihood(X, Y, Z, data, boolean=True, significance_level=0.05, **kwargs):
    return power_divergence(X, Y, Z, data, boolean, "log_likelihood", significance_level)


def freeman_tuckey(X, Y, Z, data, boolean=True, significance_level=0.05, **kwargs):
    return power_divergence(X, Y, Z, data, boolean, "freeman_tuckey", significance_level)


def modified_log_likelihood(X, Y, Z, data, boolean=True, significance_level=0.05, **kwargs):
    return power_divergence(X, Y, Z, data, boolean, "modified_log_likelihood", significance_level)


def neyman(X, Y, Z, data, boolean=True, significance_level=0.05, **kwargs):
    return power_divergence(X, Y, Z, data, boolean, "neyman", significance_level)


def cressie_read(X, Y, Z, data, boolean=True, significance_level=0.05, **kwargs):
    return power_divergence(X, Y, Z, data, boolean, "cressie_read", significance_level)


def gauss_local_rss(y, X):
    """Residual sum of squares of `y` regressed on `X` with an intercept.

    The first half of `bnlearn.structure_learning.LogLikelihoodGauss._local_log_likelihood`.
    """
    return _lib.ols_rss(y, X)


def gauss_local_score(y, X, variance_floor: float = 1e-12) -> float:
    """Multivariate Gaussian local score of one node family.

    Reproduces `LogLikelihoodGauss._local_log_likelihood`: the residual sum of
    squares comes from the Mojo least-squares kernel, and the score is

        -0.5 * n * (log(2 pi) + 1 + log(max(rss / n, variance_floor)))

    Higher is better, as in bnlearn.
    """
    y = np.ascontiguousarray(y, dtype=np.float64).reshape(-1)
    n = y.size
    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    rss, _beta = _lib.ols_rss(y, X)
    variance = max(rss / n, variance_floor)
    return float(-0.5 * n * (np.log(2.0 * np.pi) + 1.0 + np.log(variance)))
