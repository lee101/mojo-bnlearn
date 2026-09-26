"""Correctness-gated benchmark for mojo-bnlearn.

Every case checks its output against a reference before it is timed, so a
regression in the Mojo kernels shows up as a correctness failure rather than as
a suspiciously good number. The reference in each case is the fastest
reasonable NumPy/pandas formulation, not a Python loop that NumPy would never
use.

Run with: python bench/bench.py
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import mojo_bnlearn as mbn  # noqa: E402
from mojo_bnlearn import _lib  # noqa: E402


def _time(fn, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _frame(n, cards, seed):
    import pandas as pd

    rng = np.random.default_rng(seed)
    codes = np.column_stack([rng.integers(0, c, size=n) for c in cards])
    return pd.DataFrame(codes, columns=[chr(ord("A") + i) for i in range(codes.shape[1])])


def bench_contingency(n: int = 2_000_000, cards=(2, 3)):
    """The counting pass, against the `groupby(...).size()` that bnlearn uses.

    This is the hot loop in `bnlearn.CITests`: every CI test starts by counting
    the joint states, once per call. The reference is pandas' own grouped size
    plus the `unstack`, which is what the upstream code does.
    """
    data = _frame(n, cards, 0)

    def reference():
        return data.groupby(["A", "B"]).size().unstack("B", fill_value=0).to_numpy()

    got = mbn.contingency(data, "A", "B", []).astype(np.int64)
    np.testing.assert_array_equal(got[0], reference())

    # The counting pass on its own, with the factorisation already done. The
    # end-to-end row above is the number a caller sees; this one isolates the
    # kernel from the `np.unique` that precedes it, and it is the honest way to
    # compare a single fused traversal against a grouped size.
    codes, (qx, qy) = mbn.encode(data, ["A", "B"])
    zcols = np.zeros(0, dtype=np.int32)
    zcard = np.zeros(0, dtype=np.int32)

    def kernel_only():
        return _lib.contingency_counts(codes, 0, 1, zcols, zcard, qx, qy, 1)

    np.testing.assert_array_equal(kernel_only()[0], reference())
    return (
        f"contingency n={n} {cards[0]}x{cards[1]}",
        _time(reference, 3),
        _time(kernel_only, 3),
    )


def bench_contingency_end_to_end(n: int = 2_000_000, cards=(2, 3)):
    """The full `contingency` call, factorisation included.

    The kernel is 15x the grouped size, but `np.unique` has to sort every column
    to number the states, and that sort costs more than the traversal it
    enables. Reporting only the kernel would overstate what a caller sees.
    """
    data = _frame(n, cards, 0)

    def reference():
        return data.groupby(["A", "B"]).size().unstack("B", fill_value=0).to_numpy()

    np.testing.assert_array_equal(
        mbn.contingency(data, "A", "B", []).astype(np.int64)[0], reference()
    )
    return (
        f"contingency e2e n={n}",
        _time(reference, 3),
        _time(lambda: mbn.contingency(data, "A", "B", []), 3),
    )


def bench_contingency_stratified(n: int = 1_000_000, cards=(3, 4, 2, 2)):
    """Same counting pass, but stratified over two conditioning columns."""
    data = _frame(n, cards, 1)

    def reference():
        out = []
        for _state, frame in data.groupby(["C", "D"]):
            out.append(
                frame.groupby(["A", "B"]).size().unstack("B", fill_value=0).to_numpy()
            )
        return out

    def ours():
        counts = mbn.contingency(data, "A", "B", ["C", "D"])
        return [c for c in counts if c.sum()]

    want = reference()
    got = ours()
    assert len(got) == len(want)
    for a, b in zip(got, want):
        np.testing.assert_array_equal(a, b)

    return (
        f"contingency | C,D  n={n}",
        _time(reference, 3),
        _time(ours, 3),
    )


def _numpy_power_divergence(obs, lambda_):
    """A vectorised NumPy implementation of the same statistic over a stack."""
    obs = obs.astype(np.float64)
    rows = obs.sum(-1, keepdims=True)
    cols = obs.sum(-2, keepdims=True)
    total = obs.sum((-1, -2), keepdims=True)
    expected = rows * cols / total
    live = (rows > 0) & (cols > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        if lambda_ == "pearson":
            terms = (obs - expected) ** 2 / expected
        elif lambda_ == "log-likelihood":
            terms = np.where(obs > 0, 2.0 * obs * np.log(obs / expected), 0.0)
        else:
            lam = {"mod-log-likelihood": -1.0, "neyman": -2.0, "freeman-tukey": -0.5,
                   "cressie-read": 2.0 / 3.0}[lambda_]
            denom = 0.5 * lam * (lam + 1.0)
            terms = obs * ((obs / expected) ** lam - 1.0) / denom
    terms = np.where(live, terms, 0.0)
    return terms.sum((-1, -2)), expected


def bench_power_divergence(ntab: int = 200_000, qx: int = 20, qy: int = 20):
    """Expected frequencies plus the statistic over a large stack of tables.

    The vectorised NumPy reference is the fairest baseline: it is fully
    broadcast, so it is doing the same arithmetic in the same order of
    magnitude of work. Yates' correction is off in both, because a vectorised
    reference for the corrected statistic would need the same per-table dof
    branch the kernel already does serially.
    """
    rng = np.random.default_rng(2)
    tables = rng.integers(0, 25, size=(ntab, qx, qy)).astype(np.float64)

    want, _ = _numpy_power_divergence(tables, "cressie-read")
    got, dof, expected, _p = _lib.power_divergence(tables, "cressie-read", False)
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-4)
    np.testing.assert_allclose(expected.sum(-1), tables.sum(-1), rtol=1e-12)
    assert int(dof.min()) == (qx - 1) * (qy - 1)

    return (
        f"power_divergence {ntab}x{qx}x{qy}",
        _time(lambda: _numpy_power_divergence(tables, "cressie-read"), 3),
        _time(lambda: _lib.power_divergence(tables, "cressie-read", False), 3),
    )


def bench_chi2_sf(count: int = 200_000):
    """Upper chi-square tails, against scipy's own compiled implementation."""
    from scipy.stats import chi2

    x = np.random.default_rng(3).uniform(0.01, 200.0, count)
    want = chi2.sf(x, 7)
    got = _lib.chi2_sf_array(x, 7)
    np.testing.assert_allclose(got, want, rtol=1e-5)

    return (
        f"chi2_sf x{count}",
        _time(lambda: chi2.sf(x, 7), 3),
        _time(lambda: _lib.chi2_sf_array(x, 7), 3),
    )


def bench_chi2_sf_scalar(count: int = 20_000):
    """The same tail probabilities one foreign call at a time."""
    from scipy.stats import chi2

    x = np.random.default_rng(3).uniform(0.01, 200.0, count)
    got = np.array([_lib.chi2_sf(float(v), 7) for v in x])
    np.testing.assert_allclose(got, chi2.sf(x, 7), rtol=1e-5)
    return (
        f"chi2_sf scalar x{count}",
        _time(lambda: chi2.sf(x, 7), 3),
        _time(lambda: np.array([_lib.chi2_sf(float(v), 7) for v in x]), 3),
    )


def bench_ols_rss(n: int = 2_000_000, p: int = 8):
    """Least-squares residual sum of squares, against `numpy.linalg.lstsq`.

    numpy solves the design matrix directly through LAPACK; this kernel forms
    the normal equations and refines once, so it does more flops per unknown and
    far fewer passes over the data. For a handful of parents that trade is worth
    making, and the number below says by how much.
    """
    rng = np.random.default_rng(4)
    X = rng.standard_normal((n, p))
    beta = np.linspace(-2.0, 2.0, p)
    y = X @ beta + rng.standard_normal(n) * 0.5 + 1.0

    def reference():
        design = np.column_stack((np.ones(n), X))
        b, *_ = np.linalg.lstsq(design, y, rcond=None)
        r = y - design @ b
        return float(r @ r)

    got, _beta = _lib.ols_rss(y, X)
    assert got == np.float64(reference()).item() or abs(got - reference()) <= 1e-8 * abs(reference())

    return f"ols_rss n={n} p={p}", _time(reference, 3), _time(lambda: _lib.ols_rss(y, X), 3)


def main():
    print(f"{'case':<28}{'reference':>12}{'mojo-bnlearn':>16}{'ratio':>10}")
    print("-" * 68)
    for fn in (
        bench_contingency,
        bench_contingency_end_to_end,
        bench_contingency_stratified,
        bench_power_divergence,
        bench_chi2_sf,
        bench_chi2_sf_scalar,
        bench_ols_rss,
    ):
        label, ref, got = fn()
        ratio = ref / got if got else float("nan")
        print(f"{label:<28}{ref*1e3:>10.2f}ms{got*1e3:>14.2f}ms{ratio:>9.2f}x")


if __name__ == "__main__":
    main()
