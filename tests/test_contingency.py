"""Parity of the contingency counting kernel against bnlearn's own route.

bnlearn builds the table with
`data.groupby([X, Y]).size().unstack(Y, fill_value=0)`, per stratum of Z. These
tests rebuild the same tables that way and require exact integer equality:
counting is an exact operation, so any tolerance would hide an indexing bug.
"""

import numpy as np
import pandas as pd
import pytest

import mojo_bnlearn as mbn


def pandas_tables(data, X, Y, Z):
    """The tables bnlearn's CITests actually feeds to scipy, as a list."""
    if not Z:
        return [
            data.groupby([X, Y]).size().unstack(Y, fill_value=0).to_numpy(dtype=np.float64)
        ]
    out = []
    for _state, frame in data.groupby(Z):
        out.append(
            frame.groupby([X, Y]).size().unstack(Y, fill_value=0).to_numpy(dtype=np.float64)
        )
    return out


def drop_empty(table):
    """The sparse table `unstack` produces, with empty margins cut."""
    rows = table.sum(1) > 0
    cols = table.sum(0) > 0
    return table[np.ix_(rows, cols)]


def random_categorical(n, cards, seed):
    rng = np.random.default_rng(seed)
    codes = np.column_stack([rng.integers(0, c, size=n) for c in cards])
    columns = [chr(ord("A") + i) for i in range(codes.shape[1])]
    return pd.DataFrame(codes, columns=columns)


def test_counts_match_pandas_groupby_unconditional():
    data = random_categorical(5000, [3, 4], seed=11)
    got = mbn.contingency(data, "A", "B", []).astype(np.float64)
    want = pandas_tables(data, "A", "B", [])[0]
    np.testing.assert_array_equal(got[0], want)


def test_counts_match_pandas_groupby_conditional():
    data = random_categorical(5000, [3, 4, 2, 3], seed=12)
    counts = mbn.contingency(data, "A", "B", ["C", "D"])
    assert counts.shape == (6, 3, 4)
    assert counts.sum() == len(data)
    for z, table in zip(counts, pandas_tables(data, "A", "B", ["C", "D"])):
        np.testing.assert_array_equal(drop_empty(z), drop_empty(table))


def test_stratum_index_is_mixed_radix_in_declared_order():
    """A and B are not symmetric: swapping them must transpose, not permute.

    A transposed index (qx*qy vs qy*qx) is the most likely bug in a fused
    counting kernel and it is invisible on a square table, so the fixture uses
    a 3 x 5 table and checks both orders.
    """
    data = random_categorical(4000, [3, 5], seed=13)
    ab = mbn.contingency(data, "A", "B", []).astype(np.float64)
    ba = mbn.contingency(data, "B", "A", []).astype(np.float64)
    assert ab.shape == (1, 3, 5)
    assert ba.shape == (1, 5, 3)
    np.testing.assert_array_equal(ab[0].T, ba[0])


def test_stratum_order_follows_z_column_order():
    """Conditioning on C then D must not give the same grid as D then C.

    The stratum index is a mixed-radix integer, so reordering the conditioning
    columns permutes the leading axis. Swapping two columns of unequal
    cardinality changes the grid, which is the check that would fail if the
    kernel silently used row-major instead of mixed-radix indexing.
    """
    data = random_categorical(4000, [2, 3, 2, 4], seed=17)
    cd = mbn.contingency(data, "A", "B", ["C", "D"])
    dc = mbn.contingency(data, "A", "B", ["D", "C"])
    assert cd.shape == (8, 2, 3)
    assert dc.shape == (8, 2, 3)
    assert not np.array_equal(cd, dc)
    # The multiset of tables is the same; only their order differs.
    assert sorted(t.tobytes() for t in cd) == sorted(t.tobytes() for t in dc)


def test_single_stratum_equals_unconditional():
    """One value of Z must not change the pooled table."""
    data = random_categorical(3000, [2, 3, 1], seed=14)
    pooled = mbn.contingency(data, "A", "B", []).astype(np.float64)
    per = mbn.contingency(data, "A", "B", ["C"]).astype(np.float64)
    assert per.shape[0] == 1
    np.testing.assert_array_equal(pooled[0], per[0])


def test_sparse_stratum_leaves_zero_rows():
    """A stratum where X is not fully observed must come back with zero rows."""
    data = random_categorical(2000, [3, 2, 2], seed=15)
    data.loc[data["C"] == 1, "A"] = 0
    counts = mbn.contingency(data, "A", "B", ["C"])
    assert counts.shape == (2, 3, 2)
    assert counts[1][0].sum() > 0
    assert counts[1][1:].sum() == 0
    assert counts.sum() == len(data)


def test_every_row_lands_in_exactly_one_cell():
    data = random_categorical(7777, [2, 2, 2, 2], seed=16)
    counts = mbn.contingency(data, "A", "B", ["C", "D"])
    assert counts.sum() == 7777
    assert (counts >= 0).all()


def test_labels_line_up_with_the_count_axes():
    data = random_categorical(500, [3, 4, 2], seed=18)
    counts, (xs, ys, zs) = mbn.contingency(data, "A", "B", ["C"], return_labels=True)
    assert list(xs) == sorted(data["A"].unique().tolist())
    assert list(ys) == sorted(data["B"].unique().tolist())
    assert list(zs[0]) == sorted(data["C"].unique().tolist())
    assert (counts.shape[0], counts.shape[1], counts.shape[2]) == (len(zs[0]), len(xs), len(ys))


def test_out_of_range_codes_are_rejected():
    """The kernel must not be handed a code it would use as a raw index."""
    from mojo_bnlearn import _lib

    codes = np.array([[0, 1], [1, 0], [5, 1]], dtype=np.int32)
    with pytest.raises(ValueError, match="out of range"):
        _lib.contingency_counts(
            codes, 0, 1, np.zeros(0, np.int32), np.zeros(0, np.int32), 2, 2, 1
        )
