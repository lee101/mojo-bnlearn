"""ctypes bridge to the compiled Mojo kernels.

The shared library owns no memory. Every buffer crosses the C ABI as a 64-bit
address, so the argtypes below must stay `c_int64` for addresses; `c_int`
truncates them and segfaults.
"""

import ctypes
import os
import pathlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-bnlearn.so"

# scipy's `_power_div_lambda_names`, flattened to the integer codes the kernel
# switches on.
LAMBDA_CODES = {
    "pearson": 0,
    "log-likelihood": 1,
    "mod-log-likelihood": 2,
    "neyman": 3,
    "freeman-tukey": 4,
    "cressie-read": 5,
}

STAT_FINITE = 0
STAT_POSINF = 1
STAT_NAN = 2

I32 = ctypes.c_int32
I64 = ctypes.c_int64
F64 = ctypes.c_double
ADDR = ctypes.c_int64


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found; run `bash build/build.sh` first"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.bnl_contingency_counts.restype = None
    lib.bnl_contingency_counts.argtypes = [
        ADDR, I64, I64, I64, I64, ADDR, I64, ADDR, I64, I64, I64, ADDR,
    ]
    lib.bnl_power_divergence.restype = None
    lib.bnl_power_divergence.argtypes = [
        ADDR, I64, I64, I64, I64, I64, ADDR, ADDR, ADDR, ADDR, ADDR, I64, I64,
    ]
    lib.bnl_chi2_sf_batch.restype = None
    lib.bnl_chi2_sf_batch.argtypes = [ADDR, I64, F64, ADDR]
    lib.bnl_gammaq.restype = F64
    lib.bnl_gammaq.argtypes = [F64, F64]
    lib.bnl_chi2_sf.restype = F64
    lib.bnl_chi2_sf.argtypes = [F64, F64]
    lib.bnl_ols_rss.restype = I64
    lib.bnl_ols_rss.argtypes = [ADDR, ADDR, I64, I64, ADDR, ADDR, ADDR, ADDR, ADDR, ADDR]
    return lib


lib = _load()


def _addr(a: np.ndarray) -> int:
    return a.ctypes.data


def _workers(ntab: int, cells: int) -> int:
    """How many threads to split a table stack across.

    The crossover is around two flops per byte; below it, splitting costs more
    in cache traffic than it saves. A single small table stays serial.
    """
    if ntab < 8 or cells < 64:
        return 1
    return max(1, min(8, os.cpu_count() or 1, ntab // 8))


def contingency_counts(codes, xi, yi, zcols, zcard, qx, qy, rz):
    """Stratified X x Y counts over a row-major integer-coded matrix.

    `codes` has shape (nrows, ncols) and holds state indices. Returns Int64
    counts of shape (rz, qx, qy).
    """
    codes = np.ascontiguousarray(codes, dtype=np.int32)
    if codes.ndim != 2:
        raise ValueError("codes must be a 2-D (nrows, ncols) array")
    zcols = np.ascontiguousarray(zcols, dtype=np.int32)
    zcard = np.ascontiguousarray(zcard, dtype=np.int32)
    nrows, ncols = codes.shape
    if codes.size and (codes.min() < 0 or codes.max() >= max(qx, qy, 1)):
        # Codes outside the declared cardinalities would write out of bounds,
        # so reject rather than corrupt the heap.
        if codes[:, xi].min() < 0 or codes[:, xi].max() >= qx:
            raise ValueError("X codes out of range for the declared cardinality")
        if codes[:, yi].min() < 0 or codes[:, yi].max() >= qy:
            raise ValueError("Y codes out of range for the declared cardinality")
        for c, card in zip(zcols.tolist(), zcard.tolist()):
            col = codes[:, c]
            if col.size and (col.min() < 0 or col.max() >= card):
                raise ValueError("Z codes out of range for the declared cardinality")
    counts = np.zeros(rz * qx * qy, dtype=np.int64)
    lib.bnl_contingency_counts(
        _addr(codes), nrows, ncols, xi, yi,
        _addr(zcols), len(zcols), _addr(zcard),
        qx, qy, rz, _addr(counts),
    )
    return counts.reshape(rz, qx, qy)


def power_divergence(obs, lambda_="pearson", yates=True):
    """Expected frequencies and the Cressie-Read statistic for a table stack.

    `obs` has shape (ntab, qx, qy) or (qx, qy). Returns
    `(statistic, dof, expected, pvalue)` with the same shapes as the input for
    `statistic`/`expected`; `dof` and `pvalue` are per table.
    """
    if lambda_ not in LAMBDA_CODES:
        raise ValueError(f"unknown lambda_ {lambda_!r}; expected one of {sorted(LAMBDA_CODES)}")
    obs = np.ascontiguousarray(obs, dtype=np.float64)
    if obs.ndim == 2:
        obs = obs[None, :, :]
    if obs.ndim != 3:
        raise ValueError("obs must be (qx, qy) or (ntab, qx, qy)")
    if obs.size and obs.min() < 0:
        raise ValueError("observed counts must be non-negative")
    ntab, qx, qy = obs.shape
    expected = np.zeros(ntab * qx * qy, dtype=np.float64)
    stat = np.zeros(ntab, dtype=np.float64)
    dof = np.zeros(ntab, dtype=np.int32)
    flag = np.zeros(ntab, dtype=np.int32)
    # The cell loop is compute bound (one log or exp per cell), so a large stack
    # is worth splitting. Each table writes only its own outputs, and every
    # thread gets its own marginal scratch.
    workers = _workers(ntab, qx * qy)
    scratch = np.zeros((workers, qx + qy), dtype=np.float64)
    step = -(-ntab // workers) if workers else ntab

    def run(part):
        t0 = min(part * step, ntab)
        t1 = min(t0 + step, ntab)
        if t0 >= t1:
            return
        lib.bnl_power_divergence(
            _addr(obs), ntab, qx, qy, LAMBDA_CODES[lambda_], 1 if yates else 0,
            _addr(expected), _addr(scratch[part]), _addr(stat), _addr(dof),
            _addr(flag), t0, t1,
        )

    if workers == 1:
        run(0)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(run, range(workers)))
    # The kernel reports a non-finite term through `flag` rather than relying on
    # the compiler's infinity semantics, so mirror numpy's own outcome here.
    stat = stat.copy()
    stat[flag == STAT_POSINF] = np.inf
    stat[flag == STAT_NAN] = np.nan
    pvalue = np.array([chi2_sf(s, int(d)) for s, d in zip(stat, dof)], dtype=np.float64)
    return stat, dof, expected.reshape(ntab, qx, qy), pvalue


def gammaq(a: float, x: float) -> float:
    return float(lib.bnl_gammaq(float(a), float(x)))


def chi2_sf(x: float, df: int) -> float:
    """Upper tail of a chi-square variate.

    The kernel evaluates `Q(df / 2, x / 2)`, which is only defined for a finite
    non-negative `x`. The non-finite cases are what the statistic itself can
    produce, so they are resolved here rather than inside the series: an
    infinite statistic has no chance of exceeding the threshold, and a NaN
    statistic has no defined tail.
    """
    x = float(x)
    if x != x:
        return float("nan")
    if x == float("inf"):
        return 0.0
    if x <= 0.0:
        return 1.0
    return float(lib.bnl_chi2_sf(x, float(df)))


def chi2_sf_array(x, df) -> np.ndarray:
    flat_x = np.ascontiguousarray(x, dtype=np.float64).reshape(-1)
    flat_out = np.empty(flat_x.size, dtype=np.float64)
    if flat_x.size:
        lib.bnl_chi2_sf_batch(_addr(flat_x), flat_x.size, float(df), _addr(flat_out))
    return flat_out.reshape(np.shape(x))


def ols_rss(y, X):
    """Residual sum of squares and coefficients of an OLS fit with intercept.

    `X` is (n, p); `p == 0` fits the mean. Returns `(rss, beta)` where `beta`
    has length `p + 1` and starts with the intercept.
    """
    y = np.ascontiguousarray(y, dtype=np.float64).reshape(-1)
    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2:
        raise ValueError("X must be (n, p)")
    n, p = X.shape
    if n != y.size:
        raise ValueError("X and y must have the same number of rows")
    # The kernel reads X column-major.
    xc = np.ascontiguousarray(X.T, dtype=np.float64)
    q = p + 1
    gram = np.zeros(q * q, dtype=np.float64)
    work = np.zeros(q * q, dtype=np.float64)
    rhs = np.zeros(q, dtype=np.float64)
    resid = np.zeros(max(n, 1), dtype=np.float64)
    beta = np.zeros(q, dtype=np.float64)
    rss = np.zeros(1, dtype=np.float64)
    status = lib.bnl_ols_rss(
        _addr(y), _addr(xc), n, p,
        _addr(gram), _addr(work), _addr(rhs), _addr(resid),
        _addr(beta), _addr(rss),
    )
    if status != 0:
        raise np.linalg.LinAlgError(
            "OLS design matrix is rank deficient; the Gaussian elimination "
            "solver cannot produce a least squares fit"
        )
    return float(rss[0]), beta
