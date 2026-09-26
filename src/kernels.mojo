"""Numeric kernels for the compute-oriented subset of bnlearn.

bnlearn itself is mostly graph plumbing: pandas in, a `pgmpy` model out. The
parts that are genuinely arithmetic live in three places, and those are what
this compilation unit implements:

  * `bnlearn.CITests` builds a contingency table per stratum of Z with
    `data.groupby([X, Y]).size().unstack(...)` and then hands it to
    `scipy.stats.chi2_contingency`. The groupby-size pass over every row is the
    hot loop; `bnl_contingency_counts` does it in one fused traversal.
  * `chi2_contingency` turns observed counts into expected frequencies under
    independence and accumulates a Cressie-Read power divergence. That is
    `bnl_power_divergence`, including scipy's Yates correction and its
    non-finite behaviour on zero-observed cells.
  * `bnlearn.structure_learning.LogLikelihoodGauss` is bnlearn's own scorer: a
    least-squares fit of each node on its parents, scored by residual variance.
    That is `bnl_ols_rss`.
  * The p-values above come from the chi-square survival function, i.e. the
    regularised incomplete gamma function: `bnl_chi2_sf` / `bnl_gammaq`.

Every exported symbol takes buffer addresses as plain `Int` values and rebuilds
the pointer inside the body, because `@export` rejects parametric functions and
an inferred pointer origin would make the symbol parametric.
"""

from std.math import abs, exp, lgamma, log, sqrt

comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime I32Ptr = Pointer[Int32, AnyOrigin[mut=True]]
comptime I64Ptr = Pointer[Int64, AnyOrigin[mut=True]]

from std.math import abs, exp, fma, lgamma, log

# Cressie-Read lambda codes, matching scipy's `_power_div_lambda_names`.
comptime LAMBDA_PEARSON: Int = 0
comptime LAMBDA_LOGLIK: Int = 1
comptime LAMBDA_MODLOGLIK: Int = 2
comptime LAMBDA_NEYMAN: Int = 3
comptime LAMBDA_FREEMAN: Int = 4
comptime LAMBDA_CRESSIE: Int = 5

# Statistic status codes returned in the `flag` buffer.
comptime STAT_FINITE: Int = 0
comptime STAT_POSINF: Int = 1
comptime STAT_NAN: Int = 2

comptime GAMMA_EPS: Float64 = 1.0e-16
comptime GAMMA_FPMIN: Float64 = 1.0e-300
comptime GAMMA_ITMAX: Int = 1000


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def i32p(addr: Int) -> I32Ptr:
    return I32Ptr(unsafe_from_address=addr)


def i64p(addr: Int) -> I64Ptr:
    return I64Ptr(unsafe_from_address=addr)


@export("bnl_contingency_counts")
def bnl_contingency_counts(
    codes_addr: Int,
    nrows: Int,
    ncols: Int,
    xi: Int,
    yi: Int,
    zcols_addr: Int,
    nz: Int,
    zcard_addr: Int,
    qx: Int,
    qy: Int,
    rz: Int,
    counts_addr: Int,
) abi("C"):
    """Stratified X x Y contingency counts over an integer-coded data matrix.

    `codes` is a row-major (nrows, ncols) Int32 buffer of state indices: the
    value of column `c` in row `r` lives at `codes[r * ncols + c]`. Row-major
    is the right layout here because a single row contributes to exactly one
    output cell, so a traversal touches each cache line once instead of
    streaming ncols separate columns.

    `zcols` lists the conditioning columns and `zcard` their cardinalities.
    The stratum index is a mixed-radix integer, most significant column first:
    `z = 0; for k: z = z * zcard[k] + code(zcols[k])`. The output is
    `rz * qx * qy` Int64 counts laid out as `((z * qx) + x) * qy + y`, which
    is exactly the layout the power-divergence kernel reads.
    """
    var codes = i32p(codes_addr)
    var zcols = i32p(zcols_addr)
    var zcard = i32p(zcard_addr)
    var counts = i64p(counts_addr)

    var total = rz * qx * qy
    for k in range(total):
        counts[unsafe_offset=k] = Int64(0)

    for r in range(nrows):
        var row = r * ncols
        var x = codes[unsafe_offset=row + xi]
        var y = codes[unsafe_offset=row + yi]
        var z = 0
        var xi32 = Int(x)
        var yi32 = Int(y)
        for k in range(nz):
            var c = codes[unsafe_offset=row + Int(zcols[unsafe_offset=k])]
            z = z * Int(zcard[unsafe_offset=k]) + Int(c)
        counts[unsafe_offset=((z * qx) + xi32) * qy + yi32] += Int64(1)


@export("bnl_power_divergence")
def bnl_power_divergence(
    obs_addr: Int,
    ntab: Int,
    qx: Int,
    qy: Int,
    lambda_code: Int,
    yates: Int,
    exp_addr: Int,
    scratch_addr: Int,
    stat_addr: Int,
    dof_addr: Int,
    flag_addr: Int,
    t0: Int,
    t1: Int,
) abi("C"):
    """Expected frequencies and the Cressie-Read statistic for a table stack.

    `obs` holds `ntab` dense tables of `qx * qy` observed counts back to back.
    Only tables `[t0, t1)` are processed, so the caller can split a large stack
    across threads; every table writes only its own outputs, which is what makes
    that split safe.
    For every table the kernel derives the expected frequencies under
    independence, `expected[i][j] = rowsum[i] * colsum[j] / total`, and
    accumulates the power divergence with the requested `lambda_code`.

    Three details are inherited from `scipy.stats.chi2_contingency` so that the
    numbers are directly comparable, and each is a place where a plausible
    implementation bug would show up as a different statistic:

      * Degrees of freedom are `(r - 1) * (c - 1)` over the *non-degenerate*
        rows and columns. A table with an all-zero row is one bnlearn's
        `unstack` never produces, and scipy rejects it outright, so this kernel
        skips it: it contributes nothing to the sum, and the row/column is
        dropped from the dof.
      * Yates' correction is applied to the observed counts, never to the
        expected ones, and only when the dof is exactly 1 and the caller asked
        for it. `chi2_contingency` defaults to `correction=True`, so bnlearn
        inherits the correction without saying so.
      * The general Cressie-Read branch is `o * ((o / e) ** lambda - 1) /
        (0.5 * lambda * (lambda + 1))`, which is not defined when `o == 0` and
        `e > 0` for lambda < 0: numpy yields `+inf` for mod-log-likelihood
        (it forms `e * log(e / o)`) and `nan` for neyman and freeman-tukey
        (they form `0 * inf`). The kernel reports that through `flag` instead
        of relying on the compiler's infinity semantics.

    `scratch` is caller-owned space for `qx + qy` floats: the first `qy` hold
    the column sums of the table currently being processed, the next `qx` its
    row sums. It has to be separate from `exp` because the expected frequencies
    are written in place while the marginals are still being read, and it has to
    be per-thread for the same reason.
    """
    var obs = fp(obs_addr)
    var expf = fp(exp_addr)
    var scratch = fp(scratch_addr)
    var stat = fp(stat_addr)
    var dof = i32p(dof_addr)
    var flag = i32p(flag_addr)

    var cells = qx * qy
    var lam = Float64(0.0)
    if lambda_code == LAMBDA_NEYMAN:
        lam = -2.0
    elif lambda_code == LAMBDA_FREEMAN:
        lam = -0.5
    elif lambda_code == LAMBDA_CRESSIE:
        lam = 2.0 / 3.0
    var lam_denom = 0.5 * lam * (lam + 1.0)

    for t in range(t0, t1):
        var base = t * cells
        for j in range(qy):
            scratch[unsafe_offset=j] = Float64(0.0)
        for i in range(qx):
            for j in range(qy):
                scratch[unsafe_offset=j] += obs[unsafe_offset=base + i * qy + j]
        for i in range(qx):
            var s = Float64(0.0)
            for j in range(qy):
                s += obs[unsafe_offset=base + i * qy + j]
            scratch[unsafe_offset=qy + i] = s

        var grand = Float64(0.0)
        var nzr = 0
        for i in range(qx):
            if scratch[unsafe_offset=qy + i] > 0.0:
                nzr += 1
                grand += scratch[unsafe_offset=qy + i]
        var nzc = 0
        for j in range(qy):
            if scratch[unsafe_offset=j] > 0.0:
                nzc += 1

        var d = (nzr - 1) * (nzc - 1)
        if grand <= 0.0 or d <= 0:
            for k in range(cells):
                expf[unsafe_offset=base + k] = Float64(0.0)
            stat[unsafe_offset=t] = Float64(0.0)
            dof[unsafe_offset=t] = Int32(0)
            flag[unsafe_offset=t] = Int32(STAT_FINITE)
            continue
        dof[unsafe_offset=t] = Int32(d)

        var use_yates = yates != 0 and d == 1
        var acc = Float64(0.0)
        var state = Int32(STAT_FINITE)
        for i in range(qx):
            var rs = scratch[unsafe_offset=qy + i]
            if rs <= 0.0:
                for j in range(qy):
                    expf[unsafe_offset=base + i * qy + j] = Float64(0.0)
                continue
            for j in range(qy):
                var cs = scratch[unsafe_offset=j]
                if cs <= 0.0:
                    expf[unsafe_offset=base + i * qy + j] = Float64(0.0)
                    continue
                var o = obs[unsafe_offset=base + i * qy + j]
                var e = rs * cs / grand
                expf[unsafe_offset=base + i * qy + j] = e
                var oo = o
                if use_yates:
                    var diff = e - o
                    var mag = Float64(0.5)
                    if abs(diff) < 0.5:
                        mag = abs(diff)
                    if diff > 0.0:
                        oo = o + mag
                    elif diff < 0.0:
                        oo = o - mag

                if lambda_code == LAMBDA_PEARSON:
                    var dlt = oo - e
                    acc += dlt * dlt / e
                elif lambda_code == LAMBDA_LOGLIK:
                    # 2 * xlogy(o, o / e); xlogy is 0 when its first argument is.
                    if oo > 0.0:
                        acc += 2.0 * oo * log(oo / e)
                elif lambda_code == LAMBDA_MODLOGLIK:
                    # 2 * xlogy(e, e / o): diverges when the cell is unobserved.
                    if oo > 0.0:
                        acc += 2.0 * e * log(e / oo)
                    else:
                        state = Int32(STAT_POSINF)
                elif lambda_code == LAMBDA_CRESSIE:
                    if oo > 0.0:
                        acc += oo * (exp(lam * log(oo / e)) - 1.0) / lam_denom
                    else:
                        acc += Float64(0.0)
                elif lambda_code == LAMBDA_NEYMAN:
                    if oo > 0.0:
                        acc += oo * (exp(lam * log(oo / e)) - 1.0) / lam_denom
                    else:
                        state = Int32(STAT_NAN)
                else:
                    if oo > 0.0:
                        acc += oo * (exp(lam * log(oo / e)) - 1.0) / lam_denom
                    else:
                        state = Int32(STAT_NAN)

        stat[unsafe_offset=t] = acc
        flag[unsafe_offset=t] = state


def _gser(a: Float64, x: Float64) -> Float64:
    """Series expansion of the regularised lower incomplete gamma P(a, x)."""
    var ap = a
    var total = 1.0 / a
    var delta = total
    for _ in range(GAMMA_ITMAX):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * GAMMA_EPS:
            break
    return total * exp(-x + a * log(x) - lgamma(a))


def _gcf(a: Float64, x: Float64) -> Float64:
    """Continued fraction for the regularised upper incomplete gamma Q(a, x)."""
    var b = x + 1.0 - a
    var c = 1.0 / GAMMA_FPMIN
    var d = 1.0 / b
    var h = d
    for i in range(1, GAMMA_ITMAX + 1):
        var fi = Float64(i)
        var an = -fi * (fi - a)
        b += 2.0
        d = an * d + b
        if abs(d) < GAMMA_FPMIN:
            d = GAMMA_FPMIN
        c = b + an / c
        if abs(c) < GAMMA_FPMIN:
            c = GAMMA_FPMIN
        d = 1.0 / d
        var delta = d * c
        h *= delta
        if abs(delta - 1.0) < GAMMA_EPS:
            break
    return exp(-x + a * log(x) - lgamma(a)) * h


def _gammaq(a: Float64, x: Float64) -> Float64:
    """Regularised upper incomplete gamma Q(a, x) = 1 - P(a, x)."""
    if x < a + 1.0:
        return 1.0 - _gser(a, x)
    return _gcf(a, x)


@export("bnl_gammaq")
def bnl_gammaq(a: Float64, x: Float64) abi("C") -> Float64:
    """Regularised upper incomplete gamma, the survival function of Gamma(a)."""
    if x <= 0.0:
        return 1.0
    if a <= 0.0:
        return 0.0
    return _gammaq(a, x)


@export("bnl_chi2_sf")
def bnl_chi2_sf(x: Float64, df: Float64) abi("C") -> Float64:
    """Upper-tail probability of a chi-square variate: P(X > x).

    `chi2_contingency` turns its statistic into a p-value with exactly this
    call, so the Mojo kernel is doing the same work as
    `scipy.stats.chi2.sf(x, df)` and the parity test compares them directly.
    """
    if x <= 0.0:
        return 1.0
    if df <= 0.0:
        return 0.0
    return _gammaq(df * 0.5, x * 0.5)


@export("bnl_chi2_sf_batch")
def bnl_chi2_sf_batch(
    x_addr: Int, n: Int, df: Float64, out_addr: Int
) abi("C"):
    """`bnl_chi2_sf` over a whole array, with the loop inside the kernel.

    One foreign call per statistic costs about a microsecond of ctypes overhead,
    which is two orders of magnitude more than the incomplete gamma itself for
    a typical statistic. Looping here removes that overhead; the arithmetic per
    element is unchanged.
    """
    var xs = fp(x_addr)
    var out = fp(out_addr)
    for i in range(n):
        var x = xs[unsafe_offset=i]
        if x <= 0.0:
            out[unsafe_offset=i] = 1.0
        elif df <= 0.0:
            out[unsafe_offset=i] = 0.0
        else:
            out[unsafe_offset=i] = _gammaq(df * 0.5, x * 0.5)


def _lu_solve(work: FPtr, q: Int, rhs: FPtr) -> Bool:
    """Solve `work * x = rhs` in place by Gaussian elimination with pivoting.

    `work` is destroyed. Returns False when a pivot collapses, which for a
    Gram matrix means the design matrix is rank deficient.
    """
    for k in range(q):
        var piv = k
        var best = abs(work[unsafe_offset=k * q + k])
        for i in range(k + 1, q):
            var m = abs(work[unsafe_offset=i * q + k])
            if m > best:
                best = m
                piv = i
        if best == 0.0:
            return False
        if piv != k:
            for j in range(q):
                var t = work[unsafe_offset=k * q + j]
                work[unsafe_offset=k * q + j] = work[unsafe_offset=piv * q + j]
                work[unsafe_offset=piv * q + j] = t
            var t2 = rhs[unsafe_offset=k]
            rhs[unsafe_offset=k] = rhs[unsafe_offset=piv]
            rhs[unsafe_offset=piv] = t2
        var d = work[unsafe_offset=k * q + k]
        for i in range(k + 1, q):
            var f = work[unsafe_offset=i * q + k] / d
            if f == 0.0:
                continue
            for j in range(k, q):
                work[unsafe_offset=i * q + j] -= f * work[unsafe_offset=k * q + j]
            rhs[unsafe_offset=i] -= f * rhs[unsafe_offset=k]
    for i in range(q - 1, -1, -1):
        var s = rhs[unsafe_offset=i]
        for j in range(i + 1, q):
            s -= work[unsafe_offset=i * q + j] * rhs[unsafe_offset=j]
        rhs[unsafe_offset=i] = s / work[unsafe_offset=i * q + i]
    return True


@export("bnl_ols_rss")
def bnl_ols_rss(
    y_addr: Int,
    x_addr: Int,
    n: Int,
    p: Int,
    gram_addr: Int,
    work_addr: Int,
    rhs_addr: Int,
    resid_addr: Int,
    beta_addr: Int,
    rss_addr: Int,
) abi("C") -> Int:
    """Residual sum of squares of an ordinary least squares fit with intercept.

    This is the arithmetic behind `bnlearn.structure_learning.LogLikelihoodGauss`
    for a node with `p` parents: fit `y` on `[1, X]`, then report
    `sum((y - design @ beta) ** 2)`. With `p == 0` the model is the mean, which
    is the intercept-only branch of that scorer.

    `x` is column-major, `p` columns of `n` values each: column `k` starts at
    `x[k * n]`. The remaining buffers are caller-owned scratch space: `gram`
    and `work` hold `q * q` floats each, `rhs` holds `q`, and `resid` holds
    `n`, where `q` is `p + 1`.

    One step of iterative refinement re-solves against the residual, which
    recovers the accuracy the normal equations give up to `eps * cond(X)**2`.
    Residuals are formed explicitly rather than as `y'y - beta' X'y` so that no
    catastrophic cancellation enters the reported quantity.

    Returns 0 on success and 1 when the design matrix is rank deficient, which
    the caller must not paper over with a different estimator.
    """
    var y = fp(y_addr)
    var x = fp(x_addr)
    var a = fp(gram_addr)
    var work = fp(work_addr)
    var rhs = fp(rhs_addr)
    var resid = fp(resid_addr)
    var beta = fp(beta_addr)
    var rss = fp(rss_addr)

    var q = p + 1

    for i in range(q):
        for j in range(q):
            var acc = Float64(0.0)
            for t in range(n):
                var di = Float64(1.0) if i == 0 else x[unsafe_offset=(i - 1) * n + t]
                var dj = Float64(1.0) if j == 0 else x[unsafe_offset=(j - 1) * n + t]
                acc = fma(di, dj, acc)
            a[unsafe_offset=i * q + j] = acc
        var bi = Float64(0.0)
        for t in range(n):
            var di = Float64(1.0) if i == 0 else x[unsafe_offset=(i - 1) * n + t]
            bi = fma(di, y[unsafe_offset=t], bi)
        rhs[unsafe_offset=i] = bi

    for i in range(q * q):
        work[unsafe_offset=i] = a[unsafe_offset=i]
    if not _lu_solve(work, q, rhs):
        return Int(1)
    for i in range(q):
        beta[unsafe_offset=i] = rhs[unsafe_offset=i]

    # One step of iterative refinement: the correction `db` that solves
    # `A db = X' r` is added to the current coefficients, not substituted for
    # them. The normal equations lose about `eps * cond(X)**2`, and this puts
    # most of that back before the residual is formed.
    for t in range(n):
        var fit = beta[unsafe_offset=0]
        for k in range(1, q):
            fit = fma(x[unsafe_offset=(k - 1) * n + t], beta[unsafe_offset=k], fit)
        resid[unsafe_offset=t] = y[unsafe_offset=t] - fit
    for i in range(q):
        var bi = Float64(0.0)
        for t in range(n):
            var di = Float64(1.0) if i == 0 else x[unsafe_offset=(i - 1) * n + t]
            bi = fma(di, resid[unsafe_offset=t], bi)
        rhs[unsafe_offset=i] = bi
    for i in range(q * q):
        work[unsafe_offset=i] = a[unsafe_offset=i]
    if not _lu_solve(work, q, rhs):
        return Int(1)
    for i in range(q):
        beta[unsafe_offset=i] += rhs[unsafe_offset=i]

    var acc = Float64(0.0)
    for t in range(n):
        var fit = beta[unsafe_offset=0]
        for k in range(1, q):
            fit = fma(x[unsafe_offset=(k - 1) * n + t], beta[unsafe_offset=k], fit)
        var e = y[unsafe_offset=t] - fit
        acc = fma(e, e, acc)
    rss[unsafe_offset=0] = acc

    return Int(0)
