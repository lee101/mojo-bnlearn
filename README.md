# mojo-bnlearn

`mojo-bnlearn` is the compute-oriented subset of
[bnlearn](https://github.com/erdogant/bnlearn) with the arithmetic behind its
conditional-independence tests and its Gaussian structure score implemented in
Mojo and callable from Python.

bnlearn is a Bayesian-network library, and most of what it does is graph
plumbing: it inspects a DataFrame, decides whether the data are discrete or
continuous, and hands the modelling to `pgmpy`. There is no reason to port
that. What bnlearn does compute itself lives in three places, and those are
what this port covers.

The Python package is `mojo_bnlearn`, so it installs alongside the real
`bnlearn` and the tests import both and compare them directly.

```python
import pandas as pd
import mojo_bnlearn as mbn

df = pd.DataFrame({"A": [0, 1, 0, 1, 1, 0], "B": [0, 0, 1, 1, 0, 1]})
mbn.chi_square("A", "B", [], df, boolean=False)   # (0.666..., 0.414..., 1)
mbn.contingency(df, "A", "B", []).shape           # (1, 2, 2)
mbn.gauss_local_score(y, X)                       # -0.5 n (log 2pi + 1 + log var)
```

## Why this subset

| bnlearn surface | what it computes | why it is here |
| --- | --- | --- |
| `CITests.chi_square` / `g_sq` / `log_likelihood` / `freeman_tuckey` / `modified_log_likelihood` / `neyman` / `cressie_read` / `power_divergence` | `data.groupby([X, Y]).size().unstack(...)` per stratum of Z, then `scipy.stats.chi2_contingency(..., lambda_=...)` | the grouped size is a full pass over the data and the statistic is a per-cell transcendental sum. Both are loops; both move. |
| the p-value those tests return | `1 - stats.chi2.cdf(chi, dof)`, i.e. a regularised upper incomplete gamma | a real numeric kernel with a well defined reference |
| `structure_learning.LogLikelihoodGauss` / `AICGauss` / `BICGauss` | `numpy.linalg.lstsq` of a node on its parents, scored by residual variance | this is bnlearn's own scorer, not pgmpy's, and it is one fused pass over the samples |
| `structure_learning.fit` (`hc`, `ex`, `pc`, `nb`, `tan`, `cl`, lingam) | search, plotting, graph objects | control flow and `pgmpy` calls, not arithmetic |
| `parameter_learning.fit` | builds a `pgmpy` estimator with a chosen prior and calls `BayesianEstimator.fit` | the counting that would be a kernel lives in `pgmpy.state_counts`, not in bnlearn |
| `confmatrix`, `discretize`, `impute`, `sampling`, `plot`, `skills_install` | metrics, binning, KNN/MICE imputation, `matplotlib` | scikit-learn and pandas wrappers around small tables |
| `CITests.pearsonr` | `lstsq` on the conditioning set, then correlation of residuals | bnlearn's own code, but a single small least-squares solve, which `ols_rss` already covers as a building block |

## Covered subset

| area | implemented API |
| --- | --- |
| Contingency counting | `contingency(data, X, Y, Z)` (and `encode`) → `(rz, qx, qy)` counts; the kernel also accepts raw pre-coded matrices |
| Power divergence | `power_divergence`, `chi_square`, `g_sq`, `log_likelihood`, `freeman_tuckey`, `modified_log_likelihood`, `neyman`, `cressie_read`, all six Cressie-Read lambdas, Yates' correction, per-table expected frequencies and dof |
| Tail probabilities | `chi2_sf`, `chi2_sf_array`, and the underlying `gammaq` |
| Gaussian score | `gauss_local_rss`, `gauss_local_score` (bnlearn's `LogLikelihoodGauss` formula, plus the AIC/BIC offsets) |
| Kernels | `bnl_contingency_counts`, `bnl_power_divergence` (table range, so the shim can thread it), `bnl_chi2_sf`, `bnl_chi2_sf_batch`, `bnl_gammaq`, `bnl_ols_rss` |

Not implemented: everything in the "why this subset" table above that is
graph plumbing, `pgmpy` estimator plumbing, imputation, plotting, and
`CITests.pearsonr` as a named entry point (its least squares is `ols_rss`;
call that directly if you need it). Use the real `bnlearn` for the rest.

### Conventions that had to be pinned down

Three things bnlearn leaves to `pandas` are explicit here, because the kernel
indexes states directly:

* **State order is ascending value order**, matching `groupby`'s sorted group
  keys, so state `i` means the same thing as the `i`-th row of the `unstack`
  table. `contingency(..., return_labels=True)` hands back the sorted values.
* **The conditioning grid is dense.** bnlearn's `unstack` only emits the states
  a stratum actually contains; this port builds the full `rz x qx x qy` grid and
  lets the statistic drop the empty rows and columns, which changes neither the
  statistic nor the degrees of freedom. This is the same thing bnlearn gets for
  free, because `scipy.stats.chi2_contingency` refuses a table whose expected
  frequencies contain a zero.
* **Strata are mixed-radix in the declared order of Z**, most significant
  first, so conditioning on `["C", "D"]` and on `["D", "C"]` give the same set
  of tables in a different order.

### Non-finite statistics are reproduced, not smoothed over

A cell with zero observed but positive expected breaks three of the six
lambdas: `mod-log-likelihood` becomes `+inf` and `neyman` and `freeman-tukey`
become `nan`, because scipy forms `e * log(e / o)` and `0 * inf` respectively.
The kernel reports that through a status flag rather than manufacturing an
infinity, and the shim turns the flag back into `inf`/`nan` so the numbers
match `bnlearn.CITests` exactly, including its `nan` p-values.

## Install

The repository pins its own Mojo toolchain:

```bash
bash build/build.sh          # -> dist/libmojo-bnlearn.so
PYTHONPATH=python python -m pytest tests -q
```

Set `PYTHONPATH=python` when using the package outside a Pixi task. The
parity tests need the real `bnlearn`; the `cgi` shim in `tests/conftest.py`
exists because `bnlearn` reaches `requests` through `datazets`, and that
`requests` build imports the `cgi` module that Python 3.13 removed.

## How it works

All kernels live in `src/kernels.mojo`, one compilation unit, because shared
library build cost is largely fixed. `build/build.sh` compiles it with
`mojo build --emit shared-lib` into `dist/libmojo-bnlearn.so`.

The Python layer owns every array, factorises the columns, and decides how many
threads to use. Buffers cross the C ABI as 64-bit addresses and are
reconstructed in Mojo as `Pointer[Float64, AnyOrigin[mut=True]]`, which keeps
the exported symbols non-parametric.

`bnl_contingency_counts` is a plain serial loop: it is a scatter-add over a
row-major int32 buffer, which is memory bound, and threading it is a
pessimisation. `bnl_power_divergence` is compute bound (one `log` or `exp` per
cell), so it takes a table range and the shim fans a large stack across a
`ThreadPoolExecutor`, one call per chunk with per-thread marginal scratch.
`bnl_chi2_sf_batch` loops inside the kernel because a foreign call per
statistic costs about a microsecond of ctypes overhead, which dwarfs the
incomplete gamma itself.

Mojo emits FMA and its `log` differs from glibc's by up to 3.2e-9 relative, so
nothing here is bit-identical to scipy. The parity tests state their tolerance
per operation: exact equality for the integer counts, the degrees of freedom
and the expected-frequency table; 1e-12 for Pearson's statistic, which is pure
arithmetic; `1e-7 * df` for the p-values, whose error grows with `df` because
the incomplete gamma's exponential prefactor multiplies an absolute error of
`a * |log x|`.

## Performance

Best-of-three wall clock on a shared 36-core box, against the fastest
reasonable pandas/NumPy/scipy formulation in each case. Every case verifies
numerical agreement before timing. Run-to-run variation on this machine is
large; the figures below are from a single run and are indicative, not precise.

| case | reference | mojo-bnlearn | result |
| --- | ---: | ---: | ---: |
| contingency kernel, n=2000000, 2x3 | 289 ms | 87 ms | 3.3x faster |
| contingency end to end, n=2000000 | 606 ms | 160 ms | 3.8x faster |
| contingency stratified on C,D, n=1000000 | 766 ms | 298 ms | 2.6x faster |
| power divergence, 200000 x 20 x 20 | 13790 ms | 3871 ms | 3.6x faster |
| `chi2_sf`, 200000 tails | 45 ms | 80 ms | 0.56x, slower |
| `chi2_sf`, 20000 scalar calls | 8 ms | 132 ms | 0.06x, slower |
| `ols_rss`, n=2000000, p=8 | 3194 ms | 1100 ms | 2.9x faster |

The counting and statistic kernels win because both are real loops that pandas
and NumPy express as multiple passes plus Python-level dispatch, and both are
large enough to amortise the call. The p-value rows are losses and are
reported as losses: the incomplete gamma is a short, branchy scalar recurrence
that neither vectorises nor spends enough time in arithmetic to beat scipy's
own C implementation once it is called in a batch, and the scalar row loses
almost entirely to ctypes marshalling. If a workload needs millions of
p-values, `scipy.stats.chi2.sf` is the right tool and this kernel exists for
parity and for use inside the CI tests, where there is one call per test.

Reproduce with:

```bash
python bench/bench.py
```

## Tests

80 tests, all against the real `bnlearn`, `scipy` or an analytic reference.

## License

MIT
