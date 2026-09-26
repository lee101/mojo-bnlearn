"""mojo-bnlearn: bnlearn's numeric core on compiled Mojo kernels.

Installable alongside the real `bnlearn` package, which it is tested against
for parity. Only the arithmetic is ported; the DAG plumbing stays in bnlearn.
"""

from .core import (
    CI_TESTS,
    chi_square,
    contingency,
    cressie_read,
    encode,
    freeman_tuckey,
    g_sq,
    gauss_local_rss,
    gauss_local_score,
    log_likelihood,
    modified_log_likelihood,
    neyman,
    power_divergence,
)

__all__ = [
    "CI_TESTS",
    "chi_square",
    "contingency",
    "cressie_read",
    "encode",
    "freeman_tuckey",
    "g_sq",
    "gauss_local_rss",
    "gauss_local_score",
    "log_likelihood",
    "modified_log_likelihood",
    "neyman",
    "power_divergence",
]
__version__ = "0.1.0"
