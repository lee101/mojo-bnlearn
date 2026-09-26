import pathlib
import sys
import types

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))

_LIB = _ROOT / "dist" / "libmojo-bnlearn.so"

if not _LIB.exists():
    pytest.skip(
        "libmojo-bnlearn.so not built; run `bash build/build.sh`",
        allow_module_level=True,
    )


def _install_cgi_shim():
    """Make `bnlearn` importable on Python 3.13.

    `bnlearn` -> `datazets` -> `requests`, and this `requests` build still
    imports `cgi`, which was removed from the standard library in 3.13. The
    only symbol it touches is `cgi.parse_header`, so a five-line stand-in is
    enough to reach bnlearn's real code. Without this the parity tests cannot
    run at all, and the alternative would be to skip them.
    """
    if "cgi" in sys.modules:
        return
    try:
        import cgi  # noqa: F401
    except ModuleNotFoundError:
        pass
    else:
        return

    module = types.ModuleType("cgi")

    def parse_header(line):
        parts = line.split(";")
        key = parts[0].strip()
        params = {}
        for part in parts[1:]:
            if "=" in part:
                k, v = part.split("=", 1)
                params[k.strip().lower()] = v.strip().strip('"')
        return key, params

    module.parse_header = parse_header
    module.parse_qs = lambda query, **kw: {}
    module.parse_qsl = lambda query, **kw: []
    module.escape = lambda value, **kw: value
    module.FieldStorage = object
    sys.modules["cgi"] = module


_install_cgi_shim()
