# Python 3.10 package compatibility report v1

## Scope

- **Date:** 2026-08-04
- **Branch:** `fix/python-310-package-compatibility`
- **Motivation:** formally publish the Python 3.10 compatibility already used by
  the CLI fallback path, without changing scientific behaviour.
- **Previous requirement:** `>=3.11`.
- **New requirement:** `>=3.10`.

## Compatibility audit

The source and tests were reviewed for Python 3.11-only syntax and standard
library APIs, including typing, `pathlib`, `dataclasses`, `enum`,
`importlib.metadata`, and `subprocess`. No incompatible production use was
found. `zip(strict=...)` is supported from Python 3.10 and the string APIs in
use are available from Python 3.9.

`tomllib` is the only version-specific import. It remains optional: Python
3.11+ uses it, while Python 3.10 uses the built-in BellLab TOML-subset parser.
No `tomli` dependency was added. The parser supports the public configuration
scope: named and nested tables, `[[recordings]]`, booleans, integers, floats,
UTF-8 strings, simple lists, and relative paths. Tests explicitly simulate the
absence of `tomllib`, load the public example, and cover invalid and unknown
configuration keys.

## Validation

The baseline on Python 3.10.12 was 1398 passing tests, including with warnings
treated as errors. The final suite has 1399 passing tests, including with
warnings treated as errors; `compileall` also succeeds. The normal editable
installation was attempted without `--ignore-requires-python`, but the system
user site cannot replace its pre-existing read-only `belllab` console script.
This is an environment-permission limitation, not a package requirement
failure. The clean wheel installation succeeds from a temporary directory
outside the source tree. Both sdist and wheel were built with Python 3.10 and
the wheel metadata contains `Requires-Python: >=3.10`; its module and console
CLI smoke tests report BellLab 0.13.0 and all schema versions as 1.0.

## Limitations and invariants

The internal parser is intentionally not a universal TOML implementation; it
supports only the documented CLI configuration subset. The environment lacks
`ensurepip` for Python 3.10, so isolated build and wheel virtual environments
could not be created; installing `python3.10-venv` is required before repeating
those exact venv commands. Python 3.11+ continues to use `tomllib`. No
scientific logic, algorithms, parameters, schemas, exports, visualizations,
reports, or package version were changed. No merge, tag, or publication is
part of this round.
