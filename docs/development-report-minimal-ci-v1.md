# Minimal continuous integration report v1

- **Objective:** add a small GitHub Actions regression gate without external services.
- **Branch:** `feature/minimal-ci`.
- **Baseline:** BellLab 0.13.0, Python requirement `>=3.10`, 1399 passing tests.
- **Python matrix:** 3.10 and 3.11 on Ubuntu.

## Workflow

`Continuous integration` runs on pushes and pull requests to `main`, and can
be started manually. It has read-only repository permission, cancels obsolete
executions for the same ref, and has a 20-minute timeout.

Each matrix run installs BellLab with its existing `dev` extra, runs `pytest`
and `pytest -W error`, compiles modules, builds sdist and wheel, checks wheel
metadata, and smoke-tests the wheel-installed CLI from `/tmp`.

The metadata checker is a standard-library script. It compares the wheel name,
version, and Python requirement to `pyproject.toml`, avoiding a duplicated
release version in the workflow.

## Scope and local validation

No publication, release, artifact upload, external service, coverage, lint,
type checking, containers, or additional operating systems are included.
Local validation runs the full suite, warnings-as-errors suite, compileall,
build, metadata check, and wheel smoke test when `venv` is available. No
scientific logic was changed.
