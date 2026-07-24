# Closure Note — Track Amplitude Fit Contract v1

**Data:** 2026-07-23. **Initial suite:** 134 passed. **Final suite:** 137 passed.

Changed `types.py` and `tests/test_tracking.py`. Failed fits now require every
regression-result field, including `r_squared`, to be `None`; counts,
diagnostics, amplitude unit and failure reason remain available. `r_squared=None`
is the documented representation of an undefined statistic.

All six legacy aliases were exercised against successful and failed canonical fits:
method maps to the legacy name, while tau, slope, R² and point counts read
directly from `amplitude_fit`. Optional amplitude mean/min annotations now
reflect the no-finite-points outcome. The suite passes normally and with
warnings treated as errors. No static tool is configured in `pyproject.toml`.

No tracking, modal, frequency-metric or association behavior changed. Remaining
work belongs to the separate behavioural validation stages, not this contract
closure.
