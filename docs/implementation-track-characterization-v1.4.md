# Implementation Note — Track Characterization v1.4

**Initial suite:** 106 passed.  
**Final suite:** 106 passed.

`TrackAmplitudeFit` is now the canonical amplitude-adjustment object embedded
in `SpectralTrackCharacterization`. Legacy read-only properties (`decay_method`,
`decay_tau_s`, `decay_slope`, `decay_r_squared` and point counts) forward to
that object; adjustment values are no longer stored twice.

Amplitude processing first selects finite time/amplitude pairs. Linear fits then
also require strictly positive amplitude; nonpositive values are never replaced
by epsilon or absolute value. The fit uses at least two valid points. Linear
amplitude uses log-amplitude and `tau=-1/slope`; dBFS uses levels and
`tau=-20/(slope*ln(10))`. Tau is only present for a negative slope. Constant or
increasing amplitude keeps a successful regression but reports no tau.

The fit records domain, unit, RMSE, R² when defined, available/finite/used and
discarded counts, timing and structured failure information. Nonfinite dBFS
values, including silence `-inf`, are excluded before fitting. Future work is a
dedicated validation round for all nonfinite and sparse-data scenarios; no
modal candidate or physical modal interpretation was introduced.
