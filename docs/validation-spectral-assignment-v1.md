# Validation Report — Spectral Assignment v1

**Initial suite:** 110 passed. **Final suite:** 114 passed.

The public assignment diagnostic now enforces nonnegative finite costs and
margins, known distance units, exact component decomposition and the operational
margin rule: minimum available row/column margin, or `None` without alternatives.
This makes diagnostic records auditable without retaining cost matrices.

The matching cost gate is inclusive: `selected_cost <= maximum_association_cost`.
The near-limit rule is `selected_cost >= near_threshold_ratio * maximum_cost`.
New runtime tests reject a zero/infinite maximum and ratios outside `[0, 1]`.
Existing quantitative tests cover 1 Hz, symmetric relative distance, 100-cent
semitone and 1200-cent octave, as well as public decomposition equality.

The algorithm remains deterministic and one-to-one; crossing identity is not
claimed to be physically resolved. Further complete matching tests for changing
`amplitude_weight` and real-recording crossings remain the recommended next
validation step. No modal candidate or physical mode was introduced.
