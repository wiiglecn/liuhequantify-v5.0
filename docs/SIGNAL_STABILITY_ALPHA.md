# V5.5 Signal Stability & Alpha Discovery

V5.5 consumes point-in-time OOS rows only. It does not fit predictors.

## Metrics
- Rolling Hit@K: 30/60/120/180/300
- Period stability
- Signal decay
- Residual alpha correlation
- Signal alpha matrix
- Information Gain
- Stability classification

## Interpretation
"stable", "decaying", and "unstable" are descriptive labels based on the configured
rolling-window heuristics. They are not claims of future predictability.

## Leakage rule
All V5.5 diagnostics must be generated from OOS predictions produced without
using the corresponding actual outcome during prediction.
