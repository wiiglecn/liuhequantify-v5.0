# V5.6 Regime Detection / Adaptive Ensemble

V5.6 adds two leakage-safe research components.

- Regime detection uses only point-in-time signal probability shape: entropy, top-score concentration, spread, inter-signal disagreement and correlation.
- For target fold t, regime detection is fit only on folds before t.
- Adaptive weights use historical OOS LogLoss inside the detected regime, shrunk toward global OOS weights.
- Minimum signal weight prevents collapse to one signal.
- The final 60 folds are a frozen holdout: one policy is fit before the holdout and is not updated during holdout scoring.
- Bootstrap confidence intervals and baseline comparisons remain part of the evaluation layer.

V5.6 is a historical research engine, not evidence of future predictability.
