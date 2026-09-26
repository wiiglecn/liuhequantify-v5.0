# V5.7 Regime × Signal × Horizon

## Objective

V5.6 showed that regime-conditioned dynamic weighting alone did not produce a stable long-window improvement over V5.0. V5.7 changes the unit of adaptation from one adaptive ensemble to a target-specific meta-policy.

The policy conditions on Regime, Signal and Horizon.

## Target-specific optimization

- Zodiac Top-3 (三肖)
- Zodiac Top-4 (四肖)
- Zodiac Top-6 (六肖)
- Wide Top-20 (20码)

Each target gets its own selected horizon and signal weights.

## Leakage controls

For target fold t:
- regime detector is fitted only on folds before t;
- horizon candidates are scored using an inner validation tail contained entirely in historical folds before t;
- selected policy is refit on the selected historical window, still entirely before t;
- final holdout remains frozen and is not used for policy selection;
- target-fold actual outcome is never used in policy decisions.

## Evaluation

The report includes outer OOS Hit@K, frozen holdout Hit@K, random-set baseline, binomial diagnostic p-value, bootstrap confidence intervals, regime transition matrix/persistence, selected horizon path, and final target-specific weights.

## Interpretation

V5.7 is a research upgrade, not a claim of future predictability. Success requires improvement to survive both long-window OOS evaluation and the final frozen holdout.
