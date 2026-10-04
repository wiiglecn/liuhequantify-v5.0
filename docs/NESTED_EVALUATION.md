V5.3 Nested Walk-Forward

Goal: prevent tuning, backtesting, and retuning from reusing outer test outcomes.

Three boundaries:
1. Inner OOS: compare candidate signal/model/hyperparameter policies; selection only.
2. Outer OOS: evaluate the locked policy on future observations; outer results never return to selection.
3. Final Holdout: the latest time segment is sealed and cannot be used for tuning, threshold selection, model comparison, or feature selection during development.

Rules:
- Signal selection, GBDT hyperparameters, stacking window, temperature and thresholds are policy parameters.
- Policy selection is based only on inner OOS.
- Outer OOS is reporting data, not tuning data.
- Final holdout remains sealed.
- Multiple experiments require multiple-testing correction; report Bonferroni/BH adjusted values.
- Ranking-derived LogLoss/Brier are not calibrated probabilities unless independently calibrated.

Recommended default split: final holdout 60 periods; outer initial train 180; inner initial train 90; inner OOS 30; outer step 1.

Next: V5.4 IC/IR, calibration, ensemble weight learning and prediction-store data flywheel.
