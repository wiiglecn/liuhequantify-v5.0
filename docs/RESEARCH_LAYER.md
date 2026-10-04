# V5.4 Research Layer

V5.4 extends V5.3 nested OOS evaluation into a reusable quantitative research layer.

## Rules

1. Rank IC, hit@K, information gain, Brier and LogLoss are computed on OOS observations.
2. Calibration parameters are fitted only on training or inner-OOS data, then frozen.
3. Redundancy is measured before ensemble weighting.
4. Ensemble weights are policy parameters and must be learned inside inner OOS.
5. The final holdout remains sealed: no metric, threshold, temperature or weight is selected from it.
6. Multiple experiments continue to use the V5.3 Bonferroni/BH correction utilities.

## Modules

- core/signal_metrics.py: rank IC, hit@K, LogLoss, Brier, information gain, ECE and reliability curve.
- core/calibration.py: dependency-free temperature scaling.
- core/signal_analysis.py: correlation, redundancy groups and residual information.
- core/ensemble_weighting.py: simplex-constrained ensemble combination and inner-OOS weight fitting.

Pipeline:

signal -> inner OOS metrics -> calibration -> redundancy analysis -> weight learning -> locked policy -> outer OOS -> sealed holdout

The legacy ic_ir_weighting.py remains for compatibility. New research code should use the V5.4 core modules so evaluation and leakage control share one boundary.

For Mark Six or lottery-like data, these metrics describe historical statistical behavior; they do not establish guaranteed future predictability.
