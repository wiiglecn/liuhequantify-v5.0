# V5.4.3 Research Result Engine

## Purpose

V5.4.3 does not change prediction logic. It converts V5.4.2 nested OOS output into an auditable research result layer.

## Outputs

Each layer can expose:
- aggregate OOS LogLoss / Brier / ECE / Information Gain
- Hit@K
- deterministic bootstrap 95% confidence intervals
- per-signal outer OOS metrics
- BH and Bonferroni multiple-testing corrections
- final holdout metrics kept separate from selection evidence
- frozen final policy and its learned signal weights/temperature

## Leakage contract

1. Signal selection is performed only on inner OOS.
2. Ensemble weights are learned only on inner OOS.
3. Temperature is fitted only on inner OOS.
4. Outer observations are scored with the resulting frozen policy.
5. Final holdout is scored with one policy fitted from pre-holdout OOS only.
6. Final holdout metrics are never used to choose signals, weights, or calibration.
7. Confidence intervals describe uncertainty of historical OOS metrics; they do not establish future predictability.

## Multiple testing

Per-signal exploratory p-values can be passed through:
- raw p-value
- Benjamini-Hochberg q-value
- Bonferroni-adjusted p-value

These are research diagnostics, not guarantees of a real predictive edge.

## Run

From repository root:

python v54_2_nested_research_run.py --input macau_history.json --output reports/v5.4.3_research_results.json

The runner uses the local historical dataset and does not refresh from the network.

## Next

V5.4.4 should add experiment registry, immutable run manifests, and CSV/Markdown publication tables so every research run is reproducible and comparable.
