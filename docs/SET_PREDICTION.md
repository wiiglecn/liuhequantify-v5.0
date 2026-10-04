# V5.8.2 K-Specific Meta Policy

V5.8.2 upgrades V5.8.1 from a shared set optimizer into a genuinely target-specific meta-policy.

## K-specific policy

Each target is optimized independently:
- Zodiac Top-3
- Zodiac Top-4
- Zodiac Top-6
- Wide Top-20

Each K independently selects historical horizon, signal weights, Pair lambda, Diversity lambda, and whether structural terms should be enabled.

## Anti-overfitting gate

Every structural candidate is compared against the same K-specific marginal baseline on an inner validation tail.

A structural candidate is enabled only when validation_gain >= min_gain. Otherwise the policy falls back to lambda_pair=0 and lambda_diversity=0.

## Leakage control

For each target fold, policy training uses only historical folds; horizon selection uses an earlier inner validation tail; Pair structure for each validation row uses only observations before that row; final holdout policy is fitted only on pre-holdout history; the final 60 folds are frozen.

## Research boundary

This remains historical OOS research. A positive historical gain does not establish future predictability for an independent random draw process.
