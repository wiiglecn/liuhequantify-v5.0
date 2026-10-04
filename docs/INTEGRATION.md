# V5.2 OOS + Signal Registry

OOS rule: for target index i, predictors receive only records[:i]. Signal selection must happen inside that prefix. Never select active signals once on the complete evaluation period and reuse them across folds.

Registry families: zodiac(freq,markov,cross_dim,recent,bayes,numcount), wide(freq,gap,markov,recent,zodiac,uniform), dimension(freq,decay,markov,recent,gap,prior). LLM remains non-replayable unless historical prompts/outputs are persisted.

For rank-only ensembles hit@K is primary. Rank-derived log-loss/Brier are comparison metrics only and are not calibrated probabilities.

V5.3: nested OOS signal selection + untouched final holdout + experiment registry integration.
