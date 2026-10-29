from core.nested_evaluation import NestedConfig, evaluate_nested
from core.multiple_testing import bonferroni, benjamini_hochberg


def test_nested_selection_does_not_touch_final_holdout():
    records = list(range(30))
    seen = []

    def predictor_a(train, candidates):
        # 记录拟合时可见的最新训练索引，确保不会触及最终 holdout。
        seen.append(max(train))
        return {candidate: 1.0 / len(candidates) for candidate in candidates}

    def predictor_b(train, candidates):
        return {candidate: float(index + 1)
                for index, candidate in enumerate(reversed(candidates))}

    result = evaluate_nested(
        records,
        list(range(4)),
        lambda value: value % 4,
        {"a": predictor_a, "b": predictor_b},
        NestedConfig(10, 5, 1, 5, 3, 5, (1,)),
    )
    assert result.holdout_isolated
    assert all(value < result.holdout_start for value in seen)


def test_multiple_testing():
    adjusted = bonferroni({"a": .01, "b": .20})
    assert adjusted["a"] == .02 and adjusted["b"] == .40
    q_values = benjamini_hochberg({"a": .01, "b": .20})
    assert q_values["a"] <= q_values["b"]
