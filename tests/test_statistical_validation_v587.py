import unittest

from core.statistical_validation_v587 import (
    benjamini_hochberg,
    binomial_upper_tail,
    holm_adjust,
    max_stat_adjusted_p_value,
    moving_block_bootstrap_ci,
    simulate_max_stat_null,
)


class TestStatisticalValidationV587(unittest.TestCase):
    def test_binomial_upper_tail_matches_known_cases(self):
        self.assertEqual(binomial_upper_tail(0, 10, 0.25), 1.0)
        self.assertAlmostEqual(binomial_upper_tail(10, 10, 0.5), 1 / 1024)
        self.assertEqual(binomial_upper_tail(1, 10, 0.0), 0.0)

    def test_p_value_adjustments_preserve_order_and_monotonicity(self):
        values = [0.01, 0.04, 0.03, 0.8]
        holm = holm_adjust(values)
        bh = benjamini_hochberg(values)
        self.assertEqual(len(holm), len(values))
        self.assertEqual(len(bh), len(values))
        self.assertTrue(all(0 <= p <= 1 for p in holm + bh))
        self.assertGreaterEqual(holm[0], values[0])
        self.assertGreaterEqual(bh[0], values[0])

    def test_invalid_p_values_rejected(self):
        with self.assertRaises(ValueError):
            holm_adjust([0.2, 1.1])

    def test_block_bootstrap_is_reproducible_and_paired(self):
        new = [1, 1, 0, 1, 0, 1, 1, 0, 1, 1]
        old = [0, 1, 0, 0, 0, 1, 0, 0, 1, 0]
        first = moving_block_bootstrap_ci(new, old, block_size=3, repetitions=500, seed=7)
        second = moving_block_bootstrap_ci(new, old, block_size=3, repetitions=500, seed=7)
        self.assertEqual(first, second)
        self.assertLessEqual(first[0], first[1])

    def test_null_simulation_is_reproducible(self):
        first = simulate_max_stat_null(n=30, baselines=[0.25, 0.5], trials=3, simulations=100, seed=5)
        second = simulate_max_stat_null(n=30, baselines=[0.25, 0.5], trials=3, simulations=100, seed=5)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first["max_z_95_threshold"], 0)

    def test_max_stat_p_value_has_plus_one_correction(self):
        value = max_stat_adjusted_p_value(
            100.0, n=20, baselines=[0.25], trials=2, simulations=100, seed=3
        )
        self.assertAlmostEqual(value, 1 / 101)


if __name__ == "__main__":
    unittest.main()
