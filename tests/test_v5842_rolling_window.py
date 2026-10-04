import unittest

import zodiac_ensemble as z
import wide_ensemble as w


class V5842RollingWindowTest(unittest.TestCase):
    def _zfolds(self, n=8):
        folds = []
        for i in range(n):
            probs = {}
            for name in z.SIGNAL_NAMES:
                probs[name] = {f"Z{j}": (1.0 if j == (i % 3) else 0.0) for j in range(3)}
            folds.append({"prob": probs, "all_z": ["Z0", "Z1", "Z2"], "actual": f"Z{i % 3}"})
        return folds

    def test_bma_window_excludes_old_folds(self):
        folds = self._zfolds(8)
        # Make the oldest fold dominate one signal; a one-fold window must ignore it.
        for name in z.SIGNAL_NAMES:
            folds[0]["prob"][name] = {"Z0": 1.0, "Z1": 0.0, "Z2": 0.0}
            folds[6]["prob"][name] = {"Z0": 0.0, "Z1": 1.0, "Z2": 0.0}
            folds[7]["prob"][name] = {"Z0": 0.0, "Z1": 1.0, "Z2": 0.0}
        w_old = z._bma_weights(folds, 8, z.SIGNAL_NAMES, window=8)
        w_recent = z._bma_weights(folds, 8, z.SIGNAL_NAMES, window=1)
        self.assertEqual(len(w_old), len(w_recent))
        self.assertAlmostEqual(float(w_recent.sum()), 1.0, places=9)

    def test_rank_stacking_accepts_window(self):
        folds = self._zfolds(8)
        rank_short = z._rank_stacking(folds, 7, window=2)
        rank_long = z._rank_stacking(folds, 7, window=7)
        self.assertEqual(set(rank_short), set(rank_long))
        self.assertEqual(len(rank_short), 3)

    def test_wide_rank_stacking_accepts_window(self):
        folds = []
        for i in range(8):
            prob = {name: {n: 1.0 / 49.0 for n in w.ALL_NUMS} for name in w.SIGNAL_NAMES}
            folds.append({"prob": prob, "actual": (i % 49) + 1})
        rank = w._rank_stacking(folds, 7, window=2)
        self.assertEqual(len(rank), 49)
        self.assertEqual(set(rank), set(w.ALL_NUMS))


if __name__ == "__main__":
    unittest.main()
