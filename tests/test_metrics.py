# 담당: 육주영
"""eval/metrics.py 테스트. 실행: python -m unittest discover -s tests -t . -v"""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from eval.metrics import bootstrap_ci, extraction_recall, kappa, match_events, no_yield_f1, yield_rate_error


def events(rows):
    return pd.DataFrame(rows, columns=["video", "cls", "t_enter_s"])


class F1Test(unittest.TestCase):
    def test_hand_computed(self):
        # 미양보 기준: TP 1, FN 1, FP 1 → 정밀도 0.5, 재현율 0.5, F1 0.5
        y_true = ["no_yield", "no_yield", "yield", "yield"]
        y_pred = ["no_yield", "yield", "no_yield", "yield"]
        self.assertAlmostEqual(no_yield_f1(y_true, y_pred), 0.5)

    def test_true_exclude_is_dropped(self):
        self.assertAlmostEqual(no_yield_f1(["no_yield", "exclude"], ["no_yield", "no_yield"]), 1.0)

    def test_predicted_exclude_counts_as_miss(self):
        # TP 1, FN 1 → 정밀도 1, 재현율 0.5, F1 2/3
        self.assertAlmostEqual(no_yield_f1(["no_yield", "no_yield"], ["no_yield", "exclude"]), 2 / 3)


class MatchTest(unittest.TestCase):
    def test_window_and_cls(self):
        gt = events([("A", "vehicle", 10.0), ("A", "vehicle", 50.0), ("A", "motorcycle", 80.0)])
        pred = events([("A", "vehicle", 12.0), ("A", "vehicle", 53.0), ("A", "vehicle", 80.0)])
        # 2초 차이는 일치, 3초 차이는 불일치, 차종이 다르면 불일치
        self.assertEqual(match_events(gt, pred, 2.0), [(0, 0)])

    def test_one_to_one(self):
        gt = events([("A", "vehicle", 10.0), ("A", "vehicle", 11.0)])
        pred = events([("A", "vehicle", 10.5)])
        self.assertEqual(len(match_events(gt, pred, 2.0)), 1)

    def test_extraction_recall(self):
        gt = events([("A", "vehicle", 10.0), ("A", "vehicle", 50.0)])
        pred = events([("A", "vehicle", 11.0)])
        with tempfile.TemporaryDirectory() as tmp:
            gt_path, pred_path = Path(tmp) / "gt.csv", Path(tmp) / "pred.csv"
            gt.to_csv(gt_path, index=False, encoding="utf-8-sig")
            pred.to_csv(pred_path, index=False, encoding="utf-8-sig")
            self.assertAlmostEqual(extraction_recall(str(gt_path), str(pred_path), 2.0), 0.5)


class OtherMetricsTest(unittest.TestCase):
    def test_kappa(self):
        same = ["yield", "no_yield", "exclude", "yield"]
        self.assertAlmostEqual(kappa(same, same), 1.0)
        # 관측 일치 0.5, 우연 일치 0.5 → kappa 0
        self.assertAlmostEqual(kappa(["yield", "yield", "no_yield", "no_yield"],
                                     ["yield", "no_yield", "yield", "no_yield"]), 0.0)

    def test_yield_rate_error(self):
        self.assertAlmostEqual(yield_rate_error(0.30, 0.22), 8.0)

    def test_bootstrap_ci(self):
        y_true = ["no_yield", "yield"] * 10
        self.assertEqual(bootstrap_ci(y_true, y_true, 200), (1.0, 1.0))
        y_pred = ["no_yield", "yield", "yield", "yield"] * 5
        low, high = bootstrap_ci(y_true, y_pred, 200)
        self.assertTrue(0.0 <= low <= no_yield_f1(y_true, y_pred) <= high <= 1.0)


if __name__ == "__main__":
    unittest.main()
