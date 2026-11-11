# 담당: 육주영
"""eval/report.py 테스트. 실행: python -m unittest discover -s tests -t . -v"""
import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from eval.report import evaluate, rate_table, report, wilson_ci

CFG = {"match_window_s": 2.0, "min_n_reference": 30, "bootstrap_n": 100}


def make_judged():
    """08–09시 40건(미양보 10, 양보 30), 09–10시 3건(미양보 1, 양보 1, 제외 1)."""
    results = ["no_yield"] * 10 + ["yield"] * 30 + ["no_yield", "yield", "exclude"]
    n = len(results)
    return pd.DataFrame({
        "video": ["A_20261031_0800"] * n,
        "event_id": range(1, n + 1),
        "cls": ["vehicle"] * n,
        "t_enter_s": [i * 60.0 for i in range(40)] + [3700.0, 3800.0, 3900.0],
        "intersection": ["A"] * n,
        "slot": ["08–09시"] * 40 + ["09–10시"] * 3,
        "ped_result": results,
    })


class WilsonTest(unittest.TestCase):
    def test_known_value(self):
        low, high = wilson_ci(5, 10)
        self.assertAlmostEqual(low, 0.2366, places=3)
        self.assertAlmostEqual(high, 0.7634, places=3)

    def test_zero_n(self):
        self.assertTrue(all(math.isnan(v) for v in wilson_ci(0, 0)))

    def test_stays_within_0_and_1(self):
        # 미양보 0건·전부 미양보일 때 계산 오차로 -1.4e-17, 1.0000000000000002가 나오던 건수
        self.assertEqual(wilson_ci(0, 15)[0], 0.0)
        self.assertEqual(wilson_ci(5, 5)[1], 1.0)


class RateTableTest(unittest.TestCase):
    def test_counts_and_reference_note(self):
        table = rate_table(make_judged(), 30).set_index("slot")
        self.assertEqual(table.loc["08–09시", "n_target"], 40)
        self.assertAlmostEqual(table.loc["08–09시", "violation_rate"], 0.25)
        self.assertEqual(table.loc["08–09시", "note"], "")
        # 제외는 분모에서 빠지고, 30건 미만은 참고용
        self.assertEqual(table.loc["09–10시", "n_target"], 2)
        self.assertEqual(table.loc["09–10시", "note"], "참고용")


class ReportFileTest(unittest.TestCase):
    def test_report_and_evaluate(self):
        judged = make_judged()
        # 라벨러 둘이 앞 4건을 라벨링. 첫 건만 서로 다르다.
        base = judged.head(4)[["video", "event_id", "cls", "t_enter_s"]]
        a = base.assign(ped_label=["no_yield", "no_yield", "yield", "yield"], labeler="a")
        b = base.assign(ped_label=["yield", "no_yield", "yield", "yield"], labeler="b")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "judged").mkdir()
            (tmp / "labels").mkdir()
            judged.to_csv(tmp / "judged" / "A_20261031_0800.csv", index=False, encoding="utf-8-sig")
            a.to_csv(tmp / "labels" / "A_20261031_0800_a.csv", index=False, encoding="utf-8-sig")
            b.to_csv(tmp / "labels" / "A_20261031_0800_b.csv", index=False, encoding="utf-8-sig")
            (tmp / "eval.yaml").write_text(yaml.safe_dump(CFG), encoding="utf-8")

            report(str(tmp / "judged"), str(tmp / "labels"), str(tmp / "eval.yaml"), str(tmp / "report"))
            self.assertTrue((tmp / "report" / "yield_rate.csv").exists())
            self.assertGreater((tmp / "report" / "yield_rate.png").stat().st_size, 0)

            metrics, kappas = evaluate(judged, str(tmp / "labels"), CFG)

        m = metrics.set_index("labeler")
        self.assertEqual(m.loc["a", "n_reviewed"], 4)
        self.assertAlmostEqual(m.loc["a", "extraction_recall"], 1.0)
        # 판정은 앞 4건이 모두 no_yield. 라벨러 a 기준: TP 2, FP 2 → F1 = 2/3
        self.assertAlmostEqual(m.loc["a", "no_yield_f1"], 2 / 3)
        # 정답 위반율 0.5, 영상 전체 판정 위반율 11/42
        self.assertAlmostEqual(m.loc["a", "rate_error_pp"], abs(0.5 - 11 / 42) * 100)
        self.assertEqual(kappas.loc[0, "n_both"], 4)
        self.assertAlmostEqual(kappas.loc[0, "kappa"], 0.5)

    def test_labels_without_judged_or_match(self):
        judged = make_judged()
        cols = {"cls": "vehicle", "ped_label": "yield"}
        with tempfile.TemporaryDirectory() as tmp:
            # 아직 판정 결과가 없는 영상의 라벨표 → 지표 표에서 건너뛴다
            pd.DataFrame({"video": ["A_20261031_2000"], "event_id": [1], "t_enter_s": [10.0], "labeler": "a", **cols}).to_csv(
                Path(tmp) / "A_20261031_2000_a.csv", index=False, encoding="utf-8-sig")
            # 판정 결과는 있는데 짝이 하나도 없는 라벨표(시각이 어긋남) → 재현율 0, F1은 nan
            pd.DataFrame({"video": ["A_20261031_0800"], "event_id": [1], "t_enter_s": [30.0], "labeler": "b", **cols}).to_csv(
                Path(tmp) / "A_20261031_0800_b.csv", index=False, encoding="utf-8-sig")
            metrics, _ = evaluate(judged, tmp, CFG)
        self.assertEqual(metrics["labeler"].tolist(), ["b"])
        self.assertEqual(metrics.loc[0, "extraction_recall"], 0.0)
        self.assertTrue(math.isnan(metrics.loc[0, "no_yield_f1"]))
        self.assertTrue(math.isnan(metrics.loc[0, "f1_ci_low"]))



if __name__ == "__main__":
    unittest.main()
