# 담당: 육주영
"""eval/report.py 테스트. 실행: python -m unittest discover -s tests -t . -v"""
import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from eval.report import select_clip_events, apply_red_mode, evaluate, rate_table, report, wilson_ci

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
        self.assertEqual(wilson_ci(0, 44)[0], 0.0)  # 6.9e-18이 나와 위반율 그래프(yerr 음수)가 멈추던 건수
        self.assertEqual(wilson_ci(44, 44)[1], 1.0)


class RateTableTest(unittest.TestCase):
    def test_counts_and_reference_note(self):
        table = rate_table(make_judged(), 30).set_index("slot")
        self.assertEqual(table.loc["08–09시", "n_target"], 40)
        self.assertAlmostEqual(table.loc["08–09시", "violation_rate"], 0.25)
        self.assertEqual(table.loc["08–09시", "note"], "")
        # 제외는 분모에서 빠지고, 30건 미만은 참고용
        self.assertEqual(table.loc["09–10시", "n_target"], 2)
        self.assertEqual(table.loc["09–10시", "note"], "참고용")

    def test_red_rate_counts_stopped_and_not_stopped_only(self):
        judged = make_judged().assign(
            red_result=["not_stopped"] * 10 + ["stopped"] * 20 + ["na"] * 10 + ["not_stopped", "exclude", "na"]
        )
        table = rate_table(judged, 30, "red_result", "stopped", "not_stopped").set_index("slot")
        # na·제외는 분모에서 빠진다
        self.assertEqual(table.loc["08–09시", "n_target"], 30)
        self.assertEqual(table.loc["08–09시", "n_not_stopped"], 10)
        self.assertAlmostEqual(table.loc["08–09시", "violation_rate"], 1 / 3)
        self.assertEqual(table.loc["09–10시", "n_target"], 1)

    def test_red_mode_moves_minor_stop_only(self):
        judged = make_judged().assign(
            red_result=["not_stopped"] * 10 + ["stopped"] * 20 + ["stopped_minor"] * 10 + ["na"] * 3
        )
        # 경미 정지 10건: 판정 대상은 두 모드가 같고(40), 엄격 모드에서만 위반 의심으로 센다 (기술서 3.1)
        for mode, bad in [("보통", 10), ("엄격", 20)]:
            row = rate_table(apply_red_mode(judged, mode), 30, "red_result", "stopped", "not_stopped").set_index("slot").loc["08–09시"]
            self.assertEqual((row["n_target"], row["n_not_stopped"]), (40, bad))
        self.assertNotIn("red_result", apply_red_mode(make_judged()).columns)  # 2단계 판정 전 표는 그대로


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

    def test_red_rate_file_and_not_stopped_f1(self):
        judged = make_judged().assign(red_result=["not_stopped", "stopped", "na", "not_stopped"] + ["na"] * 39)
        gt = judged.head(4)[["video", "event_id", "cls", "t_enter_s"]].assign(
            ped_label="yield", red_label=["not_stopped", "not_stopped", "na", "stopped"], labeler="a"
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "judged").mkdir()
            (tmp / "labels").mkdir()
            judged.to_csv(tmp / "judged" / "A_20261031_0800.csv", index=False, encoding="utf-8-sig")
            gt.to_csv(tmp / "labels" / "A_20261031_0800_a.csv", index=False, encoding="utf-8-sig")
            (tmp / "eval.yaml").write_text(yaml.safe_dump(CFG), encoding="utf-8")

            report(str(tmp / "judged"), str(tmp / "labels"), str(tmp / "eval.yaml"), str(tmp / "report"))
            red = pd.read_csv(tmp / "report" / "red_rate.csv", encoding="utf-8-sig").set_index("slot")
            metrics, _ = evaluate(judged, str(tmp / "labels"), CFG)

        self.assertEqual(red.loc["08–09시", "n_target"], 3)
        self.assertEqual(red.loc["08–09시", "n_not_stopped"], 2)
        # 정답 판정 대상 3건(na 1건 제외): TP 1, FN 1, FP 1 → F1 0.5
        self.assertAlmostEqual(metrics.loc[0, "not_stopped_f1"], 0.5)
        self.assertTrue(0.0 <= metrics.loc[0, "red_f1_ci_low"] <= 0.5 <= metrics.loc[0, "red_f1_ci_high"] <= 1.0)

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



class ClipSelectTest(unittest.TestCase):
    """클립을 만들 이벤트 고르기(기술서 3.4)."""
    CFG = {"min_n_reference": 30, "clip_red_slots": 2, "clip_red_per_slot": 3}

    def slot(self, slot, n, red, ped=0):
        return pd.DataFrame({
            "intersection": "A", "slot": slot, "wall_time": [f"2026-11-22 {slot[:2]}:{i:02d}:00" for i in range(n)],
            "ped_result": ["no_yield"] * ped + ["yield"] * (n - ped),
            "red_result": ["not_stopped"] * red + ["stopped"] * (n - red)})

    def test_all_no_yield_and_red_sampled_from_top_slots(self):
        df = pd.concat([self.slot("08–09시", 40, 20), self.slot("12–13시", 40, 5, ped=2), self.slot("18–19시", 40, 30), self.slot("22–23시", 10, 10)],
                       ignore_index=True)
        got = select_clip_events(df, self.CFG)
        # 미양보 2건은 모두(12–13시). 미정지는 위반율 높은 18–19시·08–09시에서 3개씩. 22–23시는 100%지만 참고용
        self.assertEqual(got.groupby("slot").size().to_dict(), {"08–09시": 3, "12–13시": 2, "18–19시": 3})
        self.assertEqual([t[-5:] for t in got[got["slot"] == "18–19시"]["wall_time"]], ["00:00", "14:00", "29:00"])   # 시각 순으로 고르게

    def test_small_video_falls_back_to_most_suspects(self):
        df = pd.concat([self.slot("08–09시", 5, 1), self.slot("12–13시", 5, 4), self.slot("18–19시", 5, 2)], ignore_index=True)
        self.assertEqual(select_clip_events(df, self.CFG).groupby("slot").size().to_dict(), {"12–13시": 3, "18–19시": 2})

    def test_event_with_both_kinds_counted_once(self):
        df = self.slot("08–09시", 5, 3, ped=1)
        self.assertEqual(list(select_clip_events(df, self.CFG).index), [0, 1, 2])
        self.assertEqual(list(select_clip_events(df.drop(columns="red_result"), self.CFG).index), [0])   # 2단계 판정 전


if __name__ == "__main__":
    unittest.main()
