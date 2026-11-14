# 담당: 육주영
"""dashboard/app.py 테스트. 실행: python -m unittest discover -s tests -t . -v"""
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd
import yaml
from streamlit.testing.v1 import AppTest

from dashboard.app import NOTICE, apply_filters, load_judged, top_slots


def make_judged() -> pd.DataFrame:
    """A 08–09시 40건(미양보 10), A 09–10시 40건(미양보 20), B 08–09시 3건(미양보 3, 참고용)."""
    rows = []
    for site, day, slot, hour, n, k in [
        ("A", "2026-10-31", "08–09시", 8, 40, 10),
        ("A", "2026-10-31", "09–10시", 9, 40, 20),
        ("B", "2026-11-07", "08–09시", 8, 3, 3),
    ]:
        for i in range(n):
            rows.append({
                "video": f"{site}_{day.replace('-', '')}_0800",
                "event_id": len(rows) + 1,
                "intersection": site,
                "wall_time": f"{day} {hour:02d}:{i:02d}:00",
                "slot": slot,
                "ped_result": "no_yield" if i < k else "yield",
                "clip": "",
            })
    return pd.DataFrame(rows)


def run_app(judged_dir, config):
    from dashboard.app import main

    main(judged_dir, config)


class DataTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        make_judged().to_csv(Path(self.tmp.name) / "all.csv", index=False, encoding="utf-8-sig")
        self.df = load_judged(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_filters(self):
        self.assertEqual(len(apply_filters(self.df, ["A"], date(2026, 10, 1), date(2026, 11, 30))), 80)
        self.assertEqual(len(apply_filters(self.df, ["A", "B"], date(2026, 11, 1), date(2026, 11, 30))), 3)

    def test_top_slots_skip_reference_only(self):
        from eval.report import rate_table

        top = top_slots(rate_table(self.df, 30))
        # B는 위반율 100%지만 3건이라 참고용 → 빠진다
        self.assertEqual(list(zip(top["intersection"], top["slot"])), [("A", "09–10시"), ("A", "08–09시")])

    def test_top_slots_skip_zero_violation(self):
        # 위반 의심이 0건인 칸은 단속 우선 시간대가 아니다
        table = pd.DataFrame({"note": ["", ""], "violation_rate": [0.0, 0.2]})
        self.assertEqual(list(top_slots(table)["violation_rate"]), [0.2])

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as empty:
            self.assertTrue(load_judged(empty).empty)


class AppSmokeTest(unittest.TestCase):
    def test_renders_with_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_judged().to_csv(Path(tmp) / "all.csv", index=False, encoding="utf-8-sig")
            config = Path(tmp) / "eval.yaml"
            config.write_text(yaml.safe_dump({"min_n_reference": 30}), encoding="utf-8")
            at = AppTest.from_function(run_app, args=(tmp, str(config)), default_timeout=30).run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.warning[0].value, NOTICE)
        self.assertEqual([m.value for m in at.metric], ["83건", "33건", "39.8%"])
        self.assertEqual(len(at.dataframe), 1)

    def run_with_red(self, kind):
        """make_judged에 적색 판정을 붙인다: 83건 중 미정지 20, 일시정지 30, 나머지 na."""
        judged = make_judged().assign(red_result=["not_stopped"] * 20 + ["stopped"] * 30 + ["na"] * 33)
        with tempfile.TemporaryDirectory() as tmp:
            judged.to_csv(Path(tmp) / "all.csv", index=False, encoding="utf-8-sig")
            config = Path(tmp) / "eval.yaml"
            config.write_text(yaml.safe_dump({"min_n_reference": 30}), encoding="utf-8")
            at = AppTest.from_function(run_app, args=(tmp, str(config)), default_timeout=30).run()
            at.sidebar.selectbox[0].select(kind).run()
        self.assertEqual(len(at.exception), 0)
        return at

    def test_red_kind_shows_red_rate(self):
        at = self.run_with_red("적색 미정지")
        # na는 분모에서 빠진다: 판정 대상 50, 미정지 20
        self.assertEqual([m.value for m in at.metric], ["50건", "20건", "40.0%"])
        self.assertEqual(at.metric[1].label, "위반 의심 (미정지)")
        self.assertEqual(len(at.dataframe), 1)

    def test_all_kinds_side_by_side_not_merged(self):
        at = self.run_with_red("전체")
        self.assertEqual([m.value for m in at.metric], ["83건", "33건", "39.8%", "50건", "20건", "40.0%"])
        self.assertEqual(len(at.dataframe), 2)

    def test_red_kind_without_red_judgment(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_judged().to_csv(Path(tmp) / "all.csv", index=False, encoding="utf-8-sig")
            config = Path(tmp) / "eval.yaml"
            config.write_text(yaml.safe_dump({"min_n_reference": 30}), encoding="utf-8")
            at = AppTest.from_function(run_app, args=(tmp, str(config)), default_timeout=30).run()
            at.sidebar.selectbox[0].select("적색 미정지").run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.metric), 0)
        self.assertEqual(at.info[0].value, "적색 미정지 판정 결과가 아직 없다.")

    def test_renders_without_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            at = AppTest.from_function(run_app, args=(tmp, "unused.yaml"), default_timeout=30).run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.warning[0].value, NOTICE)
        self.assertEqual(len(at.info), 1)


if __name__ == "__main__":
    unittest.main()
