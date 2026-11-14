# 담당: 육주영
"""rules/rule.py 테스트. 실행: python -m unittest discover -s tests -t . -v"""
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from rules.rule import in_camera_moved, judge, judge_ped, judge_red

NAN = float("nan")


class JudgePedTest(unittest.TestCase):
    def test_no_pedestrian_at_entry_is_yield(self):
        self.assertEqual(judge_ped(0, 0, False), ("yield", ""))

    def test_pedestrian_at_entry_is_no_yield(self):
        self.assertEqual(judge_ped(0, 1, False), ("no_yield", ""))

    def test_visible_pedestrian_wins_over_lost_track(self):
        self.assertEqual(judge_ped(1, 1, False), ("no_yield", ""))

    def test_lost_pedestrian_track_is_excluded(self):
        self.assertEqual(judge_ped(1, 0, False), ("exclude", "occluded"))

    def test_camera_moved_is_excluded_first(self):
        self.assertEqual(judge_ped(0, 1, True), ("exclude", "camera_moved"))


class JudgeRedTest(unittest.TestCase):
    """인자: 신호, 정지선 앞 정지 시간(초), 정지 기준(초), 우회전 전용 신호, 카메라 움직임."""

    def test_red_and_stopped_long_enough(self):
        self.assertEqual(judge_red("red", 1.0, 1.0, False, False), ("stopped", ""))

    def test_red_and_stop_too_short(self):
        self.assertEqual(judge_red("red", 0.9, 1.0, False, False), ("not_stopped", ""))
        self.assertEqual(judge_red("red", 0.0, 1.0, False, False), ("not_stopped", ""))

    def test_not_red_is_na_even_without_stop(self):
        self.assertEqual(judge_red("not_red", 0.0, 1.0, False, False), ("na", ""))

    def test_unknown_or_empty_signal_is_excluded(self):
        self.assertEqual(judge_red("unknown", 2.0, 1.0, False, False), ("exclude", "signal_unknown"))
        self.assertEqual(judge_red(NAN, NAN, 1.0, False, False), ("exclude", "signal_unknown"))

    def test_red_without_stop_time_is_excluded_not_violation(self):
        self.assertEqual(judge_red("red", NAN, 1.0, False, False), ("exclude", "stopline_hidden"))

    def test_right_turn_signal_site_is_excluded(self):
        self.assertEqual(judge_red("red", 0.0, 1.0, True, False), ("exclude", "rturn_signal"))

    def test_camera_moved_is_excluded_first(self):
        self.assertEqual(judge_red("red", 0.0, 1.0, True, True), ("exclude", "camera_moved"))


class CameraMovedTest(unittest.TestCase):
    def test_inside(self):
        self.assertTrue(in_camera_moved(101.0, [[100.0, 120.0]]))

    def test_outside(self):
        self.assertFalse(in_camera_moved(90.0, [[100.0, 120.0]]))
        self.assertFalse(in_camera_moved(90.0, []))


class JudgeFileTest(unittest.TestCase):
    def test_writes_judged_csv(self):
        features = pd.DataFrame({
            "video": ["A_20261031_0800"] * 3,
            "event_id": [1, 2, 3],
            "track_id": [11, 12, 13],
            "cls": ["vehicle", "motorcycle", "vehicle"],
            "coco_cls": ["car", "motorcycle", "bus"],
            "t_enter_s": [15.0, 101.0, 3700.0],
            "peds_lost": [0, 0, 0],
            "peds_at_entry": [1, 1, 0],
            "no_yield_at": ["near", "exit", ""],
            "signal_at_stopline": ["red", "red", "not_red"],
            "stop_sec_before_line": [0.2, 3.0, 0.0],
        })
        site = {"site": "A", "camera_moved_s": [[100.0, 120.0]], "right_turn_signal": False, "stop_min_duration_s": 1.0}
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            features.to_csv(tmp / "A_20261031_0800.csv", index=False, encoding="utf-8-sig")
            (tmp / "site.yaml").write_text(yaml.safe_dump(site), encoding="utf-8")

            judge(str(tmp / "A_20261031_0800.csv"), str(tmp / "site.yaml"), str(tmp / "judged"))
            out = pd.read_csv(tmp / "judged" / "A_20261031_0800.csv", encoding="utf-8-sig", keep_default_na=False)

        self.assertEqual(list(out["ped_result"]), ["no_yield", "exclude", "yield"])
        self.assertEqual(list(out["ped_exclude_reason"]), ["", "camera_moved", ""])
        # 두 판정은 독립: 1번은 미양보이면서 미정지, 2번은 카메라가 움직여 둘 다 제외
        self.assertEqual(list(out["red_result"]), ["not_stopped", "exclude", "na"])
        self.assertEqual(list(out["red_exclude_reason"]), ["", "camera_moved", ""])
        self.assertEqual(list(out["slot"]), ["08–09시", "08–09시", "09–10시"])
        self.assertEqual(out["wall_time"][2], "2026-10-31 09:01:40")
        self.assertEqual(list(out["vehicle_type"]), ["승용", "오토바이", "버스"])
        self.assertEqual(list(out["intersection"]), ["A"] * 3)
        self.assertEqual(list(out["status"]), ["auto"] * 3)
        self.assertEqual(list(out["clip"]), [""] * 3)
        self.assertEqual(list(out["no_yield_at"]), ["near", "exit", ""])  # 어느 횡단보도에서 위반했는지 남는다


if __name__ == "__main__":
    unittest.main()
