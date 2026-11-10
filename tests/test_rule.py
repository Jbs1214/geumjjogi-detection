# 담당: 육주영
"""rules/rule.py 테스트. 실행: python -m unittest discover -s tests -t . -v"""
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from rules.rule import in_camera_moved, judge, judge_ped


class JudgePedTest(unittest.TestCase):
    def test_no_pedestrian_is_excluded(self):
        self.assertEqual(judge_ped(0, 0, 0, False), ("exclude", "no_ped"))

    def test_pedestrian_at_entry_is_no_yield(self):
        self.assertEqual(judge_ped(2, 0, 1, False), ("no_yield", ""))

    def test_all_pedestrians_left_is_yield(self):
        self.assertEqual(judge_ped(2, 0, 0, False), ("yield", ""))

    def test_lost_pedestrian_track_is_excluded(self):
        self.assertEqual(judge_ped(2, 1, 0, False), ("exclude", "occluded"))

    def test_camera_moved_is_excluded_first(self):
        self.assertEqual(judge_ped(2, 0, 1, True), ("exclude", "camera_moved"))


class CameraMovedTest(unittest.TestCase):
    def test_overlap(self):
        self.assertTrue(in_camera_moved(95.0, 101.0, [[100.0, 120.0]]))

    def test_no_overlap(self):
        self.assertFalse(in_camera_moved(80.0, 90.0, [[100.0, 120.0]]))
        self.assertFalse(in_camera_moved(80.0, 90.0, []))


class JudgeFileTest(unittest.TestCase):
    def test_writes_judged_csv(self):
        features = pd.DataFrame({
            "video": ["A_20261031_0800"] * 3,
            "event_id": [1, 2, 3],
            "track_id": [11, 12, 13],
            "cls": ["vehicle", "motorcycle", "vehicle"],
            "coco_cls": ["car", "motorcycle", "bus"],
            "t_approach_s": [10.0, 95.0, 3695.0],
            "t_enter_s": [15.0, 101.0, 3700.0],
            "n_ped_max": [2, 1, 1],
            "peds_lost": [0, 0, 0],
            "peds_at_entry": [1, 1, 0],
            "stop_sec": [5.0, 0.0, 0.0],  # 정지했어도 진입 때 보행자가 있으면 미양보
        })
        site = {"site": "A", "camera_moved_s": [[100.0, 120.0]]}
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            features.to_csv(tmp / "A_20261031_0800.csv", index=False, encoding="utf-8-sig")
            (tmp / "site.yaml").write_text(yaml.safe_dump(site), encoding="utf-8")

            judge(str(tmp / "A_20261031_0800.csv"), str(tmp / "site.yaml"), str(tmp / "judged"))
            out = pd.read_csv(tmp / "judged" / "A_20261031_0800.csv", encoding="utf-8-sig", keep_default_na=False)

        self.assertEqual(list(out["ped_result"]), ["no_yield", "exclude", "yield"])
        self.assertEqual(list(out["ped_exclude_reason"]), ["", "camera_moved", ""])
        self.assertEqual(list(out["slot"]), ["08–09시", "08–09시", "09–10시"])
        self.assertEqual(out["wall_time"][2], "2026-10-31 09:01:40")
        self.assertEqual(list(out["vehicle_type"]), ["승용", "오토바이", "버스"])
        self.assertEqual(list(out["intersection"]), ["A"] * 3)
        self.assertEqual(list(out["status"]), ["auto"] * 3)
        self.assertEqual(list(out["clip"]), [""] * 3)


if __name__ == "__main__":
    unittest.main()
