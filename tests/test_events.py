# 담당: 김민영
"""tracking/events.py 테스트 (가짜 트랙, 실제 영상 없이). 실행: python -m unittest discover -s tests -t . -v"""
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from tracking.events import COLUMNS, exit_area, extract_events, find_events

FPS = 10.0
APPROACH = [[100, 0], [200, 0], [200, 100], [100, 100]]  # 오른쪽 위
EXIT = [[0, 200], [100, 200], [100, 300], [0, 300]]      # 왼쪽 아래 (homography 사각형)
AREA = [[0, 150], [100, 150], [100, 300], [0, 300]]      # 사각형 위로 블록이 더 나온 판정 영역


def path(track_id, points, cls="vehicle", coco="car", t0=0.0):
    """points = 하단 중심 픽셀 [(x, y), ...]를 0.1초 간격 프레임으로. 박스 폭 10."""
    f0 = round(t0 * FPS)
    return pd.DataFrame([{
        "frame": f0 + i, "t_s": (f0 + i) / FPS, "track_id": track_id, "cls": cls, "coco_cls": coco, "conf": 0.9,
        "x1": x - 5, "y1": y - 20, "x2": x + 5, "y2": y, "bx_m": 0.0, "by_m": 0.0,
    } for i, (x, y) in enumerate(points)])


RIGHT_TURN = [(150, 50), (150, 150), (50, 250)]  # 진입로 → 중간 → 출구


class FindEventsTest(unittest.TestCase):
    def test_approach_then_exit_is_event(self):
        ev = find_events(path(1, RIGHT_TURN), APPROACH, EXIT)
        self.assertEqual(list(ev["track_id"]), [1])
        self.assertAlmostEqual(ev["t_enter_s"].iloc[0], 0.2)

    def test_exit_without_approach_is_not_event(self):
        self.assertTrue(find_events(path(1, [(150, 150), (50, 250)]), APPROACH, EXIT).empty)

    def test_approach_without_exit_is_not_event(self):
        self.assertTrue(find_events(path(1, [(150, 50), (150, 150), (300, 300)]), APPROACH, EXIT).empty)

    def test_exit_before_approach_is_not_event(self):
        self.assertTrue(find_events(path(1, RIGHT_TURN[::-1]), APPROACH, EXIT).empty)

    def test_reentry_uses_first_entry(self):
        pts = RIGHT_TURN + [(150, 250), (50, 250)]  # 출구 → 밖 → 다시 출구
        ev = find_events(path(1, pts), APPROACH, EXIT)
        self.assertEqual(len(ev), 1)
        self.assertAlmostEqual(ev["t_enter_s"].iloc[0], 0.2)

    def test_pedestrian_ignored_motorcycle_kept(self):
        df = pd.concat([path(1, RIGHT_TURN, cls="pedestrian", coco="person"),
                        path(2, RIGHT_TURN, cls="motorcycle", coco="motorcycle")])
        self.assertEqual(list(find_events(df, APPROACH, EXIT)["track_id"]), [2])

    def test_area_catches_blocks_outside_rectangle(self):
        pts = [(150, 50), (50, 170)]  # 사각형 밖, 판정 영역 안
        self.assertTrue(find_events(path(1, pts), APPROACH, EXIT).empty)
        self.assertEqual(len(find_events(path(1, pts), APPROACH, AREA)), 1)

    def test_coco_cls_is_track_mode(self):
        df = path(1, RIGHT_TURN + [(50, 260)], coco="truck")
        df.loc[0, "coco_cls"] = "car"
        self.assertEqual(find_events(df, APPROACH, EXIT)["coco_cls"].iloc[0], "truck")

    def test_sorted_by_enter_time(self):
        df = pd.concat([path(1, RIGHT_TURN, t0=5.0), path(2, RIGHT_TURN, t0=1.0)])
        self.assertEqual(list(find_events(df, APPROACH, EXIT)["track_id"]), [2, 1])


A_PART = [(150, 50), (150, 90)]       # 진입로 안에서 끝남 (박스 폭 10 → 이어 보기 거리 20 px)
B_PART = [(150, 105), (50, 250)]      # 진입로 밖에서 시작해 출구로


def broken(gap_s=1.0, b_start=(150, 105), cls_b="vehicle"):
    """한 차가 진입로 끝(t 0.1)에서 끊기고 gap_s 뒤 b_start에서 다른 트랙으로 다시 잡힌 것."""
    return pd.concat([path(1, A_PART), path(2, [b_start, (50, 250)], cls=cls_b, t0=0.1 + gap_s)])


def linked(df):
    return find_events(df, APPROACH, EXIT, link_gap_s=3.0, link_dist_box=2.0)


class LinkBrokenTest(unittest.TestCase):
    def test_broken_track_is_linked(self):
        ev = linked(broken())
        self.assertEqual(list(ev["track_id"]), [2])
        self.assertAlmostEqual(ev["t_enter_s"].iloc[0], 1.2)
        self.assertEqual(ev["link_track_id"].iloc[0], 1)                  # 앞 트랙 id를 남김

    def test_normal_event_has_no_link(self):
        df = pd.concat([path(1, RIGHT_TURN), broken().assign(track_id=lambda d: d.track_id + 10)])
        ev = linked(df).set_index("track_id")["link_track_id"]
        self.assertTrue(pd.isna(ev[1]))
        self.assertEqual(ev[12], 11)

    def test_off_by_default(self):
        self.assertTrue(find_events(broken(), APPROACH, EXIT).empty)

    def test_gap_too_long(self):
        self.assertTrue(linked(broken(gap_s=3.5)).empty)

    def test_too_far(self):
        self.assertTrue(linked(broken(b_start=(150, 125))).empty)   # 35 px > 20 px

    def test_other_class_not_linked(self):
        self.assertTrue(linked(broken(cls_b="motorcycle")).empty)

    def test_approach_track_already_event_not_reused(self):
        df = pd.concat([path(1, RIGHT_TURN), path(2, B_PART, t0=0.5)])   # 1은 이미 우회전
        self.assertEqual(list(linked(df)["track_id"]), [1])

    def test_one_to_one_nearest(self):
        df = pd.concat([path(1, A_PART), path(2, B_PART, t0=1.0), path(3, [(150, 101), (50, 250)], t0=1.0)])
        self.assertEqual(list(linked(df)["track_id"]), [3])          # 3이 더 가까움(11 px < 15 px), 2는 짝 없음


class ExitAreaTest(unittest.TestCase):
    def test_empty_area_falls_back_to_corners(self):
        self.assertEqual(exit_area({"crosswalk": {"corners_px": EXIT, "area_px": []}}), EXIT)
        self.assertEqual(exit_area({"crosswalk": {"corners_px": EXIT}}), EXIT)

    def test_area_used_when_given(self):
        self.assertEqual(exit_area({"crosswalk": {"corners_px": EXIT, "area_px": AREA}}), AREA)


class ExtractEventsTest(unittest.TestCase):
    def test_csv_schema_and_ids(self):
        with tempfile.TemporaryDirectory() as d:
            tracks = Path(d) / "A_20261031_0800.csv"
            pd.concat([path(7, RIGHT_TURN, t0=3.0), path(4, RIGHT_TURN)]).to_csv(tracks, index=False, encoding="utf-8-sig")
            site = Path(d) / "site.yaml"
            site.write_text(yaml.safe_dump({"approach_roi_px": APPROACH,
                                            "crosswalk": {"corners_px": EXIT, "area_px": []}}), encoding="utf-8")
            config = Path(d) / "detect.yaml"
            config.write_text(yaml.safe_dump({"link_gap_s": 3.0, "link_dist_box": 2.0}), encoding="utf-8")
            out = pd.read_csv(extract_events(str(tracks), str(site), str(config), out_dir=d), encoding="utf-8-sig")
        self.assertEqual(list(out.columns), COLUMNS)
        self.assertEqual(list(out["event_id"]), [1, 2])
        self.assertEqual(list(out["track_id"]), [4, 7])
        self.assertEqual(set(out["video"]), {"A_20261031_0800"})
        self.assertTrue(out["t_stopline_s"].isna().all())


if __name__ == "__main__":
    unittest.main()
