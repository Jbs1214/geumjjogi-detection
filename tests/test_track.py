# 담당: 김민영
"""tracking/track.py 후처리 테스트 (가짜 트랙, 실제 영상·YOLO 없이). 실행: python -m unittest discover -s tests -t . -v"""
import unittest

import numpy as np
import pandas as pd

from tracking.homography import compute_homography
from tracking.track import COLUMNS, add_meters, drop_short, merge_tracks, remove_drivers, unify_class

FPS = 10.0


def rows(track_id, cls, t0, t1, x0=0.0, x1=None, box=(0, 0, 10, 10)):
    """t0~t1초 동안 매 프레임 한 줄. 미터 좌표는 x0 → x1로 고르게 움직인다."""
    frames = np.arange(round(t0 * FPS), round(t1 * FPS) + 1)
    xs = np.linspace(x0, x0 if x1 is None else x1, len(frames))
    return pd.DataFrame({
        "frame": frames, "t_s": frames / FPS, "track_id": track_id, "cls": cls, "coco_cls": "car", "conf": 0.9,
        "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3], "bx_m": xs, "by_m": 0.0,
    })


def ids(df):
    return sorted(df["track_id"].unique())


class RemoveDriversTest(unittest.TestCase):
    def setUp(self):
        self.moto = rows(1, "motorcycle", 0, 0, box=(0, 0, 100, 100))

    def check(self, person_box):
        df = pd.concat([self.moto, rows(2, "pedestrian", 0, 0, box=person_box)], ignore_index=True)
        return 2 in remove_drivers(df, 0.5)["track_id"].values

    def test_inside_motorcycle_is_driver(self):
        self.assertFalse(self.check((20, 10, 60, 90)))

    def test_outside_is_pedestrian(self):
        self.assertTrue(self.check((200, 0, 240, 80)))

    def test_boundary_half(self):
        self.assertFalse(self.check((80, 0, 120, 100)))   # 정확히 50% → 운전자
        self.assertTrue(self.check((81, 0, 121, 100)))    # 47.5% → 보행자

    def test_other_frame_motorcycle_ignored(self):
        df = pd.concat([self.moto, rows(2, "pedestrian", 1, 1, box=(20, 10, 60, 90))], ignore_index=True)
        self.assertIn(2, remove_drivers(df, 0.5)["track_id"].values)


class MergeTracksTest(unittest.TestCase):
    def merge(self, *parts):
        return merge_tracks(pd.concat(parts, ignore_index=True), gap_s=2.0, dist_m=1.5)

    def test_short_gap_close_merged(self):
        df = self.merge(rows(1, "vehicle", 0, 3, 0, 3), rows(2, "vehicle", 4.5, 6, 4, 6))  # 1.5초, 1 m
        self.assertEqual(ids(df), [1])

    def test_long_gap_not_merged(self):
        df = self.merge(rows(1, "vehicle", 0, 3, 0, 3), rows(2, "vehicle", 6, 7, 3, 4))   # 3초
        self.assertEqual(ids(df), [1, 2])

    def test_far_not_merged(self):
        df = self.merge(rows(1, "vehicle", 0, 3, 0, 3), rows(2, "vehicle", 4, 5, 5, 6))   # 2 m
        self.assertEqual(ids(df), [1, 2])

    def test_other_class_not_merged(self):
        df = self.merge(rows(1, "vehicle", 0, 3, 0, 3), rows(2, "pedestrian", 4, 5, 3, 3))
        self.assertEqual(ids(df), [1, 2])

    def test_overlapping_not_merged(self):
        df = self.merge(rows(1, "vehicle", 0, 3, 0, 3), rows(2, "vehicle", 2, 5, 3, 4))   # 동시에 있던 두 차
        self.assertEqual(ids(df), [1, 2])

    def test_chain_three_parts(self):
        df = self.merge(rows(5, "vehicle", 0, 2, 0, 2), rows(9, "vehicle", 3, 5, 2.5, 4), rows(7, "vehicle", 6, 8, 4.5, 6))
        self.assertEqual(ids(df), [5])

    def test_nearest_candidate(self):
        a, b = rows(1, "vehicle", 0, 2, 0, 0), rows(2, "vehicle", 0, 2, 2.5, 2.5)
        c = rows(3, "vehicle", 3, 4, 1.4, 1.4)             # 1에서 1.4 m, 2에서 1.1 m → 둘 다 후보, 2가 더 가깝다
        df = self.merge(a, b, c)
        self.assertEqual(df.loc[df["frame"] == 35, "track_id"].tolist(), [2])


class OverlapMergeTest(unittest.TestCase):
    """B가 A 끝나기 직전에 시작한 경우 (한 물체에 ID가 둘 붙었다가 A가 끝남)."""
    A_BOX, SAME, OTHER = (0, 0, 10, 10), (1, 0, 11, 10), (8, 0, 18, 10)   # IoU 0.82 / 0.11

    def merge(self, b_start, b_box, classes=("vehicle",), cls="vehicle"):
        a = rows(1, cls, 0, 3, 0, 3, box=self.A_BOX)
        b = rows(2, cls, b_start, 6, 3, 6, box=b_box)
        return merge_tracks(pd.concat([a, b], ignore_index=True), 2.0, 1.5,
                            overlap_s=0.5, overlap_iou=0.5, overlap_classes=classes)

    def test_short_overlap_high_iou_merged(self):
        self.assertEqual(ids(self.merge(2.9, self.SAME)), [1])   # 0.1초 겹침

    def test_low_iou_not_merged(self):
        self.assertEqual(ids(self.merge(2.9, self.OTHER)), [1, 2])   # 줄 선 옆 차

    def test_long_overlap_not_merged(self):
        self.assertEqual(ids(self.merge(2.3, self.SAME)), [1, 2])   # 0.7초 겹침

    def test_class_not_listed_not_merged(self):
        self.assertEqual(ids(self.merge(2.9, self.SAME, cls="pedestrian")), [1, 2])

    def test_listed_pedestrian_merged(self):
        self.assertEqual(ids(self.merge(2.9, self.SAME, classes=("pedestrian",), cls="pedestrian")), [1])

    def test_overlap_frames_kept_once(self):
        df = self.merge(2.9, self.SAME)
        self.assertFalse(df.duplicated(["frame", "track_id"]).any())
        self.assertEqual(len(df), 61)   # 0~6초 = 61프레임


class DropShortTest(unittest.TestCase):
    def test_short_dropped(self):
        df = pd.concat([rows(1, "vehicle", 0, 0.8), rows(2, "vehicle", 0, 1.0)], ignore_index=True)
        self.assertEqual(ids(drop_short(df, 1.0)), [2])

    def test_merged_fragments_kept(self):
        df = pd.concat([rows(1, "vehicle", 0, 0.6, 0, 0.6), rows(2, "vehicle", 1.5, 2.0, 1.0, 1.5)], ignore_index=True)
        merged = merge_tracks(df, 2.0, 1.5)
        self.assertEqual(ids(drop_short(merged, 1.0)), [1])   # 조각은 각각 1초 미만이지만 합치면 2초
        self.assertEqual(ids(drop_short(df, 1.0)), [])        # 먼저 지우면 둘 다 사라진다


class UnifyClassTest(unittest.TestCase):
    def test_flicker_row_takes_track_mode(self):
        df = rows(1, "pedestrian", 0, 1)
        df.loc[3, "cls"] = "vehicle"                        # 한 프레임만 vehicle로 잘못 검출
        out = unify_class(df)
        self.assertEqual(set(out["cls"]), {"pedestrian"})
        self.assertEqual(out["coco_cls"].tolist(), df["coco_cls"].tolist())   # 원래 클래스는 그대로


class MetersTest(unittest.TestCase):
    def test_bottom_center(self):
        H = compute_homography([[0, 0], [100, 0], [100, 100], [0, 100]], 4.0, 10.0)
        df = add_meters(rows(1, "vehicle", 0, 0, box=(40, 0, 60, 50)), H)
        self.assertAlmostEqual(df["bx_m"].iloc[0], 2.0)    # 가운데 x=50 → 4 m의 절반
        self.assertAlmostEqual(df["by_m"].iloc[0], 5.0)    # 아래 y=50 → 10 m의 절반

    def test_schema_columns(self):
        self.assertEqual(COLUMNS, ["frame", "t_s", "track_id", "cls", "coco_cls", "conf",
                                   "x1", "y1", "x2", "y2", "bx_m", "by_m"])


if __name__ == "__main__":
    unittest.main()
