# 담당: 김민영
"""tracking/track.py 후처리 테스트 (가짜 트랙, 실제 영상·YOLO 없이). 실행: python -m unittest discover -s tests -t . -v"""
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from tracking.homography import compute_homography
from tracking.track import (COLUMNS, add_meters, detect_and_track, drop_short, load_homography, merge_tracks,
                            remove_drivers, unify_class)

FPS = 10.0


def rows(track_id, cls, t0, t1, x0=0.0, x1=None, box=None, w=10.0):
    """t0~t1초 동안 매 프레임 한 줄. box를 주면 그 박스에 고정, 아니면 폭 w·높이 10인 박스의
    하단 중심점(픽셀)이 x0 → x1로 고르게 움직인다."""
    frames = np.arange(round(t0 * FPS), round(t1 * FPS) + 1)
    xs = np.linspace(x0, x0 if x1 is None else x1, len(frames))
    x1s, x2s = (box[0], box[2]) if box else (xs - w / 2, xs + w / 2)
    y1, y2 = (box[1], box[3]) if box else (0.0, 10.0)
    return pd.DataFrame({
        "frame": frames, "t_s": frames / FPS, "track_id": track_id, "cls": cls, "coco_cls": "car", "conf": 0.9,
        "x1": x1s, "y1": y1, "x2": x2s, "y2": y2, "bx_m": np.nan, "by_m": np.nan,
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
    """이어 붙이는 거리 = 앞 트랙 마지막 박스 폭(기본 10 px) × dist_box(1.0). 하단 중심점 픽셀 거리로 잰다."""
    def merge(self, *parts):
        return merge_tracks(pd.concat(parts, ignore_index=True), gap_s=2.0, dist_box=1.0)

    def test_short_gap_close_merged(self):
        df = self.merge(rows(1, "vehicle", 0, 3, 0, 30), rows(2, "vehicle", 4.5, 6, 35, 50))  # 1.5초, 5 px(박스 폭의 0.5배)
        self.assertEqual(ids(df), [1])

    def test_long_gap_not_merged(self):
        df = self.merge(rows(1, "vehicle", 0, 3, 0, 30), rows(2, "vehicle", 6, 7, 30, 40))   # 3초
        self.assertEqual(ids(df), [1, 2])

    def test_far_not_merged(self):
        df = self.merge(rows(1, "vehicle", 0, 3, 0, 30), rows(2, "vehicle", 4, 5, 45, 55))   # 15 px(박스 폭의 1.5배)
        self.assertEqual(ids(df), [1, 2])

    def test_big_box_allows_longer_distance(self):
        # 같은 30 px라도 가까이 있어 박스가 큰 차(폭 40)는 잇고, 멀리 있어 작은 차(폭 10)는 잇지 않는다
        near = self.merge(rows(1, "vehicle", 0, 3, 0, 30, w=40), rows(2, "vehicle", 4, 5, 60, 70, w=40))
        far = self.merge(rows(1, "vehicle", 0, 3, 0, 30), rows(2, "vehicle", 4, 5, 60, 70))
        self.assertEqual((ids(near), ids(far)), ([1], [1, 2]))

    def test_very_different_size_not_merged(self):
        # 폭 400·높이 200인 오검출 박스(플래카드) 뒤 반경 안에서 시작한 폭 40·높이 20 차: 크기 조건이 없으면 잇고, 있으면 안 잇는다
        big, car = rows(1, "vehicle", 0, 3, box=(0, 0, 400, 200)), rows(2, "vehicle", 4, 5, box=(300, 280, 340, 300))
        df = pd.concat([big, car], ignore_index=True)
        self.assertEqual(ids(merge_tracks(df, 2.0, 1.0)), [1])
        self.assertEqual(ids(merge_tracks(df, 2.0, 1.0, size_min=0.5)), [1, 2])

    def test_one_side_similar_still_merged(self):
        # 화면 끝에서 들어오는 차: 폭만 좁은 조각(10x40) → 온전한 박스(40x40). 높이가 같아 크기 조건을 넘는다
        sliver, full = rows(1, "vehicle", 0, 3, box=(0, 0, 10, 40)), rows(2, "vehicle", 3.2, 5, box=(-10, 0, 30, 40))
        self.assertEqual(ids(merge_tracks(pd.concat([sliver, full], ignore_index=True), 2.0, 1.0, size_min=0.5)), [1])

    def test_other_class_not_merged(self):
        df = self.merge(rows(1, "vehicle", 0, 3, 0, 30), rows(2, "pedestrian", 4, 5, 30, 30))
        self.assertEqual(ids(df), [1, 2])

    def test_overlapping_not_merged(self):
        df = self.merge(rows(1, "vehicle", 0, 3, 0, 30), rows(2, "vehicle", 2, 5, 30, 40))   # 동시에 있던 두 차
        self.assertEqual(ids(df), [1, 2])

    def test_chain_three_parts(self):
        df = self.merge(rows(5, "vehicle", 0, 2, 0, 20), rows(9, "vehicle", 3, 5, 25, 40), rows(7, "vehicle", 6, 8, 45, 60))
        self.assertEqual(ids(df), [5])

    def test_nearest_candidate(self):
        a, b = rows(1, "vehicle", 0, 2, 0, 0), rows(2, "vehicle", 0, 2, 18, 18)
        c = rows(3, "vehicle", 3, 4, 10, 10)               # 1에서 10 px, 2에서 8 px → 둘 다 후보, 2가 더 가깝다
        df = self.merge(a, b, c)
        self.assertEqual(df.loc[df["frame"] == 35, "track_id"].tolist(), [2])


class OverlapMergeTest(unittest.TestCase):
    """B가 A 끝나기 직전에 시작한 경우 (한 물체에 ID가 둘 붙었다가 A가 끝남)."""
    A_BOX, SAME, OTHER = (0, 0, 10, 10), (1, 0, 11, 10), (8, 0, 18, 10)   # IoU 0.82 / 0.11

    def merge(self, b_start, b_box, classes=("vehicle",), cls="vehicle"):
        a = rows(1, cls, 0, 3, 0, 3, box=self.A_BOX)
        b = rows(2, cls, b_start, 6, 3, 6, box=b_box)
        return merge_tracks(pd.concat([a, b], ignore_index=True), 2.0, 1.0,
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
        df = pd.concat([rows(1, "vehicle", 0, 0.6, 0, 6), rows(2, "vehicle", 1.5, 2.0, 10, 15)], ignore_index=True)
        merged = merge_tracks(df, 2.0, 1.0)
        self.assertEqual(ids(drop_short(merged, 1.0)), [1])   # 조각은 각각 1초 미만이지만 합치면 2초
        self.assertEqual(ids(drop_short(df, 1.0)), [])        # 먼저 지우면 둘 다 사라진다


class DetectAndTrackArgsTest(unittest.TestCase):
    def test_passes_nms_settings_and_no_conf(self):
        # 겹친 박스 합치기 설정은 detect.yaml 값을 그대로 넘기고, conf는 넘기지 않는다(ByteTrack이 낮은 신뢰도 박스로 트랙을 잇는다)
        cfg = {"model": "m.pt", "class_map": {"car": "vehicle"}, "tracker": "bytetrack.yaml", "imgsz": 1920,
               "nms_agnostic": True, "nms_iou": 0.5}
        with mock.patch("tracking.track.YOLO") as yolo, mock.patch("tracking.track.cv2.VideoCapture") as cap:
            cap.return_value.get.return_value = 30.0
            yolo.return_value.names = {0: "car"}
            yolo.return_value.track.return_value = []
            detect_and_track("v.mp4", cfg)
        kwargs = yolo.return_value.track.call_args.kwargs
        self.assertEqual((kwargs["agnostic_nms"], kwargs["iou"]), (True, 0.5))
        self.assertNotIn("conf", kwargs)


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

    def test_no_measurement_leaves_meters_empty(self):
        # 실측값이 없으면 homography 없이 미터 좌표만 비운다(기술서 11.1: 판정·추적은 픽셀)
        site = {"crosswalk": {"corners_px": [[0, 0], [100, 0], [100, 100], [0, 100]], "width_m": 0.0, "length_m": 0.0}}
        self.assertIsNone(load_homography(site))
        self.assertIsNone(load_homography({"crosswalk": {"corners_px": site["crosswalk"]["corners_px"]}}))
        df = add_meters(rows(1, "vehicle", 0, 0, box=(40, 0, 60, 50)), None)
        self.assertTrue(df[["bx_m", "by_m"]].isna().all().all())

    def test_measurement_gives_homography(self):
        site = {"crosswalk": {"corners_px": [[0, 0], [100, 0], [100, 100], [0, 100]], "width_m": 4.0, "length_m": 10.0}}
        self.assertIsNotNone(load_homography(site))

    def test_schema_columns(self):
        self.assertEqual(COLUMNS, ["frame", "t_s", "track_id", "cls", "coco_cls", "conf",
                                   "x1", "y1", "x2", "y2", "bx_m", "by_m"])


if __name__ == "__main__":
    unittest.main()
