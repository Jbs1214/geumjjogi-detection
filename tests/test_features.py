# 담당: 김민영
"""features/build.py 테스트 (가짜 트랙, 실제 영상 없이). 실행: python -m unittest discover -s tests -t . -v
화면 10 px = 1 m. 출구(0~100 px), near(200~300 px), diag(400~500 px). 차량은 y=50에서 프레임마다 10 px(10 m/s) 왼쪽으로 간다:
near 진입 frame 5, 출구 진입 frame 25 (t_enter_s 2.5). diag에는 들어가지 않는다."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from features.build import NEW_COLUMNS, build_features
from tracking.homography import compute_homography, to_meters

FPS = 10.0
EXIT = [[0, 0], [100, 0], [100, 100], [0, 100]]
SITE = {"crosswalk": {"corners_px": EXIT, "area_px": [], "width_m": 10.0, "length_m": 10.0},
        "other_crosswalks_px": {"near": [[200, 0], [300, 0], [300, 100], [200, 100]],
                                "diag": [[400, 0], [500, 0], [500, 100], [400, 100]]},
        "occluded_edge_m": 1.0}
CONFIG = {"merge_gap_s": 2.0, "speed_window_s": 0.5}
H = compute_homography(EXIT, 10.0, 10.0)


def track(tid, cls, frames, points):
    """frames마다 하단 중심 (x, y) 픽셀. points가 (x, y) 하나면 그 자리에 서 있다."""
    frames = list(frames)
    pts = np.array(points if isinstance(points, list) else [points] * len(frames), dtype=float)
    m = to_meters(H, pts)
    return pd.DataFrame({
        "frame": frames, "t_s": np.array(frames) / FPS, "track_id": tid, "cls": cls,
        "coco_cls": "car" if cls == "vehicle" else "person", "conf": 0.9,
        "x1": pts[:, 0] - 5, "y1": pts[:, 1] - 30, "x2": pts[:, 0] + 5, "y2": pts[:, 1], "bx_m": m[:, 0], "by_m": m[:, 1],
    })


CAR = track(1, "vehicle", range(41), [(350 - 10 * f, 50) for f in range(41)])
EVENTS = pd.DataFrame({"video": ["A_20261031_0800"], "event_id": [1], "track_id": [1], "cls": ["vehicle"],
                       "coco_cls": ["car"], "t_enter_s": [2.5], "t_stopline_s": [np.nan]})


def run(*peds, signal=None):
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        pd.concat([CAR, *peds]).to_csv(d / "tracks.csv", index=False, encoding="utf-8-sig")
        EVENTS.to_csv(d / "A_20261031_0800.csv", index=False, encoding="utf-8-sig")
        (d / "site.yaml").write_text(yaml.safe_dump(SITE), encoding="utf-8")
        (d / "detect.yaml").write_text(yaml.safe_dump(CONFIG), encoding="utf-8")
        out = build_features(str(d / "A_20261031_0800.csv"), str(d / "tracks.csv"), str(d / "site.yaml"),
                             signal, str(d / "detect.yaml"), out_dir=str(d / "out"))
        return pd.read_csv(out, encoding="utf-8-sig", keep_default_na=False).iloc[0]


class PedsAtEntryTest(unittest.TestCase):
    def test_ped_on_exit_at_entry(self):
        r = run(track(101, "pedestrian", range(61), (50, 80)))
        self.assertEqual((r["peds_at_entry"], r["no_yield_at"]), (1, "exit"))

    def test_ped_left_before_entry(self):
        walk = [(50, 80 + 4 * f) if f <= 20 else (50, 160) for f in range(61)]   # frame 20에 횡단보도 밖
        self.assertEqual(run(track(102, "pedestrian", range(61), walk))["peds_at_entry"], 0)

    def test_ped_on_near_at_near_entry(self):
        r = run(track(103, "pedestrian", range(11), (250, 80)))               # near 진입(frame 5) 때만 있음
        self.assertEqual((r["peds_at_entry"], r["no_yield_at"]), (1, "near"))

    def test_crosswalk_not_entered_is_ignored(self):
        self.assertEqual(run(track(104, "pedestrian", range(61), (450, 50)))["peds_at_entry"], 0)

    def test_one_missing_frame_is_interpolated(self):
        frames = [f for f in range(61) if f != 25]                           # 출구 진입 순간만 검출 빠짐
        self.assertEqual(run(track(105, "pedestrian", frames, (50, 20)))["peds_at_entry"], 1)

    def test_long_gap_not_interpolated(self):
        frames = [*range(3), *range(48, 61)]                                  # 4.5초 빔
        self.assertEqual(run(track(106, "pedestrian", frames, (50, 20)))["peds_at_entry"], 0)

    def test_two_crosswalks_summed_in_time_order(self):
        r = run(track(101, "pedestrian", range(61), (50, 80)), track(103, "pedestrian", range(11), (250, 80)))
        self.assertEqual((r["peds_at_entry"], r["no_yield_at"]), (2, "near;exit"))


class PedsLostTest(unittest.TestCase):
    def test_ended_inside_center_is_lost(self):
        r = run(track(107, "pedestrian", range(16), (50, 50)))                # t 1.5에 가운데(경계에서 5 m)에서 끝남
        self.assertEqual((r["peds_lost"], r["peds_at_entry"]), (1, 0))

    def test_ended_near_edge_is_not_lost(self):
        self.assertEqual(run(track(108, "pedestrian", range(16), (95, 50)))["peds_lost"], 0)   # 경계에서 0.5 m

    def test_ended_too_early_is_not_lost(self):
        self.assertEqual(run(track(109, "pedestrian", range(4), (50, 50)))["peds_lost"], 0)    # 진입 2초 전보다 먼저


class OutputTest(unittest.TestCase):
    def test_entry_speed(self):
        self.assertAlmostEqual(run()["entry_speed"], 10.0, places=1)

    def test_columns_and_stage2_empty(self):
        r = run()
        self.assertEqual(list(r.index), list(EVENTS.columns) + NEW_COLUMNS)
        self.assertEqual((r["signal_at_stopline"], r["stop_sec_before_line"], r["no_yield_at"]), ("", "", ""))

    def test_signal_not_implemented(self):
        with self.assertRaises(SystemExit):
            run(signal="outputs/signal/A_20261031_0800.csv")


if __name__ == "__main__":
    unittest.main()
