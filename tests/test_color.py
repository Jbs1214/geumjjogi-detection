# 담당: 김민영
"""traffic_light/color.py 테스트 (가짜 영상, 실제 영상 없이). 실행: python -m unittest discover -s tests -t . -v"""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from traffic_light.color import COLUMNS, classify_signal, red_ratio, to_segments

FPS = 10.0
POLY = [[10, 10], [29, 10], [29, 19], [10, 19]]  # 20x10 px
RED, GREEN, GRAY = (0, 0, 255), (0, 255, 0), (90, 90, 90)  # BGR


def frame(lamp) -> np.ndarray:
    """회색 바탕 60x40, 영역의 왼쪽 절반(10x10)에 램프 색."""
    img = np.full((40, 60, 3), GRAY, np.uint8)
    img[10:20, 10:20] = lamp
    return img


def run(lamps, **site):
    """lamps(프레임별 램프 색)로 영상을 만들어 classify_signal을 돌린 결과표."""
    cfg = {"signal_roi_px": POLY, "signal_red_ratio_min": 0.05, "signal_min_frames": 5, "camera_moved_s": [], **site}
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        writer = cv2.VideoWriter(str(d / "A_20261031_0800.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (60, 40))
        for lamp in lamps:
            writer.write(frame(lamp))
        writer.release()
        (d / "site.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
        out = classify_signal(str(d / "A_20261031_0800.mp4"), str(d / "site.yaml"), out_dir=str(d / "out"))
        return pd.read_csv(out, encoding="utf-8-sig")


class RedRatioTest(unittest.TestCase):
    def test_red_lamp_counts_only_red_pixels(self):
        self.assertAlmostEqual(red_ratio(frame(RED), POLY), 0.5, delta=0.02)  # 영역의 절반이 빨강

    def test_green_and_off_are_zero(self):
        self.assertEqual(red_ratio(frame(GREEN), POLY), 0.0)
        self.assertEqual(red_ratio(frame(GRAY), POLY), 0.0)


class SegmentsTest(unittest.TestCase):
    def test_short_flicker_is_ignored(self):
        seg = to_segments([True] * 10 + [False] * 3 + [True] * 10, FPS, 5, [])
        self.assertEqual(seg["state"].tolist(), ["red"])
        self.assertEqual((seg["t_start_s"][0], seg["t_end_s"][0]), (0.0, 2.3))

    def test_change_starts_at_first_frame_of_run(self):
        seg = to_segments([True] * 10 + [False] * 10, FPS, 5, [])
        self.assertEqual(seg["state"].tolist(), ["red", "not_red"])
        self.assertEqual(seg["t_start_s"].tolist(), [0.0, 1.0])  # 5프레임 늦은 1.4초가 아니다
        self.assertEqual(seg["t_end_s"].tolist(), [1.0, 2.0])  # 빈틈 없이 끝까지

    def test_camera_moved_is_unknown(self):
        seg = to_segments([True] * 30, FPS, 5, [[1.0, 1.9]])
        self.assertEqual(seg["state"].tolist(), ["red", "unknown", "red"])
        self.assertEqual(seg["t_start_s"].tolist(), [0.0, 1.0, 2.0])


class ClassifySignalTest(unittest.TestCase):
    def test_video_to_red_timetable(self):
        df = run([RED] * 20 + [GREEN] * 20)
        self.assertEqual(list(df.columns), COLUMNS)
        self.assertEqual(df["video"].unique().tolist(), ["A_20261031_0800"])
        self.assertEqual(df["state"].tolist(), ["red", "not_red"])
        self.assertAlmostEqual(df["t_start_s"][1], 2.0, delta=0.11)  # 압축으로 한 프레임 어긋날 수 있다
        self.assertEqual((df["t_start_s"][0], df["t_end_s"][1]), (0.0, 4.0))

    def test_unset_roi_stops(self):
        with self.assertRaises(SystemExit):
            run([RED] * 5, signal_roi_px=[[0, 0], [0, 0], [0, 0], [0, 0]])


if __name__ == "__main__":
    unittest.main()
