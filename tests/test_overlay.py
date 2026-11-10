# 담당: 육주영
"""eval/overlay.py 그리기 단계 테스트 (모델·실제 영상 없이). 실행: python -m unittest discover -s tests -t . -v"""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from eval.overlay import render

EVAL_CFG = {"blur_margin": 0.15, "blur_kernel_frac": 0.5, "blur_plate_frac": 0.4}
SITE_CFG = {"crosswalk": {"corners_px": [[20, 60], [140, 60], [140, 100], [20, 100]]}}
N_FRAMES, FPS, SIZE = 12, 10.0, (160, 120)


def write_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
    rng = np.random.default_rng(0)
    for _ in range(N_FRAMES):
        writer.write(rng.integers(0, 256, (SIZE[1], SIZE[0], 3), dtype=np.uint8))
    writer.release()


def read_frames(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


class RenderTest(unittest.TestCase):
    def test_output_matches_input_length_and_size(self):
        tracks = pd.DataFrame({
            "frame": range(N_FRAMES),
            "track_id": 7,
            "cls": "vehicle",
            "x1": [10 + 5 * f for f in range(N_FRAMES)],
            "y1": 30,
            "x2": [50 + 5 * f for f in range(N_FRAMES)],
            "y2": 70,
        })
        judged = pd.DataFrame({"track_id": [7], "t_approach_s": [0.2], "t_enter_s": [0.6], "ped_result": ["no_yield"]})
        blur_boxes = [[("car", 10 + 5 * f, 30, 50 + 5 * f, 70)] for f in range(N_FRAMES)]
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "src.mp4", Path(tmp) / "out.mp4"
            write_video(src)
            render(str(src), blur_boxes, tracks, judged, SITE_CFG, EVAL_CFG, str(out))
            frames = read_frames(out)
        self.assertEqual(len(frames), N_FRAMES)
        self.assertEqual(frames[0].shape, (SIZE[1], SIZE[0], 3))

    def test_empty_tracks_and_judged(self):
        tracks = pd.DataFrame(columns=["frame", "track_id", "cls", "x1", "y1", "x2", "y2"])
        judged = pd.DataFrame(columns=["track_id", "t_approach_s", "t_enter_s", "ped_result"])
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "src.mp4", Path(tmp) / "out.mp4"
            write_video(src)
            render(str(src), [], tracks, judged, SITE_CFG, EVAL_CFG, str(out))
            self.assertEqual(len(read_frames(out)), N_FRAMES)


if __name__ == "__main__":
    unittest.main()
