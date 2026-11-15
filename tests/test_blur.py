# 담당: 육주영
"""eval/blur.py 테스트 (모델·실제 영상 없이). 실행: python -m unittest discover -s tests -t . -v"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

from eval.blur import blur_frame, detect_boxes, fill_gaps

CFG = {"blur_margin": 0.15, "blur_kernel_frac": 0.5, "blur_plate_frac": 0.4}


class FillGapsTest(unittest.TestCase):
    def test_linear_interpolation(self):
        frames = [[("pedestrian", 10, 10, 20, 30)], [], [], [], [("pedestrian", 18, 10, 28, 30)]]
        out = fill_gaps(frames, 3)
        self.assertEqual(out[2], [("pedestrian", 14.0, 10.0, 24.0, 30.0)])
        self.assertEqual([len(b) for b in out], [1, 1, 1, 1, 1])
        self.assertEqual(frames[2], [])  # 입력은 바꾸지 않는다

    def test_gap_longer_than_max_is_not_filled(self):
        frames = [[("pedestrian", 10, 10, 20, 30)], [], [], [], [("pedestrian", 18, 10, 28, 30)]]
        self.assertEqual([len(b) for b in fill_gaps(frames, 2)], [1, 0, 0, 0, 1])

    def test_person_and_vehicle_are_not_linked(self):
        frames = [[("pedestrian", 10, 10, 20, 30)], [], [("vehicle", 10, 10, 20, 30)]]
        self.assertEqual(fill_gaps(frames, 3)[1], [])

    def test_far_box_is_not_linked(self):
        frames = [[("pedestrian", 10, 10, 20, 30)], [], [("pedestrian", 200, 10, 210, 30)]]
        self.assertEqual(fill_gaps(frames, 3)[1], [])


class BlurFrameTest(unittest.TestCase):
    def setUp(self):
        self.original = np.random.default_rng(0).integers(0, 256, (200, 200, 3), dtype=np.uint8)

    def test_person_whole_box(self):
        frame = self.original.copy()
        blur_frame(frame, [("pedestrian", 50, 50, 100, 150)], CFG)
        # 여백 15%: x 42.5~107.5, y 35~165
        changed = frame != self.original
        self.assertTrue(changed[60:140, 55:95].any())
        self.assertFalse(changed[:35].any())
        self.assertFalse(changed[165:].any())
        self.assertFalse(changed[:, :42].any())
        self.assertFalse(changed[:, 108:].any())

    def test_vehicle_bottom_part_only(self):
        frame = self.original.copy()
        blur_frame(frame, [("vehicle", 20, 20, 120, 120)], CFG)
        # 여백 15%: 5~135, 높이 130의 아래 40% → y 83~135
        changed = frame != self.original
        self.assertTrue(changed[90:130, 10:130].any())
        self.assertFalse(changed[:83].any())
        self.assertFalse(changed[135:].any())

    def test_box_outside_frame_is_ignored(self):
        frame = self.original.copy()
        blur_frame(frame, [("pedestrian", 300, 300, 350, 400)], CFG)
        self.assertTrue((frame == self.original).all())


class DetectBoxesTest(unittest.TestCase):
    def test_returns_project_classes(self):
        """모델의 클래스 이름이 무엇이든 사람은 pedestrian으로 돌려준다(blur_frame이 이 이름으로 박스 전체를 블러한다)."""
        boxes = SimpleNamespace(cls=np.array([0.0, 1.0]), xyxy=np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]))
        for names, class_map in (
            ({0: "person", 1: "car"}, {"person": "pedestrian", "car": "vehicle"}),  # 사전학습(COCO)
            ({0: "pedestrian", 1: "vehicle"}, {"pedestrian": "pedestrian", "vehicle": "vehicle"}),  # 미세조정 3클래스
        ):
            model = SimpleNamespace(names=names, predict=lambda *a, **k: [SimpleNamespace(boxes=boxes)])
            with tempfile.TemporaryDirectory() as tmp:
                video = str(Path(tmp) / "v.mp4")
                writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
                writer.write(np.zeros((48, 64, 3), np.uint8))
                writer.release()
                with mock.patch("eval.blur.load_model", return_value=model):
                    out = detect_boxes(video, {"model": "x.pt", "class_map": class_map, "imgsz": 64}, 0.1)
            self.assertEqual(out, [[("pedestrian", 1.0, 2.0, 3.0, 4.0), ("vehicle", 5.0, 6.0, 7.0, 8.0)]])


if __name__ == "__main__":
    unittest.main()
