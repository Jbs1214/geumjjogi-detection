# 담당: 육주영
"""eval/blur_video.py 테스트 (검출만 가짜로 대체, 실제 영상 없이). 실행: python -m unittest discover -s tests -t . -v"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import yaml

from eval.blur_video import blur_video

EVAL_CFG = {
    "blur_conf": 0.1, "blur_interp_max_s": 2.0, "blur_margin": 0.15,
    "blur_kernel_frac": 0.5, "blur_plate_frac": 0.4,
}
N_FRAMES, FPS, SIZE = 20, 10.0, (160, 120)


def fake_detect(video, detect_cfg, blur_conf):
    return [[("person", 20, 20, 80, 80)] for _ in range(N_FRAMES)]


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


class BlurVideoTest(unittest.TestCase):
    def test_whole_video_is_blurred_inside_boxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            writer = cv2.VideoWriter(str(tmp / "v.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
            rng = np.random.default_rng(0)
            for _ in range(N_FRAMES):
                writer.write(rng.integers(0, 256, (SIZE[1], SIZE[0], 3), dtype=np.uint8))
            writer.release()
            (tmp / "eval.yaml").write_text(yaml.safe_dump(EVAL_CFG), encoding="utf-8")
            (tmp / "detect.yaml").write_text(yaml.safe_dump({}), encoding="utf-8")

            with mock.patch("eval.blur_video.detect_boxes", side_effect=fake_detect):
                blur_video(str(tmp / "v.mp4"), str(tmp / "eval.yaml"), str(tmp / "detect.yaml"), str(tmp / "blur"))

            frames = read_frames(tmp / "blur" / "v_blur.mp4")
            self.assertEqual(len(frames), N_FRAMES)
            self.assertEqual(frames[0].shape[:2], (SIZE[1], SIZE[0]))
            # 잡음 영상이라 블러한 박스 안은 평평해지고, 박스 밖은 잡음이 그대로 남는다
            for frame in (frames[0], frames[-1]):
                self.assertLess(frame[30:70, 30:70].std(), frame[90:115, 90:155].std() / 2)


if __name__ == "__main__":
    unittest.main()
