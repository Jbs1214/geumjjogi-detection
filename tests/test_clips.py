# 담당: 육주영
"""eval/clips.py 테스트 (검출만 가짜로 대체, 실제 영상 없이). 실행: python -m unittest discover -s tests -t . -v"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import pandas as pd
import yaml

from eval.clips import make_clips

EVAL_CFG = {
    "blur_conf": 0.1, "blur_interp_max_s": 2.0, "blur_margin": 0.15,
    "blur_kernel_frac": 0.5, "blur_plate_frac": 0.4, "clip_pad_s": 0.5,
}
N_FRAMES, FPS, SIZE = 50, 10.0, (160, 120)


def fake_detect(video, detect_cfg, blur_conf, start, end):
    return [[("car", 10, 30, 50, 70)] for _ in range(end - start)]


def count_frames(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    return n


class MakeClipsTest(unittest.TestCase):
    def test_only_no_yield_events_get_clips(self):
        judged = pd.DataFrame({
            "video": "v",
            "event_id": [1, 2, 3],
            "track_id": [7, 8, 9],
            "t_enter_s": [2.0, 2.0, 4.75],
            "ped_result": ["no_yield", "yield", "no_yield"],
            "clip": "",
        })
        tracks = pd.DataFrame({
            "frame": range(15, 26), "track_id": 7, "cls": "vehicle", "x1": 10, "y1": 30, "x2": 50, "y2": 70,
        })
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            writer = cv2.VideoWriter(str(tmp / "v.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
            rng = np.random.default_rng(0)
            for _ in range(N_FRAMES):
                writer.write(rng.integers(0, 256, (SIZE[1], SIZE[0], 3), dtype=np.uint8))
            writer.release()
            judged.to_csv(tmp / "judged.csv", index=False, encoding="utf-8-sig")
            tracks.to_csv(tmp / "tracks.csv", index=False, encoding="utf-8-sig")
            (tmp / "eval.yaml").write_text(yaml.safe_dump(EVAL_CFG), encoding="utf-8")
            (tmp / "detect.yaml").write_text(yaml.safe_dump({}), encoding="utf-8")

            with mock.patch("eval.clips.detect_boxes", side_effect=fake_detect):
                make_clips(
                    str(tmp / "judged.csv"), str(tmp / "tracks.csv"), str(tmp / "v.mp4"),
                    str(tmp / "eval.yaml"), str(tmp / "detect.yaml"), str(tmp / "clips"),
                )

            self.assertEqual(sorted(p.name for p in (tmp / "clips").iterdir()), ["v_1.mp4", "v_3.mp4"])
            # 이벤트 1: 2.0초 ± 0.5초 → 프레임 15~25, 11장
            self.assertEqual(count_frames(tmp / "clips" / "v_1.mp4"), 11)
            # 이벤트 3: 4.75초 ± 0.5초는 영상 끝(프레임 50)을 넘으므로 프레임 42~49, 8장으로 잘린다
            self.assertEqual(count_frames(tmp / "clips" / "v_3.mp4"), 8)

            out = pd.read_csv(tmp / "judged.csv", encoding="utf-8-sig", keep_default_na=False)
            self.assertTrue(out["clip"][0].endswith("clips/v_1.mp4"))
            self.assertEqual(out["clip"][1], "")
            self.assertTrue(out["clip"][2].endswith("clips/v_3.mp4"))


if __name__ == "__main__":
    unittest.main()
