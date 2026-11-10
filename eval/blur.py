# 담당: 육주영
"""블러 공용 로직 (eval.clips, eval.overlay가 같이 쓴다). 기술서 11.1 블러 방법.

사람 박스 전체(운전자 포함)와 차량·오토바이 박스 아래 blur_plate_frac(번호판 위치)을 프레임 전체에서 블러한다.
블러용 박스는 tracks CSV를 쓰지 않고 원본 영상을 configs/detect.yaml의 model로 blur_conf 기준으로 다시 검출해 얻는다.
검출 결과와 블러 전 프레임은 메모리에서만 다루고 파일로 남기지 않는다. 수치는 configs/eval.yaml의 blur_*.
"""
import math

import cv2
import numpy as np
from ultralytics import YOLO

Box = tuple[str, float, float, float, float]  # (COCO 클래스, x1, y1, x2, y2) 픽셀


def detect_boxes(video: str, detect_cfg: dict, blur_conf: float) -> list[list[Box]]:
    """영상 전체를 검출해 프레임별 박스 목록을 돌려준다. 추적을 쓰지 않아 운전자도 포함된다."""
    model = YOLO(detect_cfg["model"])
    class_ids = [i for i, name in model.names.items() if name in detect_cfg["class_map"]]
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    frames = []
    results = model.predict(
        source=video, stream=True, conf=blur_conf, imgsz=detect_cfg["imgsz"], classes=class_ids, verbose=False
    )
    for r in results:
        frames.append([
            (model.names[int(c)], *xyxy) for c, xyxy in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist())
        ])
        print_progress("1/2 검출", len(frames), total)
    return frames


def print_progress(label: str, done: int, total: int) -> None:
    """진행률을 한 줄에 덮어 써서 보여 준다 (100프레임마다, 그리고 마지막 프레임에서)."""
    if done % 100 == 0 or done == total:
        print(f"\r{label} {done}/{total} 프레임 ({done * 100 // total}%)", end="\n" if done == total else "", flush=True)


def fill_gaps(boxes_per_frame: list[list[Box]], max_gap_frames: int) -> list[list[Box]]:
    """검출이 끊긴 프레임을 앞뒤 박스로 선형 보간해 채운다.
    사람끼리 또는 차량끼리, 중심 거리가 두 박스의 긴 변 이내인 가장 가까운 박스를 같은 대상으로 잇는다."""
    out = [list(boxes) for boxes in boxes_per_frame]
    open_ends: list[tuple[int, Box]] = []  # (마지막으로 보인 프레임, 그때 박스)
    for f, boxes in enumerate(boxes_per_frame):
        open_ends = [(lf, lb) for lf, lb in open_ends if f - lf - 1 <= max_gap_frames]
        next_ends = []
        for box in boxes:
            best = None
            for i, (_, last) in enumerate(open_ends):
                if (box[0] == "person") != (last[0] == "person"):
                    continue
                dist = math.dist(_center(box), _center(last))
                if dist <= max(_long_side(box), _long_side(last)) and (best is None or dist < best[0]):
                    best = (dist, i)
            if best is not None:
                last_f, last = open_ends.pop(best[1])
                for g in range(last_f + 1, f):
                    w = (g - last_f) / (f - last_f)
                    out[g].append((last[0], *(a + (b - a) * w for a, b in zip(last[1:], box[1:]))))
            next_ends.append((f, box))
        open_ends += next_ends
    return out


def blur_frame(frame: np.ndarray, boxes: list[Box], cfg: dict) -> None:
    """프레임을 그 자리에서 블러한다. 사람은 박스 전체, 그 밖(차량·오토바이)은 박스 아래 blur_plate_frac."""
    h, w = frame.shape[:2]
    for cls, x1, y1, x2, y2 in boxes:
        mx, my = (x2 - x1) * cfg["blur_margin"], (y2 - y1) * cfg["blur_margin"]
        x1, y1, x2, y2 = x1 - mx, y1 - my, x2 + mx, y2 + my
        if cls != "person":
            y1 = y2 - (y2 - y1) * cfg["blur_plate_frac"]
        x1, y1 = max(int(x1), 0), max(int(y1), 0)
        x2, y2 = min(math.ceil(x2), w), min(math.ceil(y2), h)
        if x2 <= x1 or y2 <= y1:
            continue
        k = max(int(min(x2 - x1, y2 - y1) * cfg["blur_kernel_frac"]) | 1, 3)  # 홀수
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (k, k), 0)


def _center(box: Box) -> tuple[float, float]:
    return (box[1] + box[3]) / 2, (box[2] + box[4]) / 2


def _long_side(box: Box) -> float:
    return max(box[3] - box[1], box[4] - box[2])
