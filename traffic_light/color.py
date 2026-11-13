# 담당: 김민영
"""차량 신호등 색 판정 (2단계). 영상마다 지정한 신호등 영역의 빨간 픽셀 비율·HSV 규칙으로 판정한다.
연속 signal_min_frames 프레임이 같을 때만 상태를 바꾼다(초기값 5).
보행자 신호등은 쓰지 않는다. 녹색·황색은 구별하지 않고 not_red로 둔다(판정은 적색 여부만 쓴다, 기술서 2.3).
카메라가 움직인 구간(camera_moved_s)은 영역이 신호등에서 벗어나므로 unknown이다.

입력: 원본 영상(--video), configs/site_<영상>.yaml (signal_roi_px, signal_red_ratio_min, signal_min_frames, camera_moved_s)
출력: outputs/signal/<영상>.csv (컬럼은 docs/csv_schema.md 3절, 영상 전체의 빨간불 시간표)
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

COLUMNS = ["video", "t_start_s", "t_end_s", "state"]
# 빨간 픽셀의 색 범위 (OpenCV HSV, H는 0~179라 빨강이 양 끝에 걸친다)
RED_H_LOW, RED_H_HIGH, MIN_S, MIN_V = 10, 165, 90, 120


def red_ratio(frame: np.ndarray, poly: list) -> float:
    """영역(다각형) 안 픽셀 중 빨간 픽셀의 비율."""
    p = np.array(poly, np.int32)
    x0, y0 = p.min(axis=0)
    x1, y1 = p.max(axis=0) + 1
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(mask, [p - [x0, y0]], 1)
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)[mask > 0]
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    return float((((h < RED_H_LOW) | (h > RED_H_HIGH)) & (s > MIN_S) & (v > MIN_V)).mean())


def to_segments(is_red: list, fps: float, min_frames: int, moved: list) -> pd.DataFrame:
    """프레임별 빨강 여부 → 상태가 이어진 구간표(t_start_s, t_end_s, state). 구간은 0초~끝을 빈틈없이 덮는다.
    새 상태가 연속 min_frames 프레임 이어질 때만 바꾸고, 바뀐 시각은 그 연속 구간의 첫 프레임이다.
    moved([[시작초, 끝초], ...]) 구간의 프레임은 unknown."""
    cur, run, red = is_red[0], 0, np.empty(len(is_red), bool)
    for i, r in enumerate(is_red):
        run = run + 1 if r != cur else 0
        if run >= min_frames:
            cur, run = r, 0
            red[i - min_frames + 1:i] = cur  # 늦게 바뀐 만큼 앞으로 되돌려 적는다
        red[i] = cur
    state = np.where(red, "red", "not_red").astype(object)
    t = np.arange(len(is_red)) / fps
    for a, b in moved:
        state[(t >= a) & (t <= b)] = "unknown"
    start = np.flatnonzero(np.r_[True, state[1:] != state[:-1]])
    return pd.DataFrame({"t_start_s": start / fps, "t_end_s": np.r_[start[1:], len(state)] / fps, "state": state[start]})


def classify_signal(video: str, site: str, out_dir: str = "outputs/signal") -> Path:
    with open(site, encoding="utf-8") as f:
        s = yaml.safe_load(f)
    poly = s["signal_roi_px"]
    if not np.any(poly):
        raise SystemExit("signal_roi_px가 비어 있다. tracking.roi_tool --only signal로 신호등 영역을 먼저 지정한다")
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    is_red = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        is_red.append(red_ratio(frame, poly) >= s["signal_red_ratio_min"])
    cap.release()
    if not is_red:
        raise SystemExit(f"영상을 읽지 못했다: {video}")
    seg = to_segments(is_red, fps, s["signal_min_frames"], s["camera_moved_s"])
    seg = seg.assign(video=Path(video).stem)[COLUMNS]
    out = Path(out_dir) / f"{Path(video).stem}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    seg.round({"t_start_s": 3, "t_end_s": 3}).to_csv(out, index=False, encoding="utf-8-sig")
    dur = (seg["t_end_s"] - seg["t_start_s"]).groupby(seg["state"]).sum()
    print(f"신호 구간 {len(seg)}개, 상태별 시간(초) {dur.round(1).to_dict()}")
    print(f"저장: {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="차량 신호등 색 판정")
    parser.add_argument("--video", required=True)
    parser.add_argument("--site", required=True)
    args = parser.parse_args()
    classify_signal(args.video, args.site)


if __name__ == "__main__":
    main()
