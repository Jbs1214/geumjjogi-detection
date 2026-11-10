# 담당: 육주영
"""분석 영상(mp4)을 만든다. 중간 발표(11.18) 시연용, 11.16까지.
블러(eval/clips.py와 같은 규칙, configs/eval.yaml의 blur_*)를 먼저 적용한 뒤 박스·궤적·횡단보도·판정 결과를 그린다. 블러 전 프레임은 파일로 남기지 않는다.
블러용 박스는 tracks CSV(운전자 제외, conf 0.25)를 쓰지 않고, 원본 영상을 configs/detect.yaml의 model로
configs/eval.yaml의 blur_conf 기준으로 다시 검출해 얻는다(운전자 포함).

입력: 원본 영상(--video), outputs/tracks/<영상>.csv, outputs/judged/<영상>.csv, configs/site_<영상>.yaml
출력: outputs/overlay/<영상>.mp4

tracks의 frame은 0부터 세는 프레임 번호로 본다. 판정 글자는 영어로 쓴다(OpenCV 글자 그리기가 한글을 지원하지 않는다).
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from eval.blur import Box, blur_frame, detect_boxes, fill_gaps, print_progress

# 표시용 값 (판정 기준이 아님)
TRAIL_S = 2.0        # 궤적을 그리는 길이
LABEL_HOLD_S = 3.0   # 출구 횡단보도 진입 시각 앞뒤로 판정 글자를 보여 주는 시간
CROSSWALK_COLOR = (0, 255, 255)
CLS_COLORS = {"vehicle": (255, 160, 0), "motorcycle": (255, 0, 255), "pedestrian": (0, 255, 0)}
RESULT_COLORS = {"no_yield": (0, 0, 255), "yield": (0, 200, 0), "exclude": (160, 160, 160)}
RESULT_TEXT = {"no_yield": "NO YIELD", "yield": "YIELD", "exclude": "EXCLUDE"}


def render(
    video: str,
    blur_boxes: list[list[Box]],
    tracks: pd.DataFrame,
    judged: pd.DataFrame,
    site_cfg: dict,
    eval_cfg: dict,
    out_path: str,
) -> None:
    """프레임마다 블러를 먼저 적용하고 횡단보도·박스·궤적·판정을 그려 저장한다."""
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    size =(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"출력 영상을 열 수 없다: {out_path}")

    # 출구 횡단보도 + 우회전 경로가 지나는 다른 횡단보도
    crosswalks = [site_cfg["crosswalk"]["corners_px"], *site_cfg.get("other_crosswalks_px", {}).values()]
    crosswalks = [np.array(c, np.int32) for c in crosswalks]
    rows_by_frame = {f: g for f, g in tracks.groupby("frame")}
    trail_frames = int(TRAIL_S * fps)
    trails: dict[int, list[tuple[int, int, int]]] = {}  # track_id -> [(frame, x, y)]

    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if f < len(blur_boxes):
            blur_frame(frame, blur_boxes[f], eval_cfg)
        cv2.polylines(frame, crosswalks, True, CROSSWALK_COLOR, 2)

        t = f / fps
        active = judged[(judged["t_enter_s"] - t).abs() <= LABEL_HOLD_S]
        result_by_track = dict(zip(active["track_id"], active["ped_result"]))

        if f in rows_by_frame:
            for r in rows_by_frame[f].itertuples():
                x1, y1, x2, y2 = int(r.x1), int(r.y1), int(r.x2), int(r.y2)
                result = result_by_track.get(r.track_id)
                color = RESULT_COLORS[result] if result else CLS_COLORS[r.cls]
                text = f"{r.track_id} {RESULT_TEXT[result]}" if result else str(r.track_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                trail = trails.setdefault(r.track_id, [])
                trail.append((f, (x1 + x2) // 2, y2))  # 박스 하단 중심점
                trail[:] = [p for p in trail if f - p[0] <= trail_frames]
                if len(trail) > 1:
                    cv2.polylines(frame, [np.array([p[1:] for p in trail], np.int32)], False, color, 2)

        writer.write(frame)
        f += 1
        print_progress("블러·저장", f, total)

    cap.release()
    writer.release()


def make_overlay(
    video: str, tracks: str, judged: str, site: str, config: str, detect: str, out_dir: str = "outputs/overlay"
) -> None:
    with open(site, encoding="utf-8") as fp:
        site_cfg = yaml.safe_load(fp)
    with open(config, encoding="utf-8") as fp:
        eval_cfg = yaml.safe_load(fp)
    with open(detect, encoding="utf-8") as fp:
        detect_cfg = yaml.safe_load(fp)

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    blur_boxes = fill_gaps(
        detect_boxes(video, detect_cfg, eval_cfg["blur_conf"]), int(eval_cfg["blur_interp_max_s"] * fps)
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    render(
        video,
        blur_boxes,
        pd.read_csv(tracks, encoding="utf-8-sig"),
        pd.read_csv(judged, encoding="utf-8-sig"),
        site_cfg,
        eval_cfg,
        str(out / f"{Path(video).stem}.mp4"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="분석 영상 만들기")
    parser.add_argument("--video", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--judged", required=True)
    parser.add_argument("--site", required=True)
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--detect", default="configs/detect.yaml")
    args = parser.parse_args()
    make_overlay(args.video, args.tracks, args.judged, args.site, args.config, args.detect)


if __name__ == "__main__":
    main()
