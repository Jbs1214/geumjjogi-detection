# 담당: 육주영
"""미양보·미정지(위반 의심) 이벤트 구간을 잘라 클립으로 저장한다.
모든 클립은 사람 박스 전체(운전자로 뺀 사람 포함)와 차량·오토바이 박스 아래 40%(번호판 위치)를
ROI 밖까지 프레임 전체에서 블러한다. 끊긴 프레임은 앞뒤 박스로 보간하고, 박스에 여백을 둔다.
블러 전 프레임은 파일로 남기지 않는다(기술서 11.1). 수치는 configs/eval.yaml의 blur_*.
블러용 박스는 tracks CSV(운전자 제외, conf 0.25)를 쓰지 않고, 원본 영상을 configs/detect.yaml의 model로
configs/eval.yaml의 blur_conf 기준으로 다시 검출해 얻는다(운전자 포함).

입력: outputs/judged/<영상>.csv (구간·판정), outputs/tracks/<영상>.csv (해당 차량 궤적), 원본 영상(--video)
출력: outputs/clips/<영상>_<event_id>.mp4, judged CSV의 clip 컬럼에 클립 경로를 채운다

지금은 1단계 미양보(no_yield)만 자른다. 미정지 클립은 MVP 뒤에 추가한다.
클립 구간은 접근~진입 앞뒤로 clip_pad_s(configs/eval.yaml)를 붙인 것이다. tracks의 frame은 0부터 세는 프레임 번호로 본다.
"""
import argparse
from pathlib import Path

import cv2
import pandas as pd
import yaml

from eval.blur import Box, blur_frame, detect_boxes, fill_gaps

TARGET_COLOR = (0, 0, 255)  # 위반 의심 차량 테두리 (표시용)


def write_clip(
    video: str,
    start: int,
    end: int,
    blur_boxes: list[list[Box]],
    target_rows: pd.DataFrame,
    eval_cfg: dict,
    out_path: str,
) -> None:
    """start~end(끝 제외) 프레임을 블러한 뒤 해당 차량 박스를 표시해 저장한다."""
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"출력 영상을 열 수 없다: {out_path}")

    rows_by_frame = {f: g for f, g in target_rows.groupby("frame")}
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for i in range(end - start):
        ok, frame = cap.read()
        if not ok:
            break
        if i < len(blur_boxes):
            blur_frame(frame, blur_boxes[i], eval_cfg)
        if start + i in rows_by_frame:
            for r in rows_by_frame[start + i].itertuples():
                cv2.rectangle(frame, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), TARGET_COLOR, 2)
        writer.write(frame)

    cap.release()
    writer.release()


def make_clips(
    judged: str, tracks: str, video: str, config: str, detect: str, out_dir: str = "outputs/clips"
) -> None:
    with open(config, encoding="utf-8") as fp:
        eval_cfg = yaml.safe_load(fp)
    with open(detect, encoding="utf-8") as fp:
        detect_cfg = yaml.safe_load(fp)
    judged_df = pd.read_csv(judged, encoding="utf-8-sig")
    tracks_df = pd.read_csv(tracks, encoding="utf-8-sig")

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    targets = judged_df[judged_df["ped_result"] == "no_yield"]
    clips = {}
    for n, r in enumerate(targets.itertuples(), 1):
        start = max(int((r.t_approach_s - eval_cfg["clip_pad_s"]) * fps), 0)
        end = min(int((r.t_enter_s + eval_cfg["clip_pad_s"]) * fps) + 1, total)
        blur_boxes = fill_gaps(
            detect_boxes(video, detect_cfg, eval_cfg["blur_conf"], start, end),
            int(eval_cfg["blur_interp_max_s"] * fps),
        )
        out_path = out / f"{Path(video).stem}_{r.event_id}.mp4"
        write_clip(
            video, start, end, blur_boxes, tracks_df[tracks_df["track_id"] == r.track_id], eval_cfg, str(out_path)
        )
        clips[r.event_id] = out_path.as_posix()
        print(f"클립 {n}/{len(targets)}: {out_path.as_posix()}")

    judged_df["clip"] = judged_df["event_id"].map(clips).fillna("")
    judged_df.to_csv(judged, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="위반 의심 클립 추출")
    parser.add_argument("--judged", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--detect", default="configs/detect.yaml")
    args = parser.parse_args()
    make_clips(args.judged, args.tracks, args.video, args.config, args.detect)


if __name__ == "__main__":
    main()
