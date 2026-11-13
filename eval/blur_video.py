# 담당: 육주영
"""영상 전체를 블러해 공유·발표용 사본을 만든다 (판정 결과 없이 원본 영상만으로).
블러 기준은 eval/blur.py와 같다(기술서 11.1). 수치는 configs/eval.yaml의 blur_*, 검출은 configs/detect.yaml의 model·imgsz.
블러 전 프레임은 파일로 남기지 않는다.

입력: 원본 영상(--video)
출력: outputs/blur/<영상>_blur.mp4 (원본과 fps·해상도가 같다)
원본 헤더의 프레임 수보다 실제로 읽히는 프레임이 적으면 블러본은 읽힌 프레임까지만 담는다(끝부분 1~3프레임).
"""
import argparse
from pathlib import Path

import cv2
import yaml

from eval.blur import blur_frame, detect_boxes, fill_gaps, print_progress


def blur_video(video: str, config: str, detect: str, out_dir: str = "outputs/blur") -> None:
    with open(config, encoding="utf-8") as fp:
        eval_cfg = yaml.safe_load(fp)
    with open(detect, encoding="utf-8") as fp:
        detect_cfg = yaml.safe_load(fp)

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    blur_boxes = fill_gaps(
        detect_boxes(video, detect_cfg, eval_cfg["blur_conf"]), int(eval_cfg["blur_interp_max_s"] * fps)
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / f"{Path(video).stem}_blur.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"출력 영상을 열 수 없다: {out_path}")

    for i, boxes in enumerate(blur_boxes, 1):
        ok, frame = cap.read()
        if not ok:
            break
        blur_frame(frame, boxes, eval_cfg)
        writer.write(frame)
        print_progress("블러·저장", i, len(blur_boxes))

    cap.release()
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="영상 전체 블러본 만들기")
    parser.add_argument("--video", required=True)
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--detect", default="configs/detect.yaml")
    args = parser.parse_args()
    blur_video(args.video, args.config, args.detect)


if __name__ == "__main__":
    main()
