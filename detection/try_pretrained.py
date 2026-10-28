# 담당: 장범수
"""사전학습 YOLO를 촬영 영상에 그대로 적용해 본다.

입력: 원본 영상(--video), configs/detect.yaml
출력: outputs/detection/<영상>_sample.mp4 (박스를 그린 앞부분 영상), 클래스별 검출 수 출력
"""
import argparse


def run(video: str, config: str, seconds: int) -> None:
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="사전학습 YOLO 시험 적용")
    parser.add_argument("--video", required=True)
    parser.add_argument("--config", default="configs/detect.yaml")
    parser.add_argument("--seconds", type=int, default=60, help="앞에서부터 몇 초만 볼지")
    args = parser.parse_args()
    run(args.video, args.config, args.seconds)


if __name__ == "__main__":
    main()
