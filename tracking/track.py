# 담당: 김민영
"""YOLO + ByteTrack으로 차량·오토바이·보행자를 추적하고 박스 하단 중심점을 미터 좌표로 바꾼다.
오토바이 박스와 크게 겹치는 pedestrian은 운전자로 보고 뺀다.

입력: 원본 영상(--video), configs/site_<교차로>.yaml, configs/detect.yaml
출력: outputs/tracks/<영상>.csv (컬럼은 docs/csv_schema.md 1절)
"""
import argparse


def track(video: str, site: str, config: str) -> None:
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="검출·추적")
    parser.add_argument("--video", required=True)
    parser.add_argument("--site", required=True)
    parser.add_argument("--config", default="configs/detect.yaml")
    args = parser.parse_args()
    track(args.video, args.site, args.config)


if __name__ == "__main__":
    main()
