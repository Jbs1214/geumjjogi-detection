# 담당: 김민영
"""차량 신호등 색 판정 (2단계). 영상마다 지정한 신호등 영역의 빨간 픽셀 비율·HSV 규칙으로 판정한다.
연속 signal_min_frames 프레임이 같을 때만 상태를 바꾼다(초기값 5).
보행자 신호등은 쓰지 않는다.

입력: 원본 영상(--video), configs/site_<영상>.yaml (signal_roi_px, signal_min_frames)
출력: outputs/signal/<영상>.csv (컬럼은 docs/csv_schema.md 3절, 영상 전체의 빨간불 시간표)
"""
import argparse


def classify_signal(video: str, site: str) -> None:
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="차량 신호등 색 판정")
    parser.add_argument("--video", required=True)
    parser.add_argument("--site", required=True)
    args = parser.parse_args()
    classify_signal(args.video, args.site)


if __name__ == "__main__":
    main()
