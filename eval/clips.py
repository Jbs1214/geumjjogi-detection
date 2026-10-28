# 담당: 육주영
"""미양보 이벤트 구간을 잘라 클립으로 저장한다.
모든 클립은 보행자 박스 전체와 차량·오토바이 박스 아래 40%(번호판 위치)를 블러한다.

입력: outputs/judged/<영상>.csv, outputs/tracks/<영상>.csv, 원본 영상(--video)
출력: outputs/clips/<영상>_<event_id>.mp4
"""
import argparse


def make_clips(judged: str, tracks: str, video: str) -> None:
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="미양보 클립 추출")
    parser.add_argument("--judged", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--video", required=True)
    args = parser.parse_args()
    make_clips(args.judged, args.tracks, args.video)


if __name__ == "__main__":
    main()
