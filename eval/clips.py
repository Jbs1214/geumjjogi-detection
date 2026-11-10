# 담당: 육주영
"""미양보·미정지(위반 의심) 이벤트 구간을 잘라 클립으로 저장한다.
모든 클립은 사람 박스 전체(운전자로 뺀 사람 포함)와 차량·오토바이 박스 아래 40%(번호판 위치)를
ROI 밖까지 프레임 전체에서 블러한다. 끊긴 프레임은 앞뒤 박스로 보간하고, 박스에 여백을 둔다.
블러 전 프레임은 파일로 남기지 않는다(기술서 11.1). 수치는 configs/eval.yaml의 blur_*.
블러용 박스는 tracks CSV(운전자 제외, conf 0.25)를 쓰지 않고, 원본 영상을 configs/detect.yaml의 model로
configs/eval.yaml의 blur_conf 기준으로 다시 검출해 얻는다(운전자 포함).

입력: outputs/judged/<영상>.csv (구간·판정), outputs/tracks/<영상>.csv (해당 차량 궤적), 원본 영상(--video)
출력: outputs/clips/<영상>_<event_id>.mp4
"""
import argparse


def make_clips(judged: str, tracks: str, video: str) -> None:
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="위반 의심 클립 추출")
    parser.add_argument("--judged", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--video", required=True)
    args = parser.parse_args()
    make_clips(args.judged, args.tracks, args.video)


if __name__ == "__main__":
    main()
