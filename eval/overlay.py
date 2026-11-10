# 담당: 육주영
"""분석 영상(mp4)을 만든다. 중간 발표(11.18) 시연용, 11.16까지.
블러(eval/clips.py와 같은 규칙, configs/eval.yaml의 blur_*)를 먼저 적용한 뒤 박스·궤적·횡단보도·판정 결과를 그린다. 블러 전 프레임은 파일로 남기지 않는다.
블러용 박스는 tracks CSV(운전자 제외, conf 0.25)를 쓰지 않고, 원본 영상을 configs/detect.yaml의 model로
configs/eval.yaml의 blur_conf 기준으로 다시 검출해 얻는다(운전자 포함).

입력: 원본 영상(--video), outputs/tracks/<영상>.csv, outputs/judged/<영상>.csv, configs/site_<영상>.yaml
출력: outputs/overlay/<영상>.mp4
"""
import argparse


def make_overlay(video: str, tracks: str, judged: str, site: str) -> None:
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="분석 영상 만들기")
    parser.add_argument("--video", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--judged", required=True)
    parser.add_argument("--site", required=True)
    args = parser.parse_args()
    make_overlay(args.video, args.tracks, args.judged, args.site)


if __name__ == "__main__":
    main()
