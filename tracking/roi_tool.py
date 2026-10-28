# 담당: 김민영
"""영상 첫 프레임에서 횡단보도 네 모서리와 진입로·출구 ROI를 클릭해 교차로 설정 파일을 만든다.

입력: 원본 영상(--video), 횡단보도 실측 치수(m)
출력: configs/site_<교차로>.yaml (형식은 configs/site_example.yaml)
"""
import argparse


def pick_points(video: str, out: str, width_m: float, length_m: float) -> None:
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="ROI·횡단보도 모서리 지정")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True, help="예: configs/site_A.yaml")
    parser.add_argument("--width-m", type=float, required=True, help="횡단보도 좌상→우상 변 실측 길이")
    parser.add_argument("--length-m", type=float, required=True, help="횡단보도 좌상→좌하 변 실측 길이")
    args = parser.parse_args()
    pick_points(args.video, args.out, args.width_m, args.length_m)


if __name__ == "__main__":
    main()
