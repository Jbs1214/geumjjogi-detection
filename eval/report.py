# 담당: 육주영
"""시간대별 양보율 = 양보 / (양보 + 미양보)를 Wilson 95% 신뢰구간과 함께 표·그래프로 만든다.
시간대는 영상 파일 이름({교차로}_{YYYYMMDD}_{HHMM})에서 읽는다.

입력: outputs/judged/*.csv, data/labels/*.csv (정답 구간 비교용)
출력: outputs/report/yield_rate.csv, outputs/report/yield_rate.png, 지표 출력
"""
import argparse


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    raise NotImplementedError


def report(judged_dir: str, labels_dir: str) -> None:
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="시간대별 양보율 보고")
    parser.add_argument("--judged", default="outputs/judged")
    parser.add_argument("--labels", default="data/labels")
    args = parser.parse_args()
    report(args.judged, args.labels)


if __name__ == "__main__":
    main()
