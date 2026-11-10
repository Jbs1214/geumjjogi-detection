# 담당: 육주영
"""교차로·시간대별 위반율 = 미양보 / (양보 + 미양보)를 Wilson 95% 신뢰구간과 함께 표·그래프로 만든다.
판정 대상이 min_n_reference(configs/eval.yaml, 초기값 30) 미만인 칸은 "참고용"으로 표시한다. 적색 위반율(미정지 / (일시정지 + 미정지))은 따로 집계한다.
시간대는 영상 파일 이름({교차로}_{YYYYMMDD}_{HHMM})에서 읽는다.

입력: outputs/judged/*.csv, data/labels/*.csv (정답 구간 비교용), configs/eval.yaml
출력: outputs/report/yield_rate.csv, outputs/report/yield_rate.png, 지표 출력
"""
import argparse


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    raise NotImplementedError


def report(judged_dir: str, labels_dir: str, config: str) -> None:
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="교차로·시간대별 위반율 보고")
    parser.add_argument("--judged", default="outputs/judged")
    parser.add_argument("--labels", default="data/labels")
    parser.add_argument("--config", default="configs/eval.yaml")
    args = parser.parse_args()
    report(args.judged, args.labels, args.config)


if __name__ == "__main__":
    main()
