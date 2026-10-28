# 담당: 김민영
"""이벤트마다 속도·거리·충돌 구간 보행자 특징을 계산한다.

입력: outputs/events/<영상>.csv, outputs/tracks/<영상>.csv, configs/site_<교차로>.yaml
출력: outputs/features/<영상>.csv (컬럼은 docs/csv_schema.md 3절)
"""
import argparse


def build_features(events: str, tracks: str, site: str) -> None:
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="이벤트 특징 계산")
    parser.add_argument("--events", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--site", required=True)
    args = parser.parse_args()
    build_features(args.events, args.tracks, args.site)


if __name__ == "__main__":
    main()
