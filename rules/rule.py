# 담당: 육주영
"""규칙 판정기. docs/judgment_criteria.md의 기준(보행자 1단계, 정지선 2단계)을 그대로 코드로 옮긴다.

입력: outputs/features/<영상>.csv, configs/site_<영상>.yaml (정지 기준값, stopline_zone_m, right_turn_signal, camera_moved_s)
제외 사유 값은 docs/csv_schema.md 5절 (no_ped·occluded·camera_moved / signal_unknown·stopline_hidden·rturn_signal·camera_moved)
출력: outputs/judged/<영상>.csv (컬럼은 docs/csv_schema.md 5절)
"""
import argparse


def judge(features: str, site: str) -> None:
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="규칙 기반 양보 판정")
    parser.add_argument("--features", required=True)
    parser.add_argument("--site", required=True)
    args = parser.parse_args()
    judge(args.features, args.site)


if __name__ == "__main__":
    main()
