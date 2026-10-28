# 담당: 육주영
"""규칙 기반 판정기 (기준선). docs/judgment_criteria.md의 기준을 그대로 코드로 옮긴다.

입력: outputs/features/<영상>.csv, configs/site_<교차로>.yaml (정지 기준값)
출력: outputs/judged/<영상>.csv (컬럼은 docs/csv_schema.md 4절)
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
