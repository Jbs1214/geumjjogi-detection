# 담당: 김민영
"""궤적에서 우회전 이벤트를 ROI 규칙(진입로 → 출구 통과)으로 뽑는다.

입력: outputs/tracks/<영상>.csv, configs/site_<교차로>.yaml
출력: outputs/events/<영상>.csv (컬럼은 docs/csv_schema.md 2절)
"""
import argparse


def extract_events(tracks: str, site: str) -> None:
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="우회전 이벤트 추출")
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--site", required=True)
    args = parser.parse_args()
    extract_events(args.tracks, args.site)


if __name__ == "__main__":
    main()
