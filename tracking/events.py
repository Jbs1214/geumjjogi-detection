# 담당: 김민영
"""궤적에서 우회전 이벤트를 ROI 규칙(우회전 진입로 ROI를 지난 트랙이 출구 횡단보도에 진입)으로 뽑는다.
출구 ROI와 접근 구간은 쓰지 않는다(기술서 11.1).

입력: outputs/tracks/<영상>.csv, configs/site_<영상>.yaml
정지선 통과 시각(t_stopline_s)은 궤적 전체에서 찾는다.
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
