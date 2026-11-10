# 담당: 김민영
"""이벤트마다 속도·거리·횡단보도 위 보행자 특징을 계산한다.
1단계는 횡단보도 진입 순간(t_enter_s)만 본다: 그때 횡단보도 위 보행자 수(peds_at_entry)와,
진입 직전 merge_gap_s 안에 횡단보도 경계에서 occluded_edge_m보다 안쪽에서 끊긴 보행자 트랙 수(peds_lost, 가림 의심).

입력: outputs/events/<영상>.csv, outputs/tracks/<영상>.csv, configs/site_<영상>.yaml,
      outputs/signal/<영상>.csv (2단계, 정지선 통과 시각의 신호 상태 조회)
출력: outputs/features/<영상>.csv (컬럼은 docs/csv_schema.md 4절)
"""
import argparse


def build_features(events: str, tracks: str, site: str, signal: str | None) -> None:
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="이벤트 특징 계산")
    parser.add_argument("--events", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--site", required=True)
    parser.add_argument("--signal", default=None, help="2단계 빨간불 시간표. MVP(1단계)에서는 생략")
    args = parser.parse_args()
    build_features(args.events, args.tracks, args.site, args.signal)


if __name__ == "__main__":
    main()
