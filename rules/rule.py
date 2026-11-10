# 담당: 육주영
"""규칙 판정기. docs/judgment_criteria.md의 기준(보행자 1단계, 정지선 2단계)을 그대로 코드로 옮긴다.

입력: outputs/features/<영상>.csv, configs/site_<영상>.yaml (정지 기준값, stopline_zone_m, right_turn_signal, camera_moved_s)
제외 사유 값은 docs/csv_schema.md 5절 (occluded·camera_moved / signal_unknown·stopline_hidden·rturn_signal·camera_moved)
출력: outputs/judged/<영상>.csv (컬럼은 docs/csv_schema.md 5절)

지금은 1단계(보행자)만 판정한다. 2단계 컬럼(red_result, red_exclude_reason)은 MVP 뒤에 추가한다.
"""
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

VEHICLE_TYPE = {"car": "승용", "bus": "버스", "truck": "트럭", "motorcycle": "오토바이"}


def judge_ped(peds_lost: int, peds_at_entry: int, camera_moved: bool) -> tuple[str, str]:
    """이벤트 1건의 (ped_result, ped_exclude_reason).
    진입 순간 횡단보도 위에 보행자가 있으면 미양보, 없으면 양보(위반 아님).
    보행자가 보이면 가림 여부와 관계없이 미양보다. 보이지 않는데 가림이 의심되면 제외한다."""
    if camera_moved:
        return "exclude", "camera_moved"
    if peds_at_entry >= 1:
        return "no_yield", ""
    if peds_lost >= 1:
        return "exclude", "occluded"
    return "yield", ""


def in_camera_moved(t_enter_s: float, intervals: list) -> bool:
    """횡단보도 진입 시각이 카메라가 움직인 구간 안에 있는가."""
    return any(start <= t_enter_s <= end for start, end in intervals)


def judge(features: str, site: str, out_dir: str = "outputs/judged") -> None:
    df = pd.read_csv(features, encoding="utf-8-sig")
    with open(site, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    video = Path(features).stem  # 예: A_20261031_0800
    start = datetime.strptime(video.split("_", 1)[1], "%Y%m%d_%H%M")
    wall = start + pd.to_timedelta(df["t_enter_s"], unit="s")

    df["intersection"] = cfg["site"]
    df["wall_time"] = wall.dt.strftime("%Y-%m-%d %H:%M:%S")
    df["slot"] = wall.dt.hour.map(lambda h: f"{h:02d}–{h + 1:02d}시")
    df["vehicle_type"] = df["coco_cls"].map(VEHICLE_TYPE)

    results = [
        judge_ped(
            r.peds_lost,
            r.peds_at_entry,
            in_camera_moved(r.t_enter_s, cfg["camera_moved_s"]),
        )
        for r in df.itertuples()
    ]
    df[["ped_result", "ped_exclude_reason"]] = pd.DataFrame(
        results, columns=["ped_result", "ped_exclude_reason"], index=df.index
    )
    df["status"] = "auto"
    df["clip"] = ""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{video}.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="규칙 기반 양보 판정")
    parser.add_argument("--features", required=True)
    parser.add_argument("--site", required=True)
    args = parser.parse_args()
    judge(args.features, args.site)


if __name__ == "__main__":
    main()
