# 담당: 육주영
"""규칙 판정기. docs/judgment_criteria.md의 기준(보행자 1단계, 정지선 2단계)을 그대로 코드로 옮긴다.

입력: outputs/features/<영상>.csv, configs/site_<영상>.yaml (정지 기준값, stopline_zone_m, right_turn_signal, camera_moved_s)
제외 사유 값은 docs/csv_schema.md 5절 (occluded·camera_moved / signal_unknown·stopline_hidden·rturn_signal·camera_moved)
출력: outputs/judged/<영상>.csv (컬럼은 docs/csv_schema.md 5절)
"""
import argparse
import math
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

VEHICLE_TYPE = {"car": "승용", "bus": "버스", "truck": "트럭", "motorcycle": "오토바이"}


def judge_ped(peds_lost: int, peds_at_entry: int, camera_moved: bool) -> tuple[str, str]:
    """이벤트 1건의 (ped_result, ped_exclude_reason).
    peds_at_entry·peds_lost는 경로상 횡단보도 전부의 합이다(features.build). no_yield_at은 judged에 그대로 남는다.
    진입 순간 횡단보도 위에 보행자가 있으면 미양보, 없으면 양보(위반 아님).
    보행자가 보이면 가림 여부와 관계없이 미양보다. 보이지 않는데 가림이 의심되면 제외한다."""
    if camera_moved:
        return "exclude", "camera_moved"
    if peds_at_entry >= 1:
        return "no_yield", ""
    if peds_lost >= 1:
        return "exclude", "occluded"
    return "yield", ""


def judge_red(
    signal_at_stopline: str | float,  # CSV의 빈 칸은 pandas가 nan(float)으로 읽는다
    stop_sec_before_line: float,
    stop_min_duration_s: float,
    right_turn_signal: bool,
    camera_moved: bool,
) -> tuple[str, str]:
    """이벤트 1건의 (red_result, red_exclude_reason). 보행자 판정과 독립이다.
    정지선 통과 시각의 신호가 적색일 때만 판정하고, 녹색·황색(not_red)은 na다.
    적색인데 정지 시간을 잴 수 없으면(빈 값) 미정지로 단정하지 않고 제외한다."""
    if camera_moved:
        return "exclude", "camera_moved"
    if right_turn_signal:
        return "exclude", "rturn_signal"
    if signal_at_stopline == "not_red":
        return "na", ""
    if signal_at_stopline != "red":  # unknown이거나 신호 판정을 돌리지 않아 비어 있음
        return "exclude", "signal_unknown"
    if math.isnan(stop_sec_before_line):
        return "exclude", "stopline_hidden"
    if stop_sec_before_line >= stop_min_duration_s:
        return "stopped", ""
    return "not_stopped", ""


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

    moved = [in_camera_moved(t, cfg["camera_moved_s"]) for t in df["t_enter_s"]]
    results = [
        judge_ped(r.peds_lost, r.peds_at_entry, m)
        + judge_red(r.signal_at_stopline, r.stop_sec_before_line, cfg["stop_min_duration_s"], cfg["right_turn_signal"], m)
        for r, m in zip(df.itertuples(), moved)
    ]
    columns = ["ped_result", "ped_exclude_reason", "red_result", "red_exclude_reason"]
    df[columns] = pd.DataFrame(results, columns=columns, index=df.index)
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
