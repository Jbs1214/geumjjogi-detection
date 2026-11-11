# 담당: 김민영
"""이벤트마다 속도·거리·횡단보도 위 보행자 특징을 계산한다.
1단계는 경로상 각 횡단보도(crosswalk = exit, other_crosswalks_px)에 차량이 진입하는 순간만 본다:
그 위 보행자 수의 합(peds_at_entry), 보행자가 있던 횡단보도 이름(no_yield_at),
진입 직전 merge_gap_s 안에 그 횡단보도 경계에서 occluded_edge_m보다 안쪽에서 끊긴 보행자 트랙 수의 합(peds_lost, 가림 의심).

진입 순간: 출구는 이벤트의 t_enter_s, 다른 횡단보도는 그 차량 트랙이 출구 진입 전에 처음 들어간 프레임.
차가 들어가지 않은 횡단보도는 세지 않는다. 점은 박스 하단 중심(픽셀), 출구 영역은 area_px(없으면 corners_px).
그 프레임에 검출이 빠진 보행자 트랙은 앞뒤 행 간격이 merge_gap_s 이하이면 직선 보간한 위치로 센다
(검출이 잠깐 빠진 순간 서 있던 보행자가 0명으로 세어지지 않게. 트랙 이어 붙이기와 같은 기준).
entry_speed는 출구 진입 순간 속도(미터 좌표 speed_window_s 이동평균 후 차분, 참고값).
signal_at_stopline·stop_sec_before_line은 2단계라 지금은 비워 둔다.

입력: outputs/events/<영상>.csv, outputs/tracks/<영상>.csv, configs/site_<영상>.yaml,
      outputs/signal/<영상>.csv (2단계, 정지선 통과 시각의 신호 상태 조회)
출력: outputs/features/<영상>.csv (컬럼은 docs/csv_schema.md 4절)
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from tracking.events import exit_area, inside
from tracking.homography import to_meters
from tracking.track import load_homography

NEW_COLUMNS = ["entry_speed", "peds_lost", "peds_at_entry", "no_yield_at", "signal_at_stopline", "stop_sec_before_line"]


def crosswalks(site: dict) -> dict:
    """경로상 횡단보도 이름 → 픽셀 다각형. 출구 횡단보도 이름은 exit."""
    return {"exit": exit_area(site), **(site.get("other_crosswalks_px") or {})}


def ped_positions_at(peds: pd.DataFrame, frame: int, max_gap: int) -> pd.DataFrame:
    """frame 순간 보행자 트랙별 하단 중심(x, y). 그 프레임에 행이 없으면 앞뒤 행 간격이 max_gap 프레임 이하일 때만 직선 보간."""
    rows = []
    for tid, g in peds[peds["frame"].between(frame - max_gap, frame + max_gap)].groupby("track_id"):
        before, after = g[g["frame"] <= frame], g[g["frame"] >= frame]
        if before.empty or after.empty:
            continue
        b, a = before.iloc[-1], after.iloc[0]
        if a["frame"] - b["frame"] > max_gap:
            continue
        w = 0.0 if a["frame"] == b["frame"] else (frame - b["frame"]) / (a["frame"] - b["frame"])
        rows.append((tid, b["cx"] + w * (a["cx"] - b["cx"]), b["y2"] + w * (a["y2"] - b["y2"])))
    return pd.DataFrame(rows, columns=["track_id", "x", "y"])


def vehicle_entries(v: pd.DataFrame, polys: dict, exit_frame: int) -> dict:
    """횡단보도 이름 → 차량이 그 횡단보도에 처음 들어간 프레임. 출구는 exit_frame,
    다른 횡단보도는 출구 진입 전(같은 프레임 포함)에 들어간 것만. 안 들어간 횡단보도는 없다."""
    out = {"exit": exit_frame}
    before = v[v["frame"] <= exit_frame]
    for name, poly in polys.items():
        if name != "exit":
            hit = before[inside(poly, before["cx"], before["y2"])]
            if len(hit):
                out[name] = int(hit["frame"].min())
    return out


def count_lost(ends: pd.DataFrame, t_entry: float, poly_px: list, poly_m: np.ndarray, gap_s: float, edge_m: float) -> int:
    """진입 직전 gap_s 안에 끝난 보행자 트랙 중, 마지막 점이 횡단보도 안이고 경계에서 edge_m보다 안쪽인 것의 수(기술서 11.1).
    걸어 나간 사람은 경계 근처에서 끝나고, 안쪽에서 사라졌다면 진입 순간 가려져 있었을 수 있다."""
    e = ends[(ends["t_s"] >= t_entry - gap_s) & (ends["t_s"] < t_entry)]
    e = e[inside(poly_px, e["cx"], e["y2"])]
    return int(sum(cv2.pointPolygonTest(poly_m, (float(x), float(y)), True) > edge_m for x, y in zip(e["bx_m"], e["by_m"])))


def speed_at(v: pd.DataFrame, frame: int, window: int) -> float:
    """트랙 한 개(프레임 순)의 frame 순간 속도(m/s). 미터 좌표를 window 프레임 중심 이동평균한 뒤 차분한다."""
    sx = v["bx_m"].rolling(window, center=True, min_periods=1).mean()
    sy = v["by_m"].rolling(window, center=True, min_periods=1).mean()
    speed = (np.hypot(sx.diff(), sy.diff()) / v["t_s"].diff())[v["frame"] == frame]
    return float(speed.iloc[0]) if len(speed) else np.nan


def build_features(events: str, tracks: str, site: str, signal: str | None = None,
                   config: str = "configs/detect.yaml", out_dir: str = "outputs/features") -> Path:
    if signal:
        raise SystemExit("2단계(정지선 통과 시각의 신호 상태)는 아직 구현 전이다. --signal 없이 실행한다")
    with open(config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(site, encoding="utf-8") as f:
        s = yaml.safe_load(f)
    ev = pd.read_csv(events, encoding="utf-8-sig")
    tr = pd.read_csv(tracks, encoding="utf-8-sig")
    tr = tr.assign(cx=(tr["x1"] + tr["x2"]) / 2).sort_values(["track_id", "frame"])
    fps = (tr["frame"] / tr["t_s"])[tr["t_s"] > 0].median()
    gap = int(round(cfg["merge_gap_s"] * fps))
    window = max(1, int(round(cfg["speed_window_s"] * fps)))

    H = load_homography(s)
    polys = crosswalks(s)
    polys_m = {k: to_meters(H, np.array(p, dtype=float)).astype(np.float32) for k, p in polys.items()}
    peds = tr[tr["cls"] == "pedestrian"]
    ends = peds.groupby("track_id").tail(1)
    vehicles = dict(tuple(tr[tr["track_id"].isin(ev["track_id"])].groupby("track_id")))

    rows = []
    for e in ev.itertuples():
        v = vehicles[e.track_id]
        exit_frame = int(v["frame"].iloc[(v["t_s"] - e.t_enter_s).abs().argmin()])
        lost, at, names = 0, 0, []
        for name, frame in sorted(vehicle_entries(v, polys, exit_frame).items(), key=lambda kv: kv[1]):
            pos = ped_positions_at(peds, frame, gap)
            n = int(inside(polys[name], pos["x"], pos["y"]).sum())
            at += n
            if n:
                names.append(name)
            lost += count_lost(ends, frame / fps, polys[name], polys_m[name], cfg["merge_gap_s"], s["occluded_edge_m"])
        rows.append((speed_at(v, exit_frame, window), lost, at, ";".join(names)))

    out_df = pd.concat([ev, pd.DataFrame(rows, columns=NEW_COLUMNS[:4], index=ev.index)], axis=1)
    out_df = out_df.assign(signal_at_stopline="", stop_sec_before_line=np.nan).round({"entry_speed": 3})
    out = Path(out_dir) / f"{Path(events).stem}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"이벤트 {len(out_df)}건: 진입 순간 보행자 있음 {(out_df['peds_at_entry'] > 0).sum()}건, "
          f"가림 의심 후보 {((out_df['peds_at_entry'] == 0) & (out_df['peds_lost'] > 0)).sum()}건")
    print(f"저장: {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="이벤트 특징 계산")
    parser.add_argument("--events", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--site", required=True)
    parser.add_argument("--signal", default=None, help="2단계 빨간불 시간표. MVP(1단계)에서는 생략")
    parser.add_argument("--config", default="configs/detect.yaml", help="merge_gap_s·speed_window_s")
    args = parser.parse_args()
    build_features(args.events, args.tracks, args.site, args.signal, args.config)


if __name__ == "__main__":
    main()
