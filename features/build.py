# 담당: 김민영
"""이벤트마다 속도·거리·횡단보도 위 보행자 특징을 계산한다.
1단계는 경로상 각 횡단보도(crosswalk = exit, other_crosswalks_px)에 차량이 진입하는 순간만 본다:
그 위 보행자 수의 합(peds_at_entry), 보행자가 있던 횡단보도 이름(no_yield_at),
진입 직전 merge_gap_s 안에 그 횡단보도 경계에서 사람 박스 높이 × occluded_edge_box(픽셀)보다 안쪽에서 끊긴 보행자 트랙 수의 합
(peds_lost, 가림 의심. 박스 높이를 자로 써서 실측값이 필요 없다, 기술서 11.1 좌표 기준).

진입 순간: 출구는 이벤트의 t_enter_s, 다른 횡단보도는 그 차량 트랙이 출구 진입 전에 처음 들어간 프레임.
끊긴 트랙을 이어 본 이벤트(events의 link_track_id)는 앞 트랙 경로까지 합쳐서 본다(경로상 횡단보도 전부, 기술서 11.1).
차가 들어가지 않은 횡단보도는 세지 않는다. 점은 박스 하단 중심(픽셀), 출구 영역은 area_px(없으면 corners_px).
보행자는 그 횡단보도의 ped_extra_px(줄무늬~연석 사이 차도)에 있어도 센다(차 진입 시각에는 쓰지 않음).
보행자 박스는 신뢰도 ped_conf_min 이상만 쓴다(가려져 위치가 틀어진 저신뢰 박스가 경계 밖 사람을 안으로 넣지 않게, 기술서 11.20).
그 프레임에 검출이 빠진 보행자 트랙은 앞뒤 행 간격이 merge_gap_s 이하이면 직선 보간한 위치로 센다
(검출이 잠깐 빠진 순간 서 있던 보행자가 0명으로 세어지지 않게. 트랙 이어 붙이기와 같은 기준).
entry_speed는 출구 진입 순간 속도(미터 좌표 speed_window_s 이동평균 후 차분, 참고값). 실측값이 없어 미터 좌표가 비면 비운다.
2단계(--signal을 줄 때만 채운다, 기술서 2.3): signal_at_stopline은 정지선 통과 시각(t_stopline_s)의 빨간불 시간표 상태,
stop_level은 차 앞 끝(박스 아래변 세 점 중 정지선 너머로 가장 나간 점)이 정지 구간(stop_zone_px) 안에서 멈춘 위치
(strict 엄격 구간 / minor 보통 구간에서만 / none 정지 없음). 판정할 수 없으면 비우고 이유를 stop_note에 적는다.
정지 = 앞 끝이 stop_min_duration_s 동안 max(stop_move_min_px, 박스 높이 × stop_move_ratio) 픽셀 이하로 움직임(속도는 쓰지 않음).
엄격 구간 = 앞 끝이 정지선(stop_line_px)을 박스 높이 × stop_strict_margin_ratio 이하로 넘은 곳. 이어 본 이벤트는 앞 트랙 경로 포함.

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

NEW_COLUMNS = ["entry_speed", "peds_lost", "peds_at_entry", "no_yield_at", "signal_at_stopline", "stop_level", "stop_note"]


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


def count_lost(ends: pd.DataFrame, t_entry: float, poly_px: list, gap_s: float, edge_box: float) -> int:
    """진입 직전 gap_s 안에 끝난 보행자 트랙 중, 마지막 점이 횡단보도 안이고 경계에서 그 박스 높이 × edge_box(픽셀)보다
    안쪽인 것의 수(기술서 11.1). 걸어 나간 사람은 경계 근처에서 끝나고, 안쪽에서 사라졌다면 진입 순간 가려져 있었을 수 있다."""
    e = ends[(ends["t_s"] >= t_entry - gap_s) & (ends["t_s"] < t_entry)]
    poly = np.array(poly_px, np.float32)
    return int(sum(cv2.pointPolygonTest(poly, (float(x), float(y)), True) > (y - y1) * edge_box
                   for x, y, y1 in zip(e["cx"], e["y2"], e["y1"])))


def speeds(v: pd.DataFrame, window: int) -> pd.Series:
    """트랙(프레임 순)의 행마다 직전 행부터의 속도(m/s). 미터 좌표를 window 프레임 중심 이동평균한 뒤 차분한다."""
    sx = v["bx_m"].rolling(window, center=True, min_periods=1).mean()
    sy = v["by_m"].rolling(window, center=True, min_periods=1).mean()
    return np.hypot(sx.diff(), sy.diff()) / v["t_s"].diff()


def speed_at(v: pd.DataFrame, frame: int, window: int) -> float:
    """트랙 한 개(프레임 순)의 frame 순간 속도(m/s)."""
    speed = speeds(v, window)[v["frame"] == frame]
    return float(speed.iloc[0]) if len(speed) else np.nan


def signal_at(seg: pd.DataFrame, t: float) -> str:
    """빨간불 시간표(outputs/signal)에서 t가 들어 있는 구간의 상태. 구간은 시작 포함·끝 제외. t가 없거나 표 밖이면 빈 값."""
    hit = seg.loc[(seg["t_start_s"] <= t) & (t < seg["t_end_s"]), "state"]
    return hit.iloc[0] if len(hit) else ""


def front_points(path: pd.DataFrame, line_px: list, exit_pt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """행마다 차 앞 끝 (x, y)와 그 점이 정지선을 넘은 픽셀 거리(출구 쪽이 +). 앞 끝 = 박스 아래변의
    왼쪽·가운데·오른쪽 점 중 정지선 너머(출구 쪽)로 가장 나간 점(기술서 2.3. 정지선에 가장 가까운 점으로 고르면
    선을 넘은 뒤 뒤쪽 점이 골라진다). 정지선은 양 끝을 지나는 직선으로 본다."""
    (ax, ay), (bx, by) = np.asarray(line_px, dtype=float)
    norm = np.hypot(bx - ax, by - ay)

    def signed(x, y):
        return ((bx - ax) * (y - ay) - (by - ay) * (x - ax)) / norm

    sign = 1.0 if signed(*exit_pt) >= 0 else -1.0
    y = path["y2"].to_numpy(dtype=float)
    xs = np.stack([path["x1"].to_numpy(dtype=float), ((path["x1"] + path["x2"]) / 2).to_numpy(dtype=float),
                   path["x2"].to_numpy(dtype=float)], axis=1)
    d = sign * signed(xs, y[:, None])
    pick = d.argmax(axis=1)
    rows = np.arange(len(path))
    return np.stack([xs[rows, pick], y], axis=1), d[rows, pick]


def stop_level_in_zone(path: pd.DataFrame, t_stopline: float, line_px: list, zone_px: list, exit_pt: np.ndarray,
                       min_dur: float, move_min_px: float, move_ratio: float, margin_ratio: float) -> tuple[str, str]:
    """(stop_level, stop_note). 앞 끝이 정지 구간 안에서 min_dur 동안 max(move_min_px, 박스 높이 × move_ratio) 이하로
    움직였으면 정지(창의 처음·끝 모두 구간 안, 이동 = 창 시작점에서 가장 멀어진 거리, 박스 높이 = 창 안 중앙값).
    정지 창 중 하나라도 앞 끝이 정지선을 박스 높이 × margin_ratio 이하로만 넘었으면 strict, 아니면 minor, 정지가 없으면 none.
    판정할 수 없으면 ("", 사유): stopline_hidden = 통과 시각이 없음, 구간 안에 앞 끝이 잡힌 행이 없음,
    구간에서 검출이 min_dur보다 길게 빠지고 보이는 정지가 없음(그 사이 멈췄는지 모름).
    zone_unseen = 앞 끝이 처음 잡힐 때 이미 구간 안이고 정지가 없음(구간에 들어온 뒤를 다 보지 못함)."""
    if np.isnan(t_stopline):
        return "", "stopline_hidden"
    path = path.sort_values("t_s")
    front, over = front_points(path, line_px, exit_pt)
    zone = inside(zone_px, front[:, 0], front[:, 1])
    if not zone.any():
        return "", "stopline_hidden"
    t = path["t_s"].to_numpy(dtype=float)
    height = (path["y2"] - path["y1"]).to_numpy(dtype=float)
    level = "none"
    for i in np.flatnonzero(zone):
        j = int(np.searchsorted(t, t[i] + min_dur - 1e-9))   # 창 끝 = 시작에서 min_dur 뒤 첫 행
        if j >= len(t) or not zone[j]:
            continue
        h = float(np.median(height[i:j + 1]))
        moved = float(np.hypot(*(front[i:j + 1] - front[i]).T).max())
        if moved > max(move_min_px, h * move_ratio):
            continue
        if over[i:j + 1].max() <= h * margin_ratio:
            return "strict", ""
        level = "minor"
    if level != "none":
        return level, ""
    if zone[0]:
        return "", "zone_unseen"
    gap = np.diff(t) > min_dur
    if (gap & (zone[:-1] | zone[1:])).any():
        return "", "stopline_hidden"
    return "none", ""


def build_features(events: str, tracks: str, site: str, signal: str | None = None,
                   config: str = "configs/detect.yaml", out_dir: str = "outputs/features") -> Path:
    with open(config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(site, encoding="utf-8") as f:
        s = yaml.safe_load(f)
    ev = pd.read_csv(events, encoding="utf-8-sig", dtype={"link_track_id": "Int64"})  # 빈칸 때문에 float(26069.0)로 저장되지 않게
    tr = pd.read_csv(tracks, encoding="utf-8-sig")
    tr = tr.assign(cx=(tr["x1"] + tr["x2"]) / 2).sort_values(["track_id", "frame"])
    fps = (tr["frame"] / tr["t_s"])[tr["t_s"] > 0].median()
    gap = int(round(cfg["merge_gap_s"] * fps))
    window = max(1, int(round(cfg["speed_window_s"] * fps)))

    polys = crosswalks(s)
    extra = s.get("ped_extra_px") or {}  # 보행자만 세는 줄무늬~연석 사이 차도 {횡단보도 이름: 다각형}
    peds = tr[(tr["cls"] == "pedestrian") & (tr["conf"] >= cfg["ped_conf_min"])]
    ends = peds.groupby("track_id").tail(1)
    links = ev["link_track_id"] if "link_track_id" in ev else pd.Series(np.nan, index=ev.index)  # 컬럼 추가 전 events도 읽음
    vehicles = dict(tuple(tr[tr["track_id"].isin(ev["track_id"]) | tr["track_id"].isin(links.dropna())].groupby("track_id")))
    stage2 = bool(signal) and bool(s.get("stop_zone_px")) and bool(s.get("stop_line_px"))  # 정지 구간이 없는 영상은 정지 판정을 비움
    if signal:
        seg = pd.read_csv(signal, encoding="utf-8-sig")
    if stage2:
        exit_pt = np.array(polys["exit"], dtype=float).mean(axis=0)  # 정지선의 출구 쪽을 가리는 데만 씀

    rows = []
    for e, link in zip(ev.itertuples(), links):
        v = vehicles[e.track_id]
        exit_frame = int(v["frame"].iloc[(v["t_s"] - e.t_enter_s).abs().argmin()])
        path = v if pd.isna(link) else pd.concat([vehicles[int(link)], v])  # 속도는 이벤트 트랙만으로
        lost, at, names = 0, 0, []
        for name, frame in sorted(vehicle_entries(path, polys, exit_frame).items(), key=lambda kv: kv[1]):
            pos = ped_positions_at(peds, frame, gap)
            hit = inside(polys[name], pos["x"], pos["y"])
            if name in extra:  # 줄무늬~연석 사이 차도도 그 횡단보도로 센다. 겹친 자리의 사람도 한 번만
                hit |= inside(extra[name], pos["x"], pos["y"])
            n = int(hit.sum())
            at += n
            if n:
                names.append(name)
            lost += count_lost(ends, frame / fps, polys[name], cfg["merge_gap_s"], s["occluded_edge_box"])
        sig, level, note = "", "", ""
        if signal:
            sig = signal_at(seg, e.t_stopline_s)
        if stage2:
            level, note = stop_level_in_zone(path, e.t_stopline_s, s["stop_line_px"], s["stop_zone_px"], exit_pt,
                                             s["stop_min_duration_s"], s["stop_move_min_px"], s["stop_move_ratio"],
                                             s["stop_strict_margin_ratio"])
        rows.append((speed_at(v, exit_frame, window), lost, at, ";".join(names), sig, level, note))

    out_df = pd.concat([ev, pd.DataFrame(rows, columns=NEW_COLUMNS, index=ev.index)], axis=1)
    out_df = out_df.round({"entry_speed": 3})
    out = Path(out_dir) / f"{Path(events).stem}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"이벤트 {len(out_df)}건: 진입 순간 보행자 있음 {(out_df['peds_at_entry'] > 0).sum()}건, "
          f"가림 의심 후보 {((out_df['peds_at_entry'] == 0) & (out_df['peds_lost'] > 0)).sum()}건")
    if signal:
        red = out_df["signal_at_stopline"] == "red"
        print(f"정지선 통과 시각의 신호 {out_df['signal_at_stopline'].replace('', '없음').value_counts().to_dict()}, "
              f"빨간불 중 정지 위치 {out_df.loc[red, 'stop_level'].replace('', '못 정함').value_counts().to_dict()}, "
              f"못 정한 이유 {out_df.loc[red & (out_df['stop_note'] != ''), 'stop_note'].value_counts().to_dict()}")
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
