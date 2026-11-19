# 담당: 김민영
"""궤적에서 우회전 이벤트를 ROI 규칙(우회전 진입로 ROI를 지난 트랙이 출구 횡단보도에 진입)으로 뽑는다.
출구 ROI는 쓰지 않는다(기술서 11.1).

트랙마다 진입로 ROI에 처음 들어온 뒤 출구 횡단보도 영역에 처음 들어온 순간이 t_enter_s다(트랙당 최대 1건).
점은 박스 하단 중심(픽셀). ROI가 픽셀 다각형이라 미터로 바꾸지 않고 픽셀에서 본다.
출구 영역은 crosswalk.area_px(보이는 블록 전체), 비어 있으면 corners_px.
t_stopline_s(2단계)는 출구 진입 전 경로에서 하단 중심점이 stop_line_px를 넘은 시각이다(기술서 2.3). 못 찾으면 비운다.
끊긴 트랙을 이어 본 이벤트(link_broken)는 link_track_id에 앞 트랙 id를 남긴다(나머지는 비움).
한 차에 박스가 두 개 잡혀 두 번 나온 차량 이벤트는 바깥 박스 것만 남긴다(drop_contained, 기술서 8).
카메라가 움직인 구간(camera_moved_s)은 여기서 거르지 않는다(rules.rule이 제외로 판정한다).

입력: outputs/tracks/<영상>.csv, configs/site_<영상>.yaml
출력: outputs/events/<영상>.csv (컬럼은 docs/csv_schema.md 2절)
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

COLUMNS = ["video", "event_id", "track_id", "cls", "coco_cls", "t_enter_s", "t_stopline_s", "link_track_id"]
VEHICLES = ("vehicle", "motorcycle")


def inside(poly: list, x: pd.Series, y: pd.Series) -> np.ndarray:
    """점마다 다각형 안(경계 포함)인지."""
    p = np.array(poly, np.float32)
    return np.array([cv2.pointPolygonTest(p, (float(a), float(b)), False) >= 0 for a, b in zip(x, y)], dtype=bool)


def exit_area(site: dict) -> list:
    """출구 판정 영역. area_px가 비어 있으면 homography 사각형(corners_px)."""
    cw = site["crosswalk"]
    return cw.get("area_px") or cw["corners_px"]


def link_broken(v: pd.DataFrame, in_a: np.ndarray, in_e: np.ndarray, events: pd.Index,
                gap_s: float, dist_box: float) -> pd.DataFrame:
    """진입로를 지났지만 출구에 안 간 트랙 A와, 진입로 없이 출구에 들어간 트랙 B가 끊긴 같은 차면 B를 우회전으로 본다.
    B는 A가 끝난 뒤 gap_s 안에, A 마지막 박스 폭 × dist_box 픽셀 안에서 시작해야 한다(같은 cls, 1:1, 가까운 쌍부터).
    정지선 근처는 출구 사각형에서 멀어 미터 거리가 부풀므로 픽셀로 본다.
    돌려주는 값: B track_id 인덱스, t_enter_s(출구 첫 진입 t_s)·link_track_id(A). features가 A 경로의 횡단보도도 본다."""
    g = v.sort_values("frame").groupby("track_id")
    first, last = g.first(), g.last()
    passed_a = set(v.loc[in_a, "track_id"])
    exit_t = v[in_e].groupby("track_id")["t_s"].min()
    a_ids = [t for t in passed_a if t not in exit_t.index]  # 진입로만 지나고 끊긴 트랙
    b_ids = [t for t in exit_t.index if t not in passed_a and t not in events]
    pairs = []
    for b in b_ids:
        fb = first.loc[b]
        bx, by = (fb["x1"] + fb["x2"]) / 2, fb["y2"]
        for a in a_ids:
            la = last.loc[a]
            if la["cls"] != fb["cls"] or not la["t_s"] <= fb["t_s"] <= la["t_s"] + gap_s:
                continue
            d = np.hypot((la["x1"] + la["x2"]) / 2 - bx, la["y2"] - by)
            if d <= dist_box * (la["x2"] - la["x1"]):
                pairs.append((d, a, b))
    used_a, linked = set(), {}
    for d, a, b in sorted(pairs):
        if a not in used_a and b not in linked:
            used_a.add(a)
            linked[b] = (exit_t[b], a)
    return pd.DataFrame(list(linked.values()), columns=["t_enter_s", "link_track_id"],
                        index=pd.Index(list(linked), name="track_id"))


def find_events(df: pd.DataFrame, approach: list, exit_poly: list,
                link_gap_s: float = 0.0, link_dist_box: float = 0.0) -> pd.DataFrame:
    """차량·오토바이 트랙마다 진입로 ROI 첫 프레임 이후(같은 프레임 포함) 출구 영역 첫 프레임을 찾는다.
    출구를 먼저 지나고 진입로에 들어간 트랙(역방향)은 이벤트가 아니다. link_gap_s > 0이면 끊긴 트랙도 잇는다(link_broken).
    t_enter_s 순으로 돌려준다."""
    v = df[df["cls"].isin(VEHICLES)]
    x, y = (v["x1"] + v["x2"]) / 2, v["y2"]
    in_a, in_e = inside(approach, x, y), inside(exit_poly, x, y)
    first_a = v[in_a].groupby("track_id")["frame"].min()
    e = v[in_e & (v["frame"] >= v["track_id"].map(first_a))]  # 진입로를 안 지난 트랙은 NaN과 비교돼 빠진다
    out = e.groupby("track_id").agg(cls=("cls", "first"), t_enter_s=("t_s", "min"))
    if link_gap_s > 0:
        linked = link_broken(v, in_a, in_e, out.index, link_gap_s, link_dist_box)
        cls = v.groupby("track_id")["cls"].first()
        out = pd.concat([out, linked.assign(cls=linked.index.map(cls))])
    out["link_track_id"] = out.get("link_track_id", pd.Series(dtype="Int64")).astype("Int64")  # 이어 보지 않은 이벤트는 비움
    mode = v.groupby("track_id")["coco_cls"].agg(lambda s: s.mode().iloc[0])  # 트랙 전체 최빈값
    out["coco_cls"] = out.index.map(mode)  # Series를 바로 넣으면 out이 비었을 때 mode의 행이 새로 생긴다
    return out.reset_index().sort_values(["t_enter_s", "track_id"], ignore_index=True)


def drop_contained(df: pd.DataFrame, ev: pd.DataFrame, gap_s: float, contain_min: float, frames_min: float) -> pd.DataFrame:
    """한 차에 박스가 두 개 잡혀 두 번 나온 이벤트를 거른다(기술서 8). 차량(vehicle)끼리만, t_enter_s 차이 gap_s 이하인 쌍만 본다.
    두 트랙이 함께 있는 프레임 중 '겹친 넓이 ÷ 작은 박스 넓이'가 contain_min 이상인 프레임의 비율이 frames_min 이상이면 같은 차다.
    평균 박스가 큰 쪽(바깥 박스) 이벤트를 남기고, coco_cls는 평균 conf가 높은 트랙 값으로 바꾼다. frames_min이 0이면 끈다."""
    if frames_min <= 0:
        return ev
    cand = list(ev[ev["cls"] == "vehicle"].sort_values("t_enter_s").itertuples())
    boxes = df[df["track_id"].isin([c.track_id for c in cand])]
    boxes = boxes.assign(area=(boxes["x2"] - boxes["x1"]) * (boxes["y2"] - boxes["y1"]))
    tracks = {t: g.set_index("frame") for t, g in boxes.groupby("track_id")}
    ev, dropped = ev.copy(), set()
    for i, a in enumerate(cand):
        for b in cand[i + 1:]:
            if b.t_enter_s - a.t_enter_s > gap_s:
                break
            if a.track_id in dropped or b.track_id in dropped:
                continue
            ta, tb = tracks[a.track_id], tracks[b.track_id]
            j = ta.merge(tb, left_index=True, right_index=True, suffixes=("_a", "_b"))
            if j.empty:
                continue
            iw = (np.minimum(j["x2_a"], j["x2_b"]) - np.maximum(j["x1_a"], j["x1_b"])).clip(lower=0)
            ih = (np.minimum(j["y2_a"], j["y2_b"]) - np.maximum(j["y1_a"], j["y1_b"])).clip(lower=0)
            if (iw * ih / np.minimum(j["area_a"], j["area_b"]) >= contain_min).mean() < frames_min:
                continue
            outer, inner = (a, b) if ta["area"].mean() >= tb["area"].mean() else (b, a)
            best = a if ta["conf"].mean() >= tb["conf"].mean() else b
            ev.loc[ev["track_id"] == outer.track_id, "coco_cls"] = best.coco_cls
            dropped.add(inner.track_id)
    return ev[~ev["track_id"].isin(dropped)].reset_index(drop=True)


def stopline_time(v: pd.DataFrame, line: list, after: tuple) -> float:
    """트랙(프레임 순)의 박스 하단 중심점이 정지선을 넘은 시각. after는 정지선을 지난 쪽의 점(출구 횡단보도).
    정지선은 두 끝점을 지나는 직선으로 본다(우회전 차는 찍어 둔 끝점 바깥을 지날 수 있다).
    정지선 위에 멈춘 차는 점이 떨려 여러 번 넘나들므로 마지막으로 넘은 때의 첫 프레임을 쓴다.
    선 앞에서 뒤로 넘어간 적이 없으면(선을 지난 뒤에야 잡힌 트랙) nan."""
    (ax, ay), (bx, by) = line

    def side(x, y):
        return (bx - ax) * (y - ay) - (by - ay) * (x - ax)

    s = side((v["x1"] + v["x2"]) / 2, v["y2"]).to_numpy() * np.sign(side(*after))
    cross = np.flatnonzero((s[:-1] < 0) & (s[1:] >= 0))
    return float(v["t_s"].iloc[cross[-1] + 1]) if len(cross) else np.nan


def stopline_times(df: pd.DataFrame, ev: pd.DataFrame, line: list, after: tuple) -> list:
    """이벤트마다 출구 진입(t_enter_s)까지의 경로에서 정지선 통과 시각을 찾는다. 이어 본 이벤트는 앞 트랙 경로까지 합친다."""
    ids = set(ev["track_id"]) | set(ev["link_track_id"].dropna())
    tracks = dict(tuple(df[df["track_id"].isin(ids)].sort_values("frame").groupby("track_id")))
    out = []
    for e in ev.itertuples():
        v = tracks[e.track_id] if pd.isna(e.link_track_id) else pd.concat([tracks[e.link_track_id], tracks[e.track_id]])
        out.append(stopline_time(v[v["t_s"] <= e.t_enter_s], line, after))
    return out


def extract_events(tracks: str, site: str, config: str = "configs/detect.yaml", out_dir: str = "outputs/events") -> Path:
    with open(site, encoding="utf-8") as f:
        s = yaml.safe_load(f)
    with open(config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    df = pd.read_csv(tracks)
    found = find_events(df, s["approach_roi_px"], exit_area(s), cfg["link_gap_s"], cfg["link_dist_box"])
    ev = drop_contained(df, found, cfg["dedup_gap_s"], cfg["dedup_contain_min"], cfg["dedup_frames_min"])
    line = s.get("stop_line_px")
    stop = stopline_times(df, ev, line, np.mean(exit_area(s), axis=0)) if line and np.any(line) else np.nan  # 정지선을 안 찍었으면 비움
    ev = ev.assign(video=Path(tracks).stem, event_id=range(1, len(ev) + 1), t_stopline_s=stop)[COLUMNS]
    out = Path(out_dir) / f"{Path(tracks).stem}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    ev.round({"t_enter_s": 3, "t_stopline_s": 3}).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"우회전 이벤트 {len(ev)}건 {ev['cls'].value_counts().to_dict()}, 정지선 통과 시각 없음 {int(ev['t_stopline_s'].isna().sum())}건, "
          f"중복 제거 {len(found) - len(ev)}건")
    print(f"저장: {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="우회전 이벤트 추출")
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--site", required=True)
    parser.add_argument("--config", default="configs/detect.yaml")
    args = parser.parse_args()
    extract_events(args.tracks, args.site, args.config)


if __name__ == "__main__":
    main()
