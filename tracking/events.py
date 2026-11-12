# 담당: 김민영
"""궤적에서 우회전 이벤트를 ROI 규칙(우회전 진입로 ROI를 지난 트랙이 출구 횡단보도에 진입)으로 뽑는다.
출구 ROI는 쓰지 않는다(기술서 11.1).

트랙마다 진입로 ROI에 처음 들어온 뒤 출구 횡단보도 영역에 처음 들어온 순간이 t_enter_s다(트랙당 최대 1건).
점은 박스 하단 중심(픽셀). ROI가 픽셀 다각형이라 미터로 바꾸지 않고 픽셀에서 본다.
출구 영역은 crosswalk.area_px(보이는 블록 전체), 비어 있으면 corners_px.
t_stopline_s는 2단계 컬럼이라 지금은 비워 둔다(MVP 먼저).
끊긴 트랙을 이어 본 이벤트(link_broken)는 link_track_id에 앞 트랙 id를 남긴다(나머지는 비움).
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


def extract_events(tracks: str, site: str, config: str = "configs/detect.yaml", out_dir: str = "outputs/events") -> Path:
    with open(site, encoding="utf-8") as f:
        s = yaml.safe_load(f)
    with open(config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ev = find_events(pd.read_csv(tracks), s["approach_roi_px"], exit_area(s), cfg["link_gap_s"], cfg["link_dist_box"])
    ev = ev.assign(video=Path(tracks).stem, event_id=range(1, len(ev) + 1), t_stopline_s=np.nan)[COLUMNS]
    out = Path(out_dir) / f"{Path(tracks).stem}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    ev.round({"t_enter_s": 3}).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"우회전 이벤트 {len(ev)}건 {ev['cls'].value_counts().to_dict()}")
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
