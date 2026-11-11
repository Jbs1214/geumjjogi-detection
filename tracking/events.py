# 담당: 김민영
"""궤적에서 우회전 이벤트를 ROI 규칙(우회전 진입로 ROI를 지난 트랙이 출구 횡단보도에 진입)으로 뽑는다.
출구 ROI는 쓰지 않는다(기술서 11.1).

트랙마다 진입로 ROI에 처음 들어온 뒤 출구 횡단보도 영역에 처음 들어온 순간이 t_enter_s다(트랙당 최대 1건).
점은 박스 하단 중심(픽셀). ROI가 픽셀 다각형이라 미터로 바꾸지 않고 픽셀에서 본다.
출구 영역은 crosswalk.area_px(보이는 블록 전체), 비어 있으면 corners_px.
t_stopline_s는 2단계 컬럼이라 지금은 비워 둔다(MVP 먼저).
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

COLUMNS = ["video", "event_id", "track_id", "cls", "coco_cls", "t_enter_s", "t_stopline_s"]
VEHICLES = ("vehicle", "motorcycle")


def inside(poly: list, x: pd.Series, y: pd.Series) -> np.ndarray:
    """점마다 다각형 안(경계 포함)인지."""
    p = np.array(poly, np.float32)
    return np.array([cv2.pointPolygonTest(p, (float(a), float(b)), False) >= 0 for a, b in zip(x, y)], dtype=bool)


def exit_area(site: dict) -> list:
    """출구 판정 영역. area_px가 비어 있으면 homography 사각형(corners_px)."""
    cw = site["crosswalk"]
    return cw.get("area_px") or cw["corners_px"]


def find_events(df: pd.DataFrame, approach: list, exit_poly: list) -> pd.DataFrame:
    """차량·오토바이 트랙마다 진입로 ROI 첫 프레임 이후(같은 프레임 포함) 출구 영역 첫 프레임을 찾는다.
    출구를 먼저 지나고 진입로에 들어간 트랙(역방향)은 이벤트가 아니다. t_enter_s 순으로 돌려준다."""
    v = df[df["cls"].isin(VEHICLES)]
    x, y = (v["x1"] + v["x2"]) / 2, v["y2"]
    in_a, in_e = inside(approach, x, y), inside(exit_poly, x, y)
    first_a = v[in_a].groupby("track_id")["frame"].min()
    e = v[in_e & (v["frame"] >= v["track_id"].map(first_a))]  # 진입로를 안 지난 트랙은 NaN과 비교돼 빠진다
    out = e.groupby("track_id").agg(cls=("cls", "first"), t_enter_s=("t_s", "min"))
    mode = v.groupby("track_id")["coco_cls"].agg(lambda s: s.mode().iloc[0])  # 트랙 전체 최빈값
    out["coco_cls"] = out.index.map(mode)  # Series를 바로 넣으면 out이 비었을 때 mode의 행이 새로 생긴다
    return out.reset_index().sort_values(["t_enter_s", "track_id"], ignore_index=True)


def extract_events(tracks: str, site: str, out_dir: str = "outputs/events") -> Path:
    with open(site, encoding="utf-8") as f:
        s = yaml.safe_load(f)
    ev = find_events(pd.read_csv(tracks), s["approach_roi_px"], exit_area(s))
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
    args = parser.parse_args()
    extract_events(args.tracks, args.site)


if __name__ == "__main__":
    main()
