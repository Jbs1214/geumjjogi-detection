# 담당: 김민영
"""YOLO + ByteTrack으로 차량·오토바이·보행자를 추적하고 박스 하단 중심점을 미터 좌표로 바꾼다.
오토바이 박스와 driver_overlap_min 이상 겹치는 pedestrian은 운전자로 보고 뺀다.
짧은 트랙(min_track_s)은 지우고, 끊긴 같은 종류 트랙은 merge_gap_s·merge_dist_m 안이면 이어 붙인다.
시간이 겹쳐 ID가 바뀐 트랙도 merge_overlap_s·merge_overlap_iou 조건이면 잇는다(merge_overlap_classes만).
COCO 원래 클래스는 coco_cls 컬럼에 남긴다.

순서: 추적 → 운전자 빼기 → 미터 변환 → 이어 붙이기 → 짧은 트랙 지우기.
짧은 트랙을 이어 붙인 뒤에 지우는 이유: 가림으로 쪼개진 조각은 하나하나가 짧아서, 먼저 지우면 이어 붙일 수 없다.

입력: 원본 영상(--video), configs/site_<영상>.yaml, configs/detect.yaml
출력: outputs/tracks/<영상>.csv (컬럼은 docs/csv_schema.md 1절). frame은 0부터, t_s = frame / fps
"""
import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO

from tracking.homography import compute_homography, to_meters

COLUMNS = ["frame", "t_s", "track_id", "cls", "coco_cls", "conf", "x1", "y1", "x2", "y2", "bx_m", "by_m"]


def load_homography(site: dict) -> np.ndarray:
    """출구 횡단보도 모서리·실측치 → H. 실측치가 0이면 compute_homography가 에러를 내 추적 전에 멈춘다."""
    cw = site["crosswalk"]
    return compute_homography(cw["corners_px"], cw["width_m"], cw["length_m"])


def detect_and_track(video: str, cfg: dict) -> pd.DataFrame:
    """프레임마다 검출 + ByteTrack. class_map에 있는 COCO 클래스만 남긴다. ID가 없는 박스는 버린다.
    conf는 넘기지 않는다: ByteTrack은 낮은 신뢰도 박스(track_low_thresh)로 가려진 트랙을 잇는데,
    conf를 넘기면 그 박스가 추적 전에 버려진다. 새 트랙 시작 기준은 tracker 설정의 new_track_thresh."""
    cap = cv2.VideoCapture(video)
    fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not fps:
        raise SystemExit(f"영상을 읽을 수 없다: {video}")
    model = YOLO(cfg["model"])
    class_map = cfg["class_map"]
    class_ids = [i for i, name in model.names.items() if name in class_map]
    results = model.track(video, stream=True, persist=True, tracker=cfg["tracker"],
                          imgsz=cfg["imgsz"], classes=class_ids, verbose=False)
    rows = []
    for frame, r in enumerate(results):  # 영상을 처음부터 한 장씩 읽으므로 0부터 세는 프레임 번호와 같다
        b = r.boxes
        if b.id is not None:
            for tid, c, conf, (x1, y1, x2, y2) in zip(b.id.int().tolist(), b.cls.int().tolist(),
                                                     b.conf.tolist(), b.xyxy.tolist()):
                coco = model.names[c]
                rows.append((frame, frame / fps, tid, class_map[coco], coco, conf, x1, y1, x2, y2))
        if frame % int(fps * 60) == 0:
            print(f"  추적 {frame / fps / 60:.0f}분 / {total / fps / 60:.0f}분")
    return pd.DataFrame(rows, columns=COLUMNS[:10])


def remove_drivers(df: pd.DataFrame, min_overlap: float) -> pd.DataFrame:
    """같은 프레임에서 사람 박스 면적의 min_overlap 이상이 오토바이 박스와 겹치면 운전자로 보고 뺀다."""
    peds, motos = df[df["cls"] == "pedestrian"], df[df["cls"] == "motorcycle"]
    if peds.empty or motos.empty:
        return df
    p = peds.reset_index().merge(motos[["frame", "x1", "y1", "x2", "y2"]], on="frame", suffixes=("", "_m"))
    iw = (np.minimum(p["x2"], p["x2_m"]) - np.maximum(p["x1"], p["x1_m"])).clip(lower=0)
    ih = (np.minimum(p["y2"], p["y2_m"]) - np.maximum(p["y1"], p["y1_m"])).clip(lower=0)
    # 분모는 사람 박스. 오토바이 박스가 사람보다 커도(옆으로 긴 차체) 운전자는 거의 다 안에 들어간다
    ratio = iw * ih / ((p["x2"] - p["x1"]) * (p["y2"] - p["y1"]))
    return df.drop(index=p.loc[ratio >= min_overlap, "index"].unique())


def add_meters(df: pd.DataFrame, H: np.ndarray) -> pd.DataFrame:
    """박스 하단 중심점(지면에 닿는 점) → 미터 좌표."""
    m = to_meters(H, np.column_stack([(df["x1"] + df["x2"]) / 2, df["y2"]]))
    return df.assign(bx_m=m[:, 0], by_m=m[:, 1])


def mean_iou(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """같은 프레임끼리 맞춘 두 박스 표(x1, y1, x2, y2)의 평균 IoU."""
    iw = (np.minimum(a["x2"], b["x2"]) - np.maximum(a["x1"], b["x1"])).clip(lower=0)
    ih = (np.minimum(a["y2"], b["y2"]) - np.maximum(a["y1"], b["y1"])).clip(lower=0)
    inter = iw * ih
    union = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]) + (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) - inter
    return float((inter / union).mean())


def merge_tracks(df: pd.DataFrame, gap_s: float, dist_m: float, overlap_s: float = 0.0,
                 overlap_iou: float = 0.5, overlap_classes: tuple = ()) -> pd.DataFrame:
    """트랙 A가 끝나고 gap_s 안에 A 마지막 점에서 dist_m 안에서 시작한 같은 종류 트랙 B는 A로 본다.
    시작 순으로 보며, 후보가 여럿이면 가장 가까운 것. A←B←C처럼 여러 번 이어질 수 있다.
    overlap_classes의 클래스는 B가 A 끝나기 overlap_s 전부터 시작했어도(한 물체에 ID가 둘 붙었다가 하나가 끝남)
    겹친 프레임의 평균 IoU가 overlap_iou 이상이면 잇는다. IoU 조건은 줄 선 다른 차·옆 사람을 잇지 않으려는 것.
    겹친 프레임은 conf가 높은 행 하나만 남긴다."""
    df = df.sort_values(["track_id", "frame"])
    g = df.groupby("track_id")
    info = pd.DataFrame({
        "cls": g["cls"].agg(lambda s: s.mode().iloc[0]),  # 프레임마다 클래스가 흔들릴 수 있어 최빈값
        "start": g["t_s"].min(), "end": g["t_s"].max(),
        "fx": g["bx_m"].first(), "fy": g["by_m"].first(), "lx": g["bx_m"].last(), "ly": g["by_m"].last(),
    }).sort_values("start")

    def same_object(a_tid, b_tid) -> bool:
        a, b = (g.get_group(t).set_index("frame")[["x1", "y1", "x2", "y2"]] for t in (a_tid, b_tid))
        common = a.index.intersection(b.index)
        return len(common) > 0 and mean_iou(a.loc[common], b.loc[common]) >= overlap_iou

    rename, tails = {}, {}  # tails = 이어 붙일 수 있는 트랙의 끝 {대표 id: (cls, 끝 시각, 끝 x, 끝 y, 끝 조각 id)}
    for tid, b in info.iterrows():
        tails = {k: v for k, v in tails.items() if b["start"] - v[1] <= gap_s}  # 너무 오래전에 끝난 것은 버림
        best, best_d = None, dist_m
        for root, (cls, end, lx, ly, last) in tails.items():
            d = math.hypot(b["fx"] - lx, b["fy"] - ly)
            if cls != b["cls"] or d > best_d:
                continue
            overlapped = (cls in overlap_classes and end - overlap_s <= b["start"] <= end < b["end"]
                          and same_object(last, tid))
            if b["start"] > end or overlapped:
                best, best_d = root, d
        root = tid if best is None else best
        if best is not None:
            rename[tid] = best
        tails[root] = (b["cls"], b["end"], b["lx"], b["ly"], tid)
    df = df.assign(track_id=df["track_id"].map(lambda t: rename.get(t, t)))
    return df.sort_values("conf", ascending=False).drop_duplicates(["frame", "track_id"]).sort_values(["track_id", "frame"])


def unify_class(df: pd.DataFrame) -> pd.DataFrame:
    """트랙마다 cls를 최빈값 하나로 맞춘다. ByteTrack은 프레임마다 검출 클래스를 그대로 붙여서
    보행자 트랙에 vehicle 행이 한두 개 섞이고, 그 행이 횡단보도 위에 있으면 차가 보행자로 세어진다.
    coco_cls는 원래 값을 남긴다(events가 최빈값을 쓴다, docs/csv_schema.md)."""
    mode = df.groupby("track_id")["cls"].agg(lambda s: s.mode().iloc[0])
    return df.assign(cls=df["track_id"].map(mode))


def drop_short(df: pd.DataFrame, min_s: float) -> pd.DataFrame:
    """처음~끝 시간이 min_s보다 짧은 트랙(잠깐 나온 오검출 등)을 지운다."""
    t = df.groupby("track_id")["t_s"]
    return df[t.transform("max") - t.transform("min") >= min_s]


def track(video: str, site: str, config: str, out_dir: str = "outputs/tracks") -> Path:
    with open(config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(site, encoding="utf-8") as f:
        H = load_homography(yaml.safe_load(f))  # 실측치가 없으면 오래 걸리는 추적 전에 멈춘다
    df = detect_and_track(video, cfg)
    n_raw, n_rows = df["track_id"].nunique(), len(df)
    df = remove_drivers(df, cfg["driver_overlap_min"])
    n_drivers = n_rows - len(df)
    df = add_meters(df, H)
    df = merge_tracks(df, cfg["merge_gap_s"], cfg["merge_dist_m"], cfg["merge_overlap_s"],
                      cfg["merge_overlap_iou"], tuple(cfg["merge_overlap_classes"]))
    n_merged = df["track_id"].nunique()
    df = unify_class(drop_short(df, cfg["min_track_s"]))
    df = df.sort_values(["frame", "track_id"])[COLUMNS].round(
        {"t_s": 3, "conf": 3, "x1": 1, "y1": 1, "x2": 1, "y2": 1, "bx_m": 3, "by_m": 3})
    out = Path(out_dir) / f"{Path(video).stem}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"트랙 {n_raw}개 → 이어 붙인 뒤 {n_merged}개 → 짧은 것 지운 뒤 {df['track_id'].nunique()}개. "
          f"운전자로 뺀 행 {n_drivers}개")
    print(f"저장: {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="검출·추적")
    parser.add_argument("--video", required=True)
    parser.add_argument("--site", required=True)
    parser.add_argument("--config", default="configs/detect.yaml")
    args = parser.parse_args()
    track(args.video, args.site, args.config)


if __name__ == "__main__":
    main()
