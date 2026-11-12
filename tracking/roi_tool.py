# 담당: 김민영
"""영상 한 장면에서 출구 횡단보도 네 모서리, 출구 판정 영역(선택), 경로가 지나는 다른 횡단보도(other_crosswalks_px),
우회전 진입로 ROI, 정지선, 차량 신호등을 클릭해 교차로 설정 파일을 만든다.
다른 횡단보도는 화면에 보이는 부분만 다각형으로 찍는다(docs/judgment_criteria.md).
출구 네 모서리는 실측 크기를 아는 직사각형(homography 기준)이라, 연석이 둥글어 사각형 밖으로 나온 블록은
판정 영역(crosswalk.area_px)으로 따로 감싼다. 사각형을 늘리면 인도까지 들어가 기다리는 사람이 보행자로 세어진다.

입력: 원본 영상(--video), 횡단보도 실측 치수(m, 나중에 채워도 된다)
출력: configs/site_<영상>.yaml (형식은 configs/site_example.yaml. camera_moved_s·right_turn_signal은 영상을 보고 직접 고친다)

조작: 왼쪽 클릭 = 점 추가, 오른쪽 클릭 = 마지막 점 취소, Enter = 다음 단계,
      다른 횡단보도: 3점 이상 Enter = 하나 끝(터미널에 이름), 점 없이 Enter = 다음 단계,
      출구 판정 영역: 3점 이상 Enter, s = 건너뛰기(판정 영역·다른 횡단보도·정지선·신호등만), Esc = 저장하지 않고 종료,
      휠 = 커서 위치 확대·축소, 가운데 버튼 끌기 = 확대한 화면 이동,
      n / b = 5초 뒤 / 앞 장면 (차가 가리면 넘긴다. 찍은 점은 그대로)
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from tracking.homography import compute_homography

EXAMPLE = Path(__file__).resolve().parent.parent / "configs" / "site_example.yaml"
MAX_VIEW_W = 1600  # 원본(1920·4K)이 화면을 넘지 않게 줄여 보여 준다
MAX_ZOOM = 8.0
SEEK_SEC = 5.0
POLY = "3+"  # 점 개수 자리: 3점 이상 다각형 하나

# (키, 점 개수(정수 = 정확히 그 개수, POLY = 3점 이상 다각형 하나, None = 3점 이상 다각형 여러 개),
#  건너뛰기 가능, 화면 이름, 화면 힌트, 터미널 안내).
# 화면은 OpenCV가 한글을 못 그려 짧은 영어로 쓰고, 한글 안내는 터미널과 tracking/README.md에 둔다.
STEPS = [
    ("corners_px", 4, False, "EXIT CROSSWALK", "1 top-left > 2 top-right > 3 bottom-right > 4 bottom-left",
     "출구 횡단보도(꺾은 뒤 마지막으로 지나는 곳) 네 모서리: 좌상 → 우상 → 우하 → 좌하 (화면 기준 시계 방향). "
     "양쪽 블록이 모두 있는 행만 감싸는 직사각형"),
    ("area_px", POLY, True, "EXIT AREA (optional)", "all visible blocks incl. curb-side ones, 3+ points. Same as corners: s",
     "출구 판정 영역(선택): 사각형 밖으로 나온 블록(둥근 연석 쪽 등)까지 보이는 블록 전체 테두리 3점 이상 → Enter. "
     "사각형과 같으면 s"),
    ("other_crosswalks_px", None, True, "OTHER CROSSWALKS on the path",
     "stripes only (no sidewalk), 3+ points > Enter > name in terminal. Done: Enter (0 points)",
     "출구 말고 우회전 차가 지나는 횡단보도(진입 쪽·대각선 등). 보이는 부분의 블록(줄무늬)만 감싸서(인도 제외) 3점 이상 → Enter → 터미널에 이름. "
     "다 찍었으면 점 없이 Enter, 하나도 없으면 s"),
    ("approach_roi_px", 4, False, "APPROACH (lane past stop line to turn)", "4 points around the right-turn path",
     "우회전 진입로 영역 4점 (우회전 차로부터 정지선을 지나 꺾기 시작하는 곳까지, 우회전 경로만)"),
    ("stop_line_px", 2, True, "STOP LINE", "2 ends. No line: front edge of the entry crosswalk",
     "정지선 양 끝 2점 (2단계). 정지선이 없으면 진입 쪽 횡단보도 앞 가장자리 두 점. 2단계를 안 하면 s"),
    ("signal_roi_px", 4, True, "TRAFFIC LIGHT (car, not pedestrian)", "4 points around the light",
     "우회전 방향 차량 신호등 4점 (2단계, 보행자 신호등 아님, s로 건너뛰기)"),
]
COLORS = {"corners_px": (0, 255, 255), "area_px": (0, 140, 255), "other_crosswalks_px": (255, 200, 0), "approach_roi_px": (0, 255, 0),
          "stop_line_px": (0, 0, 255), "signal_roi_px": (255, 0, 255)}
# --only 에 쓰는 짧은 이름
ONLY = {"crosswalk": "corners_px", "exit_area": "area_px", "others": "other_crosswalks_px", "approach": "approach_roi_px",
        "stop_line": "stop_line_px", "signal": "signal_roi_px"}
CROSSWALK_KEYS = ("corners_px", "area_px")  # yaml에서 crosswalk 아래에 들어가는 키


class Video:
    """원하는 시각의 장면을 읽는다. 창을 연 동안 영상을 열어 둔다."""

    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise SystemExit(f"영상을 읽을 수 없다: {path}")
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.duration = max(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps - 1.0 / fps, 0.0)

    def frame_at(self, sec: float) -> tuple[np.ndarray, float]:
        sec = min(max(sec, 0.0), self.duration)
        self.cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ok, frame = self.cap.read()
        if not ok:
            raise SystemExit(f"{sec:.0f}초 장면을 읽을 수 없다")
        return frame, sec

    def release(self) -> None:
        self.cap.release()


def draw(view: np.ndarray, k: float, x0: float, y0: float, shown: dict, current: list,
         steps: list, step: int, zoom_level: float = 1.0, t_sec: float = 0.0) -> np.ndarray:
    """view = 원본을 (x0, y0)부터 k배로 보여 주는 화면. 점은 원본 좌표 → 화면 좌표로 바꿔 그린다.
    shown = 이미 정한 영역(이번 실행에서 찍은 것 + --only 일 때 기존 값)."""
    img = view.copy()
    cur_key = steps[step][0] if step < len(steps) else None
    layers = []  # (키, 점들, 닫힌 도형인가, 이름)
    for key, val in shown.items():
        if isinstance(val, dict):  # other_crosswalks_px = {이름: 점들}
            layers += [(key, pts, True, name) for name, pts in val.items()]
        elif key != cur_key:
            layers.append((key, val, len(val) > 2, None))
    if cur_key:
        layers.append((cur_key, current, len(current) == steps[step][1], None))
    for key, pts, closed, label in layers:
        if not pts:
            continue
        color = COLORS[key]
        p = ((np.array(pts) - [x0, y0]) * k).astype(np.int32)
        cv2.polylines(img, [p], closed, color, 2)
        for n, (x, y) in enumerate(p, 1):
            cv2.circle(img, (int(x), int(y)), 5, color, -1)
            cv2.putText(img, str(n), (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if label:
            cx, cy = p.mean(axis=0).astype(int)
            cv2.putText(img, label, (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
            cv2.putText(img, label, (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    if cur_key:
        _key, n, skippable, name, hint, _ko = steps[step]
        if n is None:
            status = f"{len(current)} points, {len(shown.get(cur_key) or {})} done"
        elif n == POLY:
            status = f"{len(current)} points (3+)"
        else:
            status = "OK - press Enter" if len(current) == n else f"{len(current)}/{n} points"
        keys = "Enter: next | Right-click: undo | Esc: quit" + (" | s: skip" if skippable else "")
        nav = f"Wheel: zoom (x{zoom_level:.1f}) | Middle-drag: move | n/b: +/-{SEEK_SEC:.0f}s (t={t_sec:.0f}s)"
        lines = [(f"STEP {step + 1}/{len(steps)}  {name}  [{status}]", 0.8, COLORS[cur_key]),
                 (hint, 0.6, (255, 255, 255)), (keys, 0.6, (200, 200, 200)), (nav, 0.6, (200, 200, 200))]
        band = img[:12 + 28 * len(lines), :800]
        band[:] = (band * 0.4).astype(np.uint8)  # 밝은 도로에서도 글자가 보이게 어둡게 깐다
        for i, (text, size, color) in enumerate(lines):
            y = 32 + i * 28
            cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), 5)  # 검은 테두리
            cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 2)
    return img


def ask_name(polys: dict) -> str:
    """다른 횡단보도 이름을 터미널에서 받는다. 빈 값이면 cw1, cw2 …"""
    while True:
        default = f"cw{len(polys) + 1}"
        try:
            name = input(f"  이 횡단보도 이름 (터미널에 입력, 그냥 Enter = {default}): ").strip() or default
        except EOFError:  # 입력을 받을 수 없는 터미널(다른 프로그램이 실행)이면 기본 이름. yaml에서 고친다
            print(f"  입력 불가 → {default}")
            name = default
        if name not in polys and name != "exit":  # exit = 출구 횡단보도(판정 결과 이름)
            return name
        print(f"  이미 쓰는 이름: {name}")


def collect(video: Video, start_sec: float = 0.0, steps: list = STEPS, existing: dict | None = None) -> dict | None:
    """창에서 단계별로 점을 받는다. Esc면 None. existing = 화면에 같이 보여 줄 기존 영역."""
    frame, t_sec = video.frame_at(start_sec)
    h, w = frame.shape[:2]
    scale = min(1.0, MAX_VIEW_W / w)
    view_size = (round(w * scale), round(h * scale))
    done, current, polys, step = {}, [], {}, 0  # polys = 이번에 찍은 다른 횡단보도 {이름: 점들}
    # 화면 좌표 v ↔ 원본 좌표 o: o = (x0, y0) + v / k,  k = scale × zoom
    vs = {"zoom": 1.0, "x0": 0.0, "y0": 0.0, "drag": None, "frame": frame, "t": t_sec}

    def k() -> float:
        return scale * vs["zoom"]

    def clamp() -> None:
        vs["x0"] = min(max(vs["x0"], 0.0), w - w / vs["zoom"])
        vs["y0"] = min(max(vs["y0"], 0.0), h - h / vs["zoom"])

    def on_mouse(event, x, y, flags, _param):
        ox, oy = vs["x0"] + x / k(), vs["y0"] + y / k()  # 커서 아래의 원본 좌표
        n = steps[step][1]
        if event == cv2.EVENT_LBUTTONDOWN and (not isinstance(n, int) or len(current) < n):
            current.append([round(ox), round(oy)])  # 원본 해상도 좌표로 저장
        elif event == cv2.EVENT_RBUTTONDOWN and current:
            current.pop()
        elif event == cv2.EVENT_MOUSEWHEEL:
            # 휠 방향은 flags 위쪽 16비트(부호 있음). OpenCV 5 파이썬에는 getMouseWheelDelta가 없다.
            factor = 1.25 if (flags >> 16) > 0 else 1 / 1.25
            vs["zoom"] = min(max(vs["zoom"] * factor, 1.0), MAX_ZOOM)
            vs["x0"], vs["y0"] = ox - x / k(), oy - y / k()  # 커서 아래 점이 그 자리에 남게
            clamp()
        elif event == cv2.EVENT_MBUTTONDOWN:
            vs["drag"] = (x, y)
        elif event == cv2.EVENT_MBUTTONUP:
            vs["drag"] = None
        elif event == cv2.EVENT_MOUSEMOVE and vs["drag"]:
            dx, dy = x - vs["drag"][0], y - vs["drag"][1]
            vs["x0"], vs["y0"], vs["drag"] = vs["x0"] - dx / k(), vs["y0"] - dy / k(), (x, y)
            clamp()

    def render() -> np.ndarray:
        m = np.float32([[k(), 0, -k() * vs["x0"]], [0, k(), -k() * vs["y0"]]])
        interp = cv2.INTER_AREA if k() < 1 else cv2.INTER_LINEAR
        return cv2.warpAffine(vs["frame"], m, view_size, flags=interp)

    win = "roi_tool"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    print(f"[{step + 1}/{len(steps)}] {steps[step][-1]}")
    try:
        while step < len(steps):
            key, n, skippable, *_text = steps[step]
            shown = {**(existing or {}), **done}
            if n is None:
                shown[key] = polys  # 다시 찍을 때는 기존 다각형 대신 이번 것만 보여 준다
            cv2.imshow(win, draw(render(), k(), vs["x0"], vs["y0"], shown, current, steps, step,
                                 vs["zoom"], vs["t"]))
            pressed = cv2.waitKey(30) & 0xFF
            if pressed == 27:
                return None
            if pressed in (ord("n"), ord("b")):
                delta = SEEK_SEC if pressed == ord("n") else -SEEK_SEC
                vs["frame"], vs["t"] = video.frame_at(vs["t"] + delta)  # 카메라 고정이라 찍은 점은 그대로 둔다
                continue
            if pressed == 13 and n is None and len(current) >= 3:
                name = ask_name(polys)
                polys[name], current = list(current), []
                print(f"  {name}: {len(polys[name])}점. 다음 횡단보도를 찍거나, 다 찍었으면 점 없이 Enter")
                continue
            if pressed == 13 and (len(current) >= 3 if n == POLY else len(current) == (n or 0)):
                done[key], current = (polys if n is None else list(current)), []
            elif pressed == ord("s") and skippable:
                done[key], current = ({} if n is None else []), []
            else:
                continue
            step += 1
            if step < len(steps):
                print(f"[{step + 1}/{len(steps)}] {steps[step][-1]}")
    finally:
        cv2.destroyWindow(win)
    return done


def check_site(site: dict) -> None:
    """모서리 4점 확인. 실측치가 있으면 homography까지 만들어 본다. 없으면(0) 경고만."""
    cw = site["crosswalk"]
    if np.asarray(cw["corners_px"]).shape != (4, 2):
        raise ValueError(f"횡단보도 모서리는 4점이어야 한다: {cw['corners_px']}")
    if cw["width_m"] == 0 or cw["length_m"] == 0:
        print("경고: 실측치(width_m, length_m) 미입력. yaml에서 채우기 전에는 track을 돌릴 수 없다.")
    else:
        compute_homography(cw["corners_px"], cw["width_m"], cw["length_m"])  # 음수 등은 여기서 에러


def build_site(video: str, points: dict, width_m: float, length_m: float) -> dict:
    """site_example.yaml에 클릭한 점과 실측치만 채운다.
    camera_moved_s·right_turn_signal·판정 기준값은 example 값 그대로 둔다(사람이 영상을 보고 고친다)."""
    with open(EXAMPLE, encoding="utf-8") as f:
        site = yaml.safe_load(f)
    site["site"] = Path(video).stem.split("_")[0]  # A_20261107_0800 → A
    site["crosswalk"] = {"corners_px": points["corners_px"], "area_px": points["area_px"],
                         "width_m": width_m, "length_m": length_m}
    site.update({key: points[key] for key in ONLY.values() if key not in CROSSWALK_KEYS})
    check_site(site)  # 모서리·실측치가 잘못되면 저장 전에 에러
    return site


def update_site(site: dict, points: dict, width_m: float | None, length_m: float | None) -> dict:
    """--only: 다시 찍은 영역과 새로 준 실측치만 바꾼다. 나머지는 그대로."""
    for key, pts in points.items():
        if key in CROSSWALK_KEYS:
            site["crosswalk"][key] = pts
        else:
            site[key] = pts
    if width_m is not None:
        site["crosswalk"]["width_m"] = width_m
    if length_m is not None:
        site["crosswalk"]["length_m"] = length_m
    check_site(site)
    return site


def save_site(out: str, video: str, site: dict) -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# tracking.roi_tool 로 생성: {Path(video).name}\n")
        yaml.safe_dump(site, f, allow_unicode=True, sort_keys=False, default_flow_style=None)
    print(f"저장: {out}")


def pick_points(video: str, out: str, width_m: float | None = None, length_m: float | None = None,
                frame_sec: float = 0.0, only: list[str] | None = None) -> None:
    site = None
    if only:
        if not Path(out).exists():
            raise SystemExit(f"--only 는 이미 만든 파일을 고친다. 파일이 없다: {out}")
        site = yaml.safe_load(Path(out).read_text(encoding="utf-8"))
        keys = [ONLY[name] for name in only]
        steps = [s for s in STEPS if s[0] in keys]
        existing = {**{k: site["crosswalk"].get(k, []) for k in CROSSWALK_KEYS},
                    **{k: site.get(k, []) for k in ONLY.values() if k not in CROSSWALK_KEYS}}
    else:
        steps, existing = STEPS, None
    cap = Video(video)
    try:
        points = collect(cap, frame_sec, steps, existing)
    finally:
        cap.release()
    if points is None:
        print("취소: 저장하지 않았다.")
        return
    if site is not None:
        site = update_site(site, points, width_m, length_m)
    else:
        site = build_site(video, points, width_m or 0.0, length_m or 0.0)
    save_site(out, video, site)


def main() -> None:
    parser = argparse.ArgumentParser(description="ROI·횡단보도 모서리 지정")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True, help="예: configs/site_A_20261107_0800.yaml")
    parser.add_argument("--width-m", type=float, help="횡단보도 좌상→우상 변 실측 길이 (생략하면 0, 나중에 채운다)")
    parser.add_argument("--length-m", type=float, help="횡단보도 좌상→좌하 변 실측 길이 (생략하면 0, 나중에 채운다)")
    parser.add_argument("--frame-sec", type=float, default=0.0, help="처음 보여 줄 장면의 시각(초)")
    parser.add_argument("--only", nargs="+", choices=list(ONLY),
                        help="이미 만든 --out 파일에서 이 영역만 다시 찍는다")
    args = parser.parse_args()
    pick_points(args.video, args.out, args.width_m, args.length_m, args.frame_sec, args.only)


if __name__ == "__main__":
    main()
