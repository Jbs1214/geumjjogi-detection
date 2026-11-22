# 담당: 김민영
"""영상 한 장면에서 횡단보도(출구·진입·중간) 네 모서리와 연석 선, 우회전 진입로 ROI, 정지선, 정지 구간,
차량 신호등을 클릭해 교차로 설정 파일을 만든다.
횡단보도 영역은 줄무늬 + 양쪽 줄무늬 끝~연석 선 사이 차도다(인도 제외, 기술서 2.2 11.22).
연석에 붙어 도는 차와 연석에서 내려와 선 보행자를 그 횡단보도로 잡기 위함이다. 줄무늬(네 꼭짓점)를 먼저 찍고
연석 선을 그리면, 각 횡단보도의 양옆 테두리를 연석 선까지 곧게 늘여 빈 차도를 채우고 저장할 때 그 횡단보도 다각형에 합친다.
횡단보도 영역끼리 겹친 자리(모퉁이 등)는 저장할 때 떼어 other_crosswalks_px.corner로 따로 둔다(기술서 11.1 "횡단보도 영역"). 화면에 보이는 부분만 찍는다(docs/judgment_criteria.md).
횡단보도 네 모서리는 블록 두 줄 각각의 첫 블록과 끝 블록 바깥 모서리다(두 줄 블록이 하나씩 엇갈려 있으면 엇갈린 그대로).
출구 네 모서리는 homography 기준이고,
판정은 출구 영역(crosswalk.area_px = 네 모서리 + 연석까지 채운 곳)으로 한다.
진입·중간 횡단보도는 other_crosswalks_px의 near·diag로 저장한다.

입력: 원본 영상(--video), 횡단보도 실측 치수(m, 나중에 채워도 된다)
출력: configs/site_<영상>.yaml (형식은 configs/site_example.yaml. camera_moved_s·right_turn_signal은 영상을 보고 직접 고친다)

화면은 튜토리얼처럼 단계(미션)를 하나씩 깬다: 시작 화면 → 미션마다 왼쪽 카드(할 일, 예시 그림, 다음에 찍을 점) →
마지막 확인 화면(Enter = 저장). 맨 위 막대에 진행 상황, 맨 아래에 키 안내, 커서 옆에 다음에 찍을 점이 나온다.

조작: 왼쪽 클릭 = 점 추가, 오른쪽 클릭 = 마지막 점 취소, r = 이 단계 점 모두 지우기, Enter = 다음 단계,
      s = 건너뛰기(출구 모서리·진입로 말고 모두), Backspace = 이전 단계로 돌아가 고치기,
      Esc = 저장하지 않고 종료, 휠 = 커서 위치 확대·축소, 가운데 버튼 끌기 = 확대한 화면 이동,
      n / b = 5초 뒤 / 앞 장면 (차가 가리면 넘긴다. 찍은 점은 그대로), h = 화면 안내 숨기기·보이기
"""
import argparse
import time
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from tracking.homography import compute_homography

EXAMPLE = Path(__file__).resolve().parent.parent / "configs" / "site_example.yaml"
MAX_VIEW_W = 1600  # 원본(1920·4K)은 화면을 넘지 않게 줄이고, 작은 영상은 이 크기까지 키워 보여 준다
MAX_VIEW_H = 900   # 세로 영상도 창 아래쪽이 화면(작업 표시줄 포함)을 넘지 않게. 가로 1920·4K는 1600x900이라 그대로
MAX_ZOOM = 8.0
SEEK_SEC = 5.0
POLY = "4+"  # 점 개수 자리: 4점 이상 다각형 하나(네 모서리 + 둥근 연석 등을 따라 더)
LINES = "선"  # 점 개수 자리: 2점 이상 꺾은선 여러 개(연석 선)
TOAST_SEC = 1.2  # "미션 완료" 알림을 보여 주는 시간(초)
OVERLAP_PX = 3   # 횡단보도 영역끼리 이만큼(원본 픽셀)보다 두껍게 겹친 자리만 corner로 뗀다(경계가 맞닿은 오차는 봐줌)
JOIN_PX = 3      # 빈 차도와 그 횡단보도가 이만큼(원본 픽셀) 떨어져 있어도 붙은 것으로 보고 합친다
GAP_MAX_RATIO = 0.5  # 줄무늬 끝에서 연석 선까지가 횡단보도 길이(늘이는 테두리) × 이 값보다 멀면 채우지 않는다(엉뚱한 연석)
MERGE_EPS_PX = 1.5  # 합친 테두리를 점으로 줄일 때 허용 오차(원본 픽셀)

# 화면 모양 (픽셀). 글자는 Pillow + 맑은 고딕(eval/overlay.py와 같음). OpenCV는 한글을 못 그린다
FONT_FILE, FONT_BOLD = "malgun.ttf", "malgunbd.ttf"
BIG, TITLE, TEXT, SMALL, LABEL = 30, 24, 17, 14, 12
PAD, LINE_GAP, ACCENT = 12, 7, 6
BAR_H = 46          # 맨 위 진행 막대 높이
CARD_W = 400        # 미션 카드 폭
EX_W, EX_H = 250, 140  # 예시 그림 크기
WHITE, GRAY, YELLOW, GREEN = (255, 255, 255), (170, 170, 175), (255, 214, 10), (60, 190, 110)  # RGB
CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩"


class Step(NamedTuple):
    key: str
    n: int | str         # 정수 = 정확히 그 개수, POLY = 4점 이상 다각형, LINES = 2점 이상 꺾은선 여러 개
    skippable: bool
    name: str            # 카드 제목
    short: str           # 진행 막대 이름
    tips: tuple          # 카드 안내(한 줄에 하나)
    points: tuple        # 점마다 찍을 곳(순서대로). 다각형은 비거나 앞 몇 개만
    terminal: str        # 터미널 안내


# 안내는 점을 어디에, 어떤 순서로 찍는지 적는다(tracking/README.md "클릭 순서"·"모서리 찍는 법", 기술서 2.2·2.3과 같게).
STEPS = [
    Step("corners_px", 4, False, "출구 횡단보도 네 모서리", "출구 모서리",
         ("꺾은 뒤 마지막으로 지나는 횡단보도", "위·아래 줄 각각 끝 블록의 바깥 모서리"),
         ("왼쪽 위", "오른쪽 위", "오른쪽 아래", "왼쪽 아래"),
         "출구 횡단보도(꺾은 뒤 마지막으로 지나는 곳) 네 모서리: 좌상 → 우상 → 우하 → 좌하 (화면 기준 시계 방향). "
         "블록 두 줄 각각의 첫 블록과 끝 블록 바깥 모서리(두 줄이 하나씩 엇갈려 있으면 엇갈린 그대로). 연석까지 빈 차도는 연석 선 단계에서 채운다"),
    Step("near_px", 4, True, "진입 횡단보도", "진입 횡단보도",
         ("정지선 바로 앞 횡단보도", "위·아래 줄 각각 끝 블록의 바깥 모서리"),
         ("왼쪽 위", "오른쪽 위", "오른쪽 아래", "왼쪽 아래"),
         "진입 횡단보도(정지선 바로 앞, 꺾기 전에 지나는 곳) 네 모서리(블록 두 줄 각각의 끝 블록 바깥 모서리): 좌상 → 우상 → 우하 → 좌하. "
         "연석까지 빈 차도는 뒤의 '빈 차도' 단계에서 찍는다. 보이는 부분만. 안 보이면 s"),
    Step("diag_px", 4, True, "중간 횡단보도 (있을 때만)", "중간 횡단보도",
         ("꺾는 도중 밟는 횡단보도 (대각선 등)", "각 줄 끝 블록의 바깥 모서리 · 없으면 s"),
         ("왼쪽 위", "오른쪽 위", "오른쪽 아래", "왼쪽 아래"),
         "중간 횡단보도(꺾는 도중 밟는 곳, 스크램블 대각선 등) 네 모서리(블록 두 줄 각각의 끝 블록 바깥 모서리): 좌상 → 우상 → 우하 → 좌하. "
         "끝부분만 걸쳐도 그 횡단보도 전체(보이는 부분). 없으면 s"),
    Step("curb_px", LINES, True, "연석 선", "연석 선",
         ("연석(차도와 인도 경계)을 넉넉히 따라 찍는다", "→ 줄무늬 끝 ~ 연석이 자동으로 채워진다", "Enter = 선 하나 끝 · 반대쪽도 그린다"),
         (),
         "연석 선: 횡단보도 끝이 닿는 연석(차도와 인도 경계)을 따라 2점 이상 → Enter = 선 하나. 둥근 모퉁이는 점 여러 개로 따라간다. "
         "선을 그리면 각 횡단보도(네 꼭짓점)의 양옆 테두리를 연석 선까지 곧게 늘여 줄무늬 끝 ~ 연석 사이 빈 차도를 채운다(화면에 반투명으로 보임). "
         "양쪽 끝 연석을 모두 그린다. 횡단보도끼리 겹친 자리(빨강)는 저장할 때 corner 영역으로 따로 둔다. 다 그렸으면 점 없이 Enter, 빈 차도가 없으면 s"),
    Step("approach_roi_px", 4, False, "우회전 진입로", "진입로",
         ("우회전 차로 한 칸을 띠로 감싼다", "뒤 끝: 정지선 뒤 차 2대 · 앞 끝: 꺾기 시작"),
         ("뒤 끝(차 2대 뒤) · 차로 한쪽 선", "뒤 끝 · 차로 반대쪽 선", "앞 끝(꺾기 시작) · ②쪽", "앞 끝 · ①쪽"),
         "우회전 진입로 영역 4점 (우회전 차로부터 정지선을 지나 꺾기 시작하는 곳까지, 우회전 경로만, 차 한 대 폭). "
         "직진 차로는 넣지 않고 출구 횡단보도에 닿지 않게"),
    Step("stop_line_px", 2, True, "정지선", "정지선",
         ("흰 정지선의 횡단보도 쪽 가장자리",),
         ("한쪽 끝", "반대쪽 끝"),
         "정지선 양 끝 2점 (2단계). 흰 정지선의 횡단보도 쪽 가장자리에 찍는다(엄격 구간 경계). "
         "정지선이 없으면 진입 쪽 횡단보도 앞 가장자리 두 점. 2단계를 안 하면 s"),
    Step("stop_zone_px", POLY, True, "정지 구간", "정지 구간",
         ("정지선~횡단보도 간격을 재서", "정지선 뒤로 같은 길이만큼"),
         ("정지선 뒤 같은 간격 · 한쪽 끝", "정지선 뒤 같은 간격 · 반대쪽 끝", "횡단보도 시작선 (②쪽)", "횡단보도 시작선 나머지"),
         "정지 구간 다각형 3점 이상 (2단계, 기술서 2.3). 먼 변 = 정지선 뒤로 정지선~진입 쪽 횡단보도 간격만큼 떨어진 선(화면의 그 간격을 그대로 옮김), "
         "가까운 변 = 진입 쪽 횡단보도 시작선(줄무늬의 정지선 쪽 가장자리). 2단계를 안 하면 s"),
    Step("signal_roi_px", 4, True, "신호등 빨간 불", "신호등",
         ("차량 신호등의 빨간 불만", "휠로 확대해서 찍기"),
         ("왼쪽 위", "오른쪽 위", "오른쪽 아래", "왼쪽 아래"),
         "우회전 방향 차량 신호등의 빨간 램프만 감싸는 4점 (초록·노랑 램프와 배경 제외. 2단계, 보행자 신호등 아님, s로 건너뛰기)"),
]
COLORS = {"corners_px": (0, 255, 255), "area_px": (0, 140, 255), "near_px": (255, 200, 0), "diag_px": (255, 255, 0),  # BGR
          "curb_px": (203, 192, 255),
          "approach_roi_px": (0, 255, 0), "stop_line_px": (0, 0, 255), "stop_zone_px": (128, 0, 255),
          "signal_roi_px": (255, 0, 255)}
LABELS = {"corners_px": "exit", "near_px": "near", "diag_px": "diag"}  # 화면에서 영역에 붙이는 이름
# --only 에 쓰는 짧은 이름
ONLY = {"crosswalk": "corners_px", "near": "near_px", "diag": "diag_px", "curb": "curb_px",
        "approach": "approach_roi_px",
        "stop_line": "stop_line_px", "stop_zone": "stop_zone_px", "signal": "signal_roi_px"}
CROSSWALK_KEYS = ("corners_px", "area_px")  # yaml에서 crosswalk 아래에 들어가는 키
OTHER_KEYS = {"near_px": "near", "diag_px": "diag"}  # yaml에서 other_crosswalks_px 아래 이름
CURB_KEY = "curb_px"  # 연석 선 [꺾은선, ...]. yaml에 두지 않고, 채운 빈 차도를 저장할 때 그 횡단보도에 합친다


class Video:
    """원하는 시각의 장면을 읽는다. 창을 연 동안 영상을 열어 둔다."""

    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise SystemExit(f"영상을 읽을 수 없다: {path}")
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.duration = max(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps - 1.0, 0.0)  # 마지막 프레임 시각으로 이동하면 읽히지 않는 영상이 있어(55초 영상 54.8초) 1초 여유

    def frame_at(self, sec: float) -> tuple[np.ndarray, float]:
        sec = min(max(sec, 0.0), self.duration)
        self.cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ok, frame = self.cap.read()
        if not ok:
            raise SystemExit(f"{sec:.0f}초 장면을 읽을 수 없다")
        return frame, sec

    def release(self) -> None:
        self.cap.release()


# ---------- 예시 그림 (카드 안, OpenCV로 그림. 숫자만 쓰므로 한글 글꼴이 필요 없다) ----------

def stripes(img: np.ndarray, x0: int, y0: int, count: int, w: int, h: int, gap: int, stacked: bool = False,
            color: tuple = (235, 235, 235)) -> tuple[int, int, int, int]:
    """줄무늬 count개를 그리고 감싼 사각형 (x0, y0, x1, y1)을 돌려준다. stacked = 가로 줄무늬를 위아래로 쌓음."""
    for i in range(count):
        if stacked:
            cv2.rectangle(img, (x0, y0 + i * (h + gap)), (x0 + w, y0 + i * (h + gap) + h), color, -1)
        else:
            cv2.rectangle(img, (x0 + i * (w + gap), y0), (x0 + i * (w + gap) + w, y0 + h), color, -1)
    if stacked:
        return x0, y0, x0 + w, y0 + count * (h + gap) - gap
    return x0, y0, x0 + count * (w + gap) - gap, y0 + h


def box_corners(box: tuple) -> list[tuple[int, int]]:
    x0, y0, x1, y1 = box
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def car_path(img: np.ndarray, points: list) -> None:
    """우회전 차 경로 화살표(노랑)."""
    cv2.polylines(img, [np.array(points[:-1], np.int32)], False, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.arrowedLine(img, points[-2], points[-1], (0, 220, 255), 2, cv2.LINE_AA, tipLength=0.4)


def example(key: str, n_cur: int, blink: bool, color: tuple) -> tuple[np.ndarray, list]:
    """단계마다 어디를 어떤 순서로 찍는지 보여 주는 작은 그림(BGR)과 한글 이름표 [(x, y, 글자)](카드에서 Pillow로 씀).
    찍은 점은 색, 다음 점은 깜박이는 노란 고리."""
    img = np.full((EX_H, EX_W, 3), 75, np.uint8)
    dim = (130, 130, 130)
    labels = []
    if key == "corners_px":  # 비스듬한 블록 두 줄이 하나씩 엇갈림(아래 줄은 반 칸 오른쪽) → 각 줄 첫·끝 블록 바깥 모서리
        x0, lean, w, gap = 20, 0.3, 8, 18  # 블록 왼쪽 아래 x, 기울기(위로 갈수록 오른쪽), 블록 폭, 블록 간격

        def at(i: float, y: int, right: bool = False) -> tuple[int, int]:
            return round(x0 + i * gap + (w if right else 0) + (102 - y) * lean), y

        for top, bottom, shift in ((26, 60, 0.0), (68, 102, 0.5)):  # (위 y, 아래 y, 반 칸 어긋남)
            for i in range(9):
                cv2.fillConvexPoly(img, np.array([at(i + shift, bottom), at(i + shift, bottom, True),
                                                  at(i + shift, top, True), at(i + shift, top)], np.int32), (235, 235, 235))
        targets = [at(0, 26), at(8, 26, True), at(8.5, 102, True), at(0.5, 102)]  # 위 줄 첫·끝, 아래 줄 끝·첫
    elif key == "curb_px":  # 할 일: 오른쪽 둥근 연석 위에 점 → 결과: 줄무늬 끝 ~ 연석이 채워짐. 왼쪽 끝 연석도 선 하나
        curb = [(204, 0), (204, 44), (209, 70), (221, 92), (240, 108), (EX_W, 114)]
        sidewalk = (110, 130, 150)
        img[:, :32] = sidewalk                                              # 왼쪽 인도
        cv2.fillPoly(img, [np.array(curb + [(EX_W, 0)], np.int32)], sidewalk)  # 오른쪽 인도(둥근 모퉁이)
        cv2.line(img, (32, 0), (32, EX_H), (90, 90, 200), 2)               # 연석(차도와 인도 경계)
        cv2.polylines(img, [np.array(curb, np.int32)], False, (90, 90, 200), 2, cv2.LINE_AA)
        x0, y0, x1, y1 = stripes(img, 62, 40, 7, 8, 56, 8)
        cv2.rectangle(img, (x0, y0), (x1, y1), COLORS["near_px"], 1)        # 이미 찍은 횡단보도
        fill = img.copy()
        cv2.fillPoly(fill, [np.array([(x1, y0), (x1, y1), (222, y1), (221, 92), (209, 70), (205, y0)], np.int32)], color)
        cv2.fillPoly(fill, [np.array([(33, y0), (x0, y0), (x0, y1), (33, y1)], np.int32)], color)
        img[:] = cv2.addWeighted(fill, 0.55 if n_cur >= 2 else 0.2, img, 0.45 if n_cur >= 2 else 0.8, 0)  # 점을 찍으면 채워진다
        targets = [(204, 14), (204, 44), (209, 70), (221, 92)]
        labels = [(1, 2, "인도"), (222, 26, "인도"), (148, 104, "자동으로 채워짐"), (1, 106, "반대쪽도"), (1, 120, "선 하나")]
    elif key in ("near_px", "diag_px"):  # 십자 교차로. 네 모서리는 줄무늬 끝(연석까지 빈 차도는 연석 선 단계에서)
        img[:] = (110, 130, 150)                                       # 인도
        img[:, 88:174], img[42:99] = 75, 75                            # 세로·가로 도로
        stripes(img, 178, 47, 6, 18, 4, 5, stacked=True, color=dim)    # 오른쪽 = 출구(이미 찍음)
        entry = stripes(img, 97, 104, 9, 4, 18, 4, color=(235, 235, 235) if key == "near_px" else dim)  # 아래 = 진입
        cv2.rectangle(img, (131, 128), (173, 131), (235, 235, 235), -1)  # 정지선(들어오는 차로)
        if key == "near_px":
            targets = box_corners(entry)
            labels = [(28, 104, "진입 →"), (176, 102, "출구(완료)")]
        else:  # 오른쪽 아래 모퉁이 → 왼쪽 위 모퉁이 대각선
            a, b = np.array([170.0, 95.0]), np.array([91.0, 45.0])
            d = (b - a) / np.linalg.norm(b - a)
            nrm = np.array([-d[1], d[0]])
            ts = np.arange(6, np.linalg.norm(b - a) - 4, 8)
            for t in ts:
                c = a + d * t
                cv2.line(img, tuple((c - nrm * 9).astype(int)), tuple((c + nrm * 9).astype(int)), (235, 235, 235), 3)
            first, last = a + d * (ts[0] - 1.5), a + d * (ts[-1] + 1.5)  # 양 끝 줄무늬의 바깥
            targets = [tuple((last + nrm * 10).astype(int)), tuple((first + nrm * 10).astype(int)),  # 왼쪽 위부터 시계 방향
                       tuple((first - nrm * 10).astype(int)), tuple((last - nrm * 10).astype(int))]
            labels = [(10, 20, "대각선 →"), (176, 102, "출구(완료)")]
        car_path(img, [(152, 138), (152, 92), (166, 80), (248, 80)])
    elif key == "approach_roi_px":  # 우회전 차로 한 칸: 뒤 끝 = 정지선 뒤 차 2대, 앞 끝 = 정지선 지나 꺾기 시작하는 곳
        cv2.line(img, (125, 0), (125, EX_H), (0, 200, 255), 2)         # 중앙선
        for y in range(62, EX_H, 16):                                    # 차로 구분선
            cv2.line(img, (188, y), (188, y + 8), (235, 235, 235), 2)
        cv2.rectangle(img, (125, 52), (EX_W, 57), (235, 235, 235), -1)  # 정지선
        for top in (66, 100):                                            # 기다리는 차 2대
            cv2.rectangle(img, (203, top), (231, top + 28), (170, 120, 70), -1)
            cv2.rectangle(img, (206, top + 4), (228, top + 10), (90, 60, 40), -1)
        car_path(img, [(217, 60), (217, 30), (229, 16), (249, 12)])
        targets = [(194, 133), (240, 133), (240, 28), (194, 28)]
        labels = [(130, 46, "정지선 →"), (134, 96, "차 2대 →"), (128, 126, "뒤 끝 →"), (128, 21, "앞 끝 →")]
    elif key == "stop_line_px":
        stripes(img, 25, 12, 12, 8, 38, 8)
        cv2.rectangle(img, (25, 72), (210, 82), (235, 235, 235), -1)  # 정지선. 위쪽(횡단보도 쪽) 가장자리를 찍는다
        car_path(img, [(120, 138), (120, 96)])
        targets = [(25, 72), (210, 72)]
    elif key == "stop_zone_px":  # 정지선 뒤로 정지선~횡단보도 간격만큼
        x0, _, x1, start = stripes(img, 25, 8, 10, 8, 32, 8)          # 횡단보도 시작선 = 줄무늬 아래 끝
        line, gap = 72, 72 - start                                    # 정지선 앞 가장자리, 간격
        cv2.rectangle(img, (x0, line), (x1, line + 7), (235, 235, 235), -1)  # 정지선
        for x in range(x0, x1, 12):                                    # 정지선 뒤 같은 간격 선
            cv2.line(img, (x, line + gap), (x + 6, line + gap), color, 2)
        for y0, y1 in ((start, line), (line, line + gap)):             # 같은 길이 표시
            cv2.arrowedLine(img, (x1 + 12, (y0 + y1) // 2), (x1 + 12, y0 + 2), (235, 235, 235), 1, tipLength=0.3)
            cv2.arrowedLine(img, (x1 + 12, (y0 + y1) // 2), (x1 + 12, y1 - 2), (235, 235, 235), 1, tipLength=0.3)
        targets = [(x0, line + gap), (x1, line + gap), (x1, start), (x0, start)]
        labels = [(x1 + 18, start + gap // 2 - 8, "간격"), (x1 + 18, line + gap // 2 - 8, "같게")]
    else:  # signal_roi_px
        cv2.rectangle(img, (115, 95), (125, EX_H), (40, 40, 40), -1)  # 기둥
        cv2.rectangle(img, (45, 50), (195, 95), (25, 25, 25), -1)     # 신호등 몸체
        for cx, lamp in ((75, (0, 0, 255)), (120, (0, 70, 90)), (165, (0, 60, 0))):  # 빨강(켜짐)·노랑·초록
            cv2.circle(img, (cx, 72), 16, lamp, -1)
        targets = box_corners((56, 53, 94, 91))

    pts = np.array(targets, np.int32)
    cv2.polylines(img, [pts], len(targets) > 2 and key != "curb_px", dim, 1, cv2.LINE_AA)  # 완성 모양 미리 보기(연석 선은 열린 선)
    if n_cur > 1:
        cv2.polylines(img, [pts[:n_cur]], False, color, 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(targets):
        if i < n_cur:
            cv2.circle(img, (x, y), 6, color, -1)
        elif i == n_cur:
            cv2.circle(img, (x, y), 10 if blink else 7, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.circle(img, (x, y), 5, dim, 1, cv2.LINE_AA)
        tx = x + 8 if x + 8 <= EX_W - 12 else x - 18  # 오른쪽 끝 점은 번호를 왼쪽에
        ty = min(max(y - 8, 12), EX_H - 3)
        cv2.putText(img, str(i + 1), (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255) if i <= n_cur else dim, 1, cv2.LINE_AA)
    return img, labels


# ---------- 화면 그리기 ----------

@lru_cache
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_FILE, size)


def wrap(text: str, size: int, width: float, bold: bool = False) -> list[str]:
    """폭에 맞춰 띄어쓰기 단위로 줄을 나눈다."""
    rows, row = [], ""
    for word in text.split(" "):
        longer = f"{row} {word}".strip()
        if row and font(size, bold).getlength(longer) > width:
            rows.append(row)
            row = word
        else:
            row = longer
    return rows + [row]


def rgb(bgr: tuple) -> tuple:
    return tuple(bgr[::-1])


def para(text: str, size: int, color: tuple, width: float, bold: bool = False) -> list[tuple]:
    """카드 글자 줄들: (글자, 크기, RGB, 굵게, 들여쓰기)."""
    return [(r, size, color, bold, 0) for r in wrap(text, size, width, bold)]


def row_height(row: tuple) -> int:
    return (row[1].shape[0] if row[0] is None else row[1]) + LINE_GAP


def card_height(rows: list, button: list | None) -> int:
    h = PAD + sum(row_height(r) for r in rows) + PAD
    return h + (sum(row_height(r) for r in button) + 16 if button else 0)


def paint_card(layer: Image.Image, pen: ImageDraw.ImageDraw, rows: list, x: int, y: int, w: int,
               accent: tuple | None = None, button: list | None = None, button_fill: tuple = YELLOW) -> None:
    """어두운 둥근 카드. rows = para() 줄 또는 (None, 그림 BGR, 이름표). button = 카드 아래 강조 칸의 글자 줄."""
    h = card_height(rows, button)
    pen.rounded_rectangle([x, y, x + w, y + h], 12, fill=(18, 18, 24, 235))
    left = x + PAD
    if accent:
        pen.rectangle([x + 5, y + 10, x + 5 + ACCENT, y + h - 10], fill=accent + (255,))
        left += ACCENT + 4
    ty = y + PAD
    for r in rows:
        if r[0] is None:
            layer.paste(Image.fromarray(cv2.cvtColor(r[1], cv2.COLOR_BGR2RGB)).convert("RGBA"), (int(left), int(ty)))
            for lx, ly, text in r[2]:
                pen.text((left + lx, ty + ly), text, font=font(LABEL, True), fill=WHITE + (255,),
                         stroke_width=2, stroke_fill=(0, 0, 0, 255))
        else:
            text, size, color, bold, indent = r
            pen.text((left + indent, ty), text, font=font(size, bold), fill=color + (255,))
        ty += row_height(r)
    if button:
        bh = sum(row_height(r) for r in button) + 8
        pen.rounded_rectangle([left, ty + 2, x + w - PAD, ty + 2 + bh], 8, fill=button_fill + (255,))
        by = ty + 7
        for text, size, color, bold, indent in button:
            pen.text((left + 8 + indent, by), text, font=font(size, bold), fill=color + (255,))
            by += row_height((text, size))


def progress(pen: ImageDraw.ImageDraw, steps: list, step: int, shown: dict, width: int) -> None:
    """맨 위 진행 막대: 끝낸 미션 초록(✓), 건너뛴 미션 회색(-), 지금 미션 그 색, 남은 미션 어둡게."""
    pen.rectangle([0, 0, width, BAR_H], fill=(10, 10, 15, 210))
    f = font(SMALL, True)
    labels = [f"{i} {s.short}" for i, s in enumerate(steps, 1)]
    if sum(f.getlength(t) + 34 for t in labels) > width - 2 * PAD:  # 좁은 화면이면 번호만
        labels = [str(i) for i in range(1, len(steps) + 1)]
    x, top, bottom = PAD, 9, BAR_H - 9
    for i, (s, label) in enumerate(zip(steps, labels)):
        w = f.getlength(label) + 28
        if i < step:
            skipped = not shown.get(s.key)
            pen.rounded_rectangle([x, top, x + w, bottom], 13, fill=((80, 80, 85) if skipped else GREEN) + (255,))
            mid = (top + bottom) // 2
            if skipped:
                pen.line([(x + 8, mid), (x + 16, mid)], fill=WHITE + (255,), width=2)
            else:
                pen.line([(x + 7, mid), (x + 11, mid + 4), (x + 18, mid - 5)], fill=WHITE + (255,), width=3)
            pen.text((x + 22, top + 5), label.split(" ", 1)[-1] if " " in label else "", font=f, fill=WHITE + (255,))
        elif i == step:
            pen.rounded_rectangle([x, top, x + w, bottom], 13, fill=rgb(COLORS[s.key]) + (255,))
            pen.text((x + 14, top + 5), label, font=f, fill=(15, 15, 15, 255))
        else:
            pen.rounded_rectangle([x, top, x + w, bottom], 13, outline=(90, 90, 95, 255), fill=(35, 35, 40, 255))
            pen.text((x + 14, top + 5), label, font=f, fill=GRAY + (255,))
        x += w + 6


def keybar(pen: ImageDraw.ImageDraw, items: list, size: tuple, info: str = "") -> None:
    """맨 아래 키 안내: [키] 설명 … (오른쪽 끝에 확대 배율·장면 시각). 폭이 모자라면 줄을 늘린다."""
    width, height = size
    f_key, f_lab = font(SMALL, True), font(SMALL)
    cells = [(cap, label, f_key.getlength(cap) + 14, f_lab.getlength(label)) for cap, label in items]
    limit = width - 2 * PAD - f_lab.getlength(info) - 20
    rows, row, used = [], [], 0.0
    for c in cells:
        total = c[2] + 6 + c[3] + 18
        if row and used + total > limit:
            rows.append(row)
            row, used = [], 0.0
        row.append(c)
        used += total
    rows.append(row)
    rh = SMALL + 16
    top = height - len(rows) * rh - 6
    pen.rectangle([0, top, width, height], fill=(10, 10, 15, 210))
    for i, row in enumerate(rows):
        x, y = PAD, top + 5 + i * rh
        for cap, label, cw, lw in row:
            pen.rounded_rectangle([x, y, x + cw, y + rh - 6], 5, fill=(235, 235, 235, 255))
            pen.text((x + 7, y + 3), cap, font=f_key, fill=(20, 20, 20, 255))
            pen.text((x + cw + 6, y + 3), label, font=f_lab, fill=(225, 225, 225, 255))
            x += cw + 6 + lw + 18
    if info:
        pen.text((width - PAD - f_lab.getlength(info), top + 8), info, font=f_lab, fill=GRAY + (255,))


def pill(pen: ImageDraw.ImageDraw, text: str, x: float, y: float, color: tuple, size: tuple) -> None:
    """커서 옆 작은 이름표(화면 밖으로 안 나가게)."""
    f = font(SMALL, True)
    w = f.getlength(text) + 16
    x, y = min(x, size[0] - w - 4), min(y, size[1] - SMALL - 16)
    pen.rounded_rectangle([x, y, x + w, y + SMALL + 12], 8, fill=(0, 0, 0, 210), outline=color + (255,))
    pen.text((x + 8, y + 4), text, font=f, fill=color + (255,))


def toast_box(pen: ImageDraw.ImageDraw, text: str, size: tuple) -> None:
    """화면 가운데 큰 알림(미션 완료 등)."""
    f = font(TITLE, True)
    w = f.getlength(text) + 60
    x, y = (size[0] - w) / 2, size[1] * 0.42
    pen.rounded_rectangle([x, y, x + w, y + TITLE + 34], 16, fill=(18, 18, 24, 235), outline=GREEN + (255,), width=3)
    pen.text((x + 30, y + 15), text, font=f, fill=WHITE + (255,))


def next_hint(step: Step, n_cur: int, n_lines: int = 0) -> tuple[str, bool]:
    """(다음에 할 일, 이 단계를 Enter로 끝낼 수 있는가). n_lines = 연석 선 단계에서 다 그린 선 수."""
    if step.n == LINES:
        if n_cur >= 2:
            return "이 선 끝났으면 Enter · 둥근 곳은 점 더", True
        if n_cur == 1:
            return "연석을 따라 다음 점 (1/2)", False
        return (f"선 {n_lines}개 · 더 없으면 Enter", True) if n_lines else ("① 연석 위에 첫 점 · 빈 차도 없으면 s", False)
    if isinstance(step.n, int) and n_cur >= step.n:
        return "Enter → 다음", True
    if n_cur == 0 and step.key == "diag_px":
        return f"없으면 s · 있으면 ① {step.points[0]}", False
    if n_cur < len(step.points):
        return f"{CIRCLED[n_cur]} {step.points[n_cur]}", False
    if n_cur >= 4:
        return "다 감쌌으면 Enter", True
    if n_cur == 0:
        return "① 왼쪽 위부터 시계 방향으로", False
    return f"둘레를 따라 시계 방향으로 ({n_cur}/4)", False


def crosswalk_polys(shown: dict) -> dict:
    """횡단보도 이름 → 지금까지 정한 줄무늬 다각형(출구는 판정 영역, 없으면 네 모서리). 안 찍은 횡단보도는 뺀다."""
    polys = {"exit": shown.get("area_px") or shown.get("corners_px") or [],
             **{name: shown.get(key) or [] for key, name in OTHER_KEYS.items()}}
    return {name: p for name, p in polys.items() if len(p) >= 3}


def masks_of(*polys: list) -> tuple[list, np.ndarray]:
    """다각형들을 같은 원점의 0/1 그림으로. (그림들, 원점). 둘레에 JOIN_PX + 1 여유."""
    pts = np.concatenate([np.array(p, np.int32) for p in polys])
    origin = pts.min(axis=0) - JOIN_PX - 1
    w, h = pts.max(axis=0) - origin + JOIN_PX + 2
    out = []
    for p in polys:
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.array(p, np.int32) - origin], 1)
        out.append(m)
    return out, origin


def mask_to_polys(mask: np.ndarray, origin: np.ndarray) -> list[list]:
    """0/1 그림의 덩어리마다 바깥 테두리 다각형(원본 좌표)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [(cv2.approxPolyDP(c, MERGE_EPS_PX, True).reshape(-1, 2) + origin).tolist() for c in contours if len(c) >= 3]


def shared_mask(masks: list) -> np.ndarray:
    """두 영역 이상이 OVERLAP_PX보다 두껍게 겹친 자리(0/1). 얇게 맞닿은 곳은 뺀다."""
    shared = (sum(m.astype(np.uint16) for m in masks) >= 2).astype(np.uint8)
    return cv2.morphologyEx(shared, cv2.MORPH_OPEN, np.ones((2 * OVERLAP_PX + 1,) * 2, np.uint8))


def split_corner(polys: dict) -> tuple[dict, list]:
    """횡단보도 영역들 {이름: 다각형}에서 서로 겹친 자리를 떼어 낸다 → (겹침을 뺀 영역들, corner 다각형들).
    떼고 나서 한 덩어리가 아니게 되는(가운데가 뚫리거나 둘로 갈라지는) 영역이 있으면 떼지 않고 그대로 둔다."""
    names = [n for n, p in polys.items() if len(p) >= 3]
    if len(names) < 2:
        return polys, []
    masks, origin = masks_of(*[polys[n] for n in names])
    shared = shared_mask(masks)
    if not shared.any():
        return polys, []
    kernel = np.ones((2 * OVERLAP_PX + 1,) * 2, np.uint8)
    for _ in range(len(names) + 1):  # 떼고 남은 얇은 조각(OVERLAP_PX 이하)은 corner로 넘긴다. 넘기면 다른 영역도 다시 본다
        grown = False
        for m in masks:
            num, labels = cv2.connectedComponents(m & (1 - shared))
            if num <= 2:
                continue
            sizes = np.bincount(labels.ravel())[1:]
            for i in np.flatnonzero(np.arange(1, num) != sizes.argmax() + 1) + 1:
                piece = (labels == i).astype(np.uint8)
                if not cv2.morphologyEx(piece, cv2.MORPH_OPEN, kernel).any():
                    shared |= piece
                    grown = True
        if not grown:
            break
    out = dict(polys)
    for name, m in zip(names, masks):
        if not (m & shared).any():
            continue
        rest = m & (1 - shared)
        _, hierarchy = cv2.findContours(rest, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        pieces = mask_to_polys(rest, origin)
        if len(pieces) != 1 or (hierarchy[0][:, 3] >= 0).any():
            print(f"안내: {name} 영역은 겹친 자리를 떼면 한 덩어리가 안 되어 겹친 채로 둔다(줄무늬가 크게 겹치게 찍혔는지 확인)")
            return polys, []
        out[name] = pieces[0]
    return out, mask_to_polys(shared, origin)


def ray_hit(o: np.ndarray, d: np.ndarray, line: np.ndarray) -> tuple[float, int, np.ndarray] | None:
    """o에서 d 방향 반직선이 꺾은선 line과 처음 만나는 곳: (거리 배수 t, 선분 번호, 점). 안 만나면 None."""
    best = None
    for i, (a, b) in enumerate(zip(line[:-1], line[1:])):
        e, w = b - a, a - o
        den = d[0] * e[1] - d[1] * e[0]
        if abs(den) < 1e-9:  # 나란함
            continue
        t, u = (w[0] * e[1] - w[1] * e[0]) / den, (w[0] * d[1] - w[1] * d[0]) / den
        if t >= 0 and 0 <= u <= 1 and (best is None or t < best[0]):
            best = (t, i, o + d * t)
    return best


def extended(line: np.ndarray, by: float) -> np.ndarray:
    """연석 선 양 끝을 끝 선분 방향으로 by만큼 늘인다(선이 횡단보도 띠보다 조금 짧게 그려져도 테두리와 만나게)."""
    head, tail = line[0] - line[1], line[-1] - line[-2]
    nh, nt = np.linalg.norm(head), np.linalg.norm(tail)
    if nh == 0 or nt == 0:
        return line
    return np.vstack([line[0] + head / nh * by, line, line[-1] + tail / nt * by])


def fill_to_curb(quad: np.ndarray, line: np.ndarray) -> list | None:
    """횡단보도 네 꼭짓점 quad의 한 변(줄무늬 끝)에서 양옆 테두리를 곧게 늘여 연석 선 line까지 채운 다각형.
    네 변 중 연석에 가장 가까운 변을 쓴다. 연석 선은 양 끝을 그 변 길이만큼 늘여 본다(조금 짧게 그려도 됨).
    테두리 길이 × GAP_MAX_RATIO보다 멀거나 안 만나면 None."""
    best = None
    for j in range(4):
        a, b = quad[j], quad[(j + 1) % 4]
        side_a, side_b = a - quad[(j - 1) % 4], b - quad[(j + 2) % 4]  # 줄무늬 끝 변 양옆 테두리(바깥 방향)
        la, lb = np.linalg.norm(side_a), np.linalg.norm(side_b)
        if la == 0 or lb == 0:
            continue
        line_j = extended(line, np.linalg.norm(b - a))
        ha, hb = ray_hit(a, side_a / la, line_j), ray_hit(b, side_b / lb, line_j)
        if ha is None or hb is None or ha[0] > la * GAP_MAX_RATIO or hb[0] > lb * GAP_MAX_RATIO:
            continue
        if best is None or max(ha[0], hb[0]) < best[0]:
            # 연석 선이 휘어 있으면 두 교점 사이의 연석 점을 따라간다(b 쪽 교점 → a 쪽 교점 순서)
            mid = line_j[hb[1] + 1:ha[1] + 1] if hb[1] <= ha[1] else line_j[ha[1] + 1:hb[1] + 1][::-1]
            best = (max(ha[0], hb[0]), [a, b, hb[2], *mid, ha[2]])
    if best is None or best[0] < 1:  # 줄무늬 끝이 이미 연석에 닿음
        return None
    return np.round(np.array(best[1])).astype(int).tolist()


def curb_fills(shown: dict, lines: list) -> dict:
    """횡단보도 이름 → 연석 선으로 채운 빈 차도 다각형들. 네 꼭짓점으로 찍은 횡단보도만(출구는 네 모서리)."""
    quads = {"exit": shown.get("corners_px") or [], **{name: shown.get(key) or [] for key, name in OTHER_KEYS.items()}}
    out = {}
    for name, quad in quads.items():
        if len(quad) != 4:
            continue
        for line in lines:
            if len(line) >= 2 and (poly := fill_to_curb(np.array(quad, float), np.array(line, float))):
                out.setdefault(name, []).append(poly)
    return out


def merge(base: list, gap: list) -> list | None:
    """횡단보도 다각형과 빈 차도를 한 다각형으로 합친다. JOIN_PX 넘게 떨어져 따로 놀면 None."""
    (a, b), origin = masks_of(base, gap)
    joined = cv2.morphologyEx(a | b, cv2.MORPH_CLOSE, np.ones((2 * JOIN_PX + 1,) * 2, np.uint8))
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 1:
        return None
    return (cv2.approxPolyDP(contours[0], MERGE_EPS_PX, True).reshape(-1, 2) + origin).tolist()


def draw_regions(img: np.ndarray, k: float, x0: float, y0: float, shown: dict, current: list,
                 cur: Step | None, cursor: tuple | None) -> None:
    """찍은 영역(점·선·이름)과 지금 찍는 점, 커서 십자선·고무줄 선을 원본 좌표 → 화면 좌표로 그린다."""
    lines = list(shown.get(CURB_KEY) or []) + ([current] if cur and cur.key == CURB_KEY and len(current) >= 2 else [])
    fills = [poly for polys in curb_fills(shown, lines).values() for poly in polys]
    if fills:  # 연석 선으로 채운 빈 차도(반투명)와, 횡단보도 영역끼리 겹쳐 corner로 뗄 자리(빨강)
        tinted = img.copy()
        cv2.fillPoly(tinted, [((np.array(f) - [x0, y0]) * k).astype(np.int32) for f in fills], COLORS[CURB_KEY])
        areas = {name: [base, *curb_fills(shown, lines).get(name, [])] for name, base in crosswalk_polys(shown).items()}
        view = []
        for polys in areas.values():
            m = np.zeros(img.shape[:2], np.uint8)
            cv2.fillPoly(m, [((np.array(p) - [x0, y0]) * k).astype(np.int32) for p in polys], 1)
            view.append(m)
        if len(view) >= 2:
            tinted[shared_mask(view).astype(bool)] = (60, 60, 230)
        img[:] = cv2.addWeighted(tinted, 0.4, img, 0.6, 0)
    layers = []  # (키, 점들, 닫힌 도형인가, 이름)
    for key, val in shown.items():
        if key == CURB_KEY:
            layers += [(key, line, False, None) for line in val]
        elif cur is None or key != cur.key:
            layers.append((key, val, len(val) > 2, LABELS.get(key) if val else None))
    if cur:
        layers.append((cur.key, current, len(current) == cur.n, None))
    for key, pts, closed, label in layers:
        if not pts:
            continue
        color = COLORS[key]
        p = ((np.array(pts) - [x0, y0]) * k).astype(np.int32)
        cv2.polylines(img, [p], closed, color, 2, cv2.LINE_AA)
        for n, (x, y) in enumerate(p, 1):
            cv2.circle(img, (int(x), int(y)), 5, color, -1)
            cv2.putText(img, str(n), (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if label:
            cx, cy = p.mean(axis=0).astype(int)
            cv2.putText(img, label, (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
            cv2.putText(img, label, (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    if cur and cursor:
        color, (cx, cy) = COLORS[cur.key], cursor
        if current and (not isinstance(cur.n, int) or len(current) < cur.n):  # 마지막 점 → 커서 고무줄 선
            last = ((np.array(current[-1]) - [x0, y0]) * k).astype(int)
            cv2.line(img, (int(last[0]), int(last[1])), (cx, cy), color, 1, cv2.LINE_AA)
        cv2.line(img, (cx - 14, cy), (cx + 14, cy), color, 1)
        cv2.line(img, (cx, cy - 14), (cx, cy + 14), color, 1)


def overlay(img: np.ndarray) -> tuple[Image.Image, Image.Image, ImageDraw.ImageDraw]:
    """(장면, 안내를 그릴 투명 층, 펜)."""
    canvas = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")
    layer = Image.new("RGBA", canvas.size)
    return canvas, layer, ImageDraw.Draw(layer)


def finish(canvas: Image.Image, layer: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(Image.alpha_composite(canvas, layer).convert("RGB")), cv2.COLOR_RGB2BGR)


def draw(view: np.ndarray, k: float, x0: float, y0: float, shown: dict, current: list, steps: list, step: int, *,
         zoom: float = 1.0, t_sec: float = 0.0, show_help: bool = True,
         cursor: tuple | None = None, toast: str = "", blink: bool = True) -> np.ndarray:
    """view = 원본을 (x0, y0)부터 k배로 보여 주는 화면. shown = 이미 정한 영역(이번에 찍은 것 + --only 일 때 기존 값).
    step == len(steps)면 마지막 확인 화면. show_help = False면 카드·키 안내를 숨긴다(화면 위쪽 신호등 등을 가릴 때 h)."""
    img = view.copy()
    cur = steps[step] if step < len(steps) else None
    draw_regions(img, k, x0, y0, shown, current, cur, cursor)
    canvas, layer, pen = overlay(img)
    size = canvas.size
    progress(pen, steps, step, shown, size[0])
    w = min(CARD_W, size[0] - 2 * PAD)
    inner = w - 2 * PAD - ACCENT - 4
    if cur is None:
        rows = para("모든 미션 완료!", TITLE, GREEN, inner, True) + para("영역을 확인한다", TEXT, WHITE, inner)
        paint_card(layer, pen, rows, PAD, BAR_H + PAD, w, GREEN,
                   para("Enter 저장 · Backspace 이전 · Esc 취소", TEXT + 1, (15, 15, 15), inner - 16, True))
    else:
        color = rgb(COLORS[cur.key])
        hint, ready = next_hint(cur, len(current), len(shown.get(CURB_KEY) or []))
        fill = GREEN if ready else YELLOW
        if show_help:
            rows = para(f"{step + 1}. {cur.name}", TITLE, color, inner, True)
            if cur.key == CURB_KEY:  # 어느 횡단보도 끝이 채워졌는지(안 채워진 끝은 연석 선을 더 길게)
                lines = list(shown.get(CURB_KEY) or []) + ([current] if len(current) >= 2 else [])
                fills = curb_fills(shown, lines)
                names = {"exit": "출구", "near": "진입", "diag": "중간"}
                quads = {"exit": shown.get("corners_px"), "near": shown.get("near_px"), "diag": shown.get("diag_px")}
                done_ends = " · ".join(f"{names[n]} {len(fills.get(n, []))}" for n, q in quads.items() if q and len(q) == 4)
                if done_ends:
                    rows += para(f"채워진 끝: {done_ends}", TEXT, YELLOW, inner, True)
            for tip in cur.tips:
                rows += para(tip, TEXT, WHITE, inner)
            if inner >= EX_W:
                rows.append((None, *example(cur.key, len(current), blink, COLORS[cur.key])))
            button = para("▶ " + hint, TEXT + 1, (15, 15, 15), inner - 16, True)
            x, y, h = PAD, BAR_H + PAD, card_height(rows, button)
            if cursor and x <= cursor[0] <= x + w and y <= cursor[1] <= y + h:  # 커서가 카드 위로 오면 반대쪽으로 비킨다
                x = size[0] - w - PAD
            paint_card(layer, pen, rows, x, y, w, color, button, fill)
            keys = [("클릭", "찍기"), ("우클릭", "지우기"), ("Enter", "다음"), *([("s", "건너뛰기")] if cur.skippable else []),
                    *([("Backspace", "이전 단계")] if step > 0 else []), ("휠", "확대"), ("n / b", "장면 넘기기"), ("h", "숨기기"), ("Esc", "끝")]
            keybar(pen, keys, size, f"{zoom:.1f}배 · {t_sec:.0f}초")
        else:
            keybar(pen, [("h", "안내 보기")], size)
        if cursor:
            pill(pen, hint, cursor[0] + 18, cursor[1] + 14, color if fill == YELLOW else fill, size)
    if toast:
        toast_box(pen, toast, size)
    return finish(canvas, layer)


def intro(view: np.ndarray, steps: list) -> np.ndarray:
    """시작 화면: 무엇을 몇 단계로 하는지와 기본 조작."""
    canvas, layer, pen = overlay(view)
    width, height = canvas.size
    pen.rectangle([0, 0, width, height], fill=(0, 0, 0, 150))
    w = min(360, width - 2 * PAD)
    inner = w - 2 * PAD - ACCENT - 4
    rows = para("교차로 영역 설정", BIG, WHITE, inner, True) + para("그림대로 점을 찍으면 된다", TEXT, WHITE, inner)
    for i, s in enumerate(steps, 1):
        rows += para(f"{i}. {s.short}", TEXT, rgb(COLORS[s.key]), inner, True)
    button = para("아무 키나 눌러 시작", TEXT + 1, (15, 15, 15), inner - 16, True)
    paint_card(layer, pen, rows, (width - w) // 2, max(PAD, (height - card_height(rows, button)) // 2), w, YELLOW, button)
    return finish(canvas, layer)


def collect(video: Video, start_sec: float = 0.0, steps: list = STEPS, existing: dict | None = None) -> dict | None:
    """창에서 단계별로 점을 받는다. Esc면 None. existing = 화면에 같이 보여 줄 기존 영역."""
    frame, t_sec = video.frame_at(start_sec)
    h, w = frame.shape[:2]
    scale = min(MAX_VIEW_W / w, MAX_VIEW_H / h)  # 작은 영상(사진 등)은 키워서 안내 카드에 덜 가리게
    view_size = (round(w * scale), round(h * scale))
    done, current, step = {}, [], 0
    later = {}  # Backspace로 떠난 단계에서 찍다 만 점(다시 오면 이어서 찍는다)
    curbs = []  # 연석 선 단계에서 다 그린 선들
    # 화면 좌표 v ↔ 원본 좌표 o: o = (x0, y0) + v / k,  k = scale × zoom
    vs = {"zoom": 1.0, "x0": 0.0, "y0": 0.0, "drag": None, "frame": frame, "t": t_sec, "help": True,
          "cursor": None, "toast": ("", 0.0)}

    def k() -> float:
        return scale * vs["zoom"]

    def clamp() -> None:
        vs["x0"] = min(max(vs["x0"], 0.0), w - w / vs["zoom"])
        vs["y0"] = min(max(vs["y0"], 0.0), h - h / vs["zoom"])

    def on_mouse(event, x, y, flags, _param):
        vs["cursor"] = (x, y)
        if step >= len(steps):  # 마지막 확인 화면: 점을 더 받지 않는다
            return
        ox, oy = vs["x0"] + x / k(), vs["y0"] + y / k()  # 커서 아래의 원본 좌표
        n = steps[step].n
        if event == cv2.EVENT_LBUTTONDOWN and (n in (POLY, LINES) or len(current) < n):
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

    def screen() -> np.ndarray:
        text, until = vs["toast"]
        shown = {**(existing or {}), **done}
        if step < len(steps) and steps[step].n == LINES:
            shown[CURB_KEY] = curbs
        return draw(render(), k(), vs["x0"], vs["y0"], shown, current, steps, step,
                    zoom=vs["zoom"], t_sec=vs["t"], show_help=vs["help"], cursor=vs["cursor"],
                    toast=text if time.monotonic() < until else "", blink=int(time.monotonic() * 2) % 2 == 0)

    win = "roi_tool"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    try:
        while True:  # 시작 화면
            cv2.imshow(win, intro(render(), steps))
            pressed = cv2.waitKey(30) & 0xFF
            if pressed == 27:
                return None
            if pressed != 255:
                break
        print(f"[{step + 1}/{len(steps)}] {steps[step].terminal}")
        while True:
            cv2.imshow(win, screen())
            pressed = cv2.waitKey(30) & 0xFF
            final = step == len(steps)  # 마지막 확인 화면
            if pressed == 27:
                return None
            if pressed in (ord("n"), ord("b")):
                delta = SEEK_SEC if pressed == ord("n") else -SEEK_SEC
                vs["frame"], vs["t"] = video.frame_at(vs["t"] + delta)  # 카메라 고정이라 찍은 점은 그대로 둔다
                continue
            if pressed == ord("h"):  # 안내 숨기기/보이기(화면 위쪽 신호등 등을 가릴 때)
                vs["help"] = not vs["help"]
                continue
            if pressed == 8 and step > 0:  # Backspace: 이전 단계로 돌아가 찍은 점을 이어서 고친다
                if not final:
                    later[steps[step].key] = current
                step -= 1
                current = list(done.pop(steps[step].key))
                if steps[step].n == LINES:  # 연석 선: 그린 선들을 되살리고 이어서 그린다
                    curbs, current = current, []
                print(f"[{step + 1}/{len(steps)}] (다시) {steps[step].terminal}")
                continue
            if final:
                if pressed == 13:
                    return done
                continue
            key, n, skippable = steps[step][:3]
            if pressed == ord("r"):  # 이 단계에서 찍은 점 모두 지우기
                current = []
                continue
            if n == LINES and pressed == 13:  # 연석 선: 2점 이상 Enter = 선 하나, 점 없이 Enter = 단계 끝
                if len(current) >= 2:
                    curbs.append(list(current))
                    current = []
                    continue
                if current:
                    continue
                done[key] = list(curbs)
            elif n == LINES and pressed == ord("s"):
                curbs, current = [], []
                done[key] = []
            elif pressed == 13 and (len(current) >= 4 if n == POLY else len(current) == n):
                done[key] = list(current)
            elif pressed == ord("s") and skippable:
                done[key] = []
            else:
                continue
            vs["toast"] = (("완료!" if done[key] else "건너뜀"), time.monotonic() + TOAST_SEC)
            step += 1
            if step < len(steps):
                current = later.pop(steps[step].key, [])
                print(f"[{step + 1}/{len(steps)}] {steps[step].terminal}")
            else:
                current = []
                print("마지막 확인: Enter = 저장, Backspace = 이전 단계, Esc = 저장 안 하고 끝")
    finally:
        cv2.destroyWindow(win)


def check_site(site: dict) -> None:
    """모서리 4점 확인. 실측치가 있으면 homography까지 만들어 본다. 없으면(0) 경고만."""
    cw = site["crosswalk"]
    if np.asarray(cw["corners_px"]).shape != (4, 2):
        raise ValueError(f"횡단보도 모서리는 4점이어야 한다: {cw['corners_px']}")
    if cw["width_m"] == 0 or cw["length_m"] == 0:
        print("경고: 실측치(width_m, length_m) 미입력. yaml에서 채우기 전에는 track을 돌릴 수 없다.")
    else:
        compute_homography(cw["corners_px"], cw["width_m"], cw["length_m"])  # 음수 등은 여기서 에러


def apply_curbs(site: dict, lines: list) -> None:
    """연석 선으로 채운 빈 차도를 그 횡단보도 다각형에 합친다(출구는 판정 영역, 비어 있으면 네 모서리에 합쳐 판정 영역으로).
    네 꼭짓점이 아닌 횡단보도(예전에 합친 것 등)는 늘일 방향을 몰라 건너뛴다."""
    cw, others = site["crosswalk"], site.setdefault("other_crosswalks_px", {})
    shown = {"corners_px": cw["corners_px"], **{key: others.get(name) or [] for key, name in OTHER_KEYS.items()}}
    for name, gaps in curb_fills(shown, lines or []).items():
        for gap in gaps:
            if name == "exit":
                cw["area_px"] = merge(cw.get("area_px") or cw["corners_px"], gap)
            else:
                others[name] = merge(others[name], gap)
    skipped = [name for name, pts in others.items() if pts and len(pts) != 4 and lines and not name.startswith("corner")]
    if skipped:
        print(f"안내: 네 꼭짓점이 아닌 횡단보도({', '.join(skipped)})는 연석까지 채우지 않았다(다시 찍으면 채워진다)")


def apply_corners(site: dict) -> None:
    """횡단보도 영역(출구 판정 영역·other_crosswalks_px)끼리 겹친 자리를 떼어 corner, corner2 …로 둔다(기술서 11.1).
    예전에 뗀 corner도 한 영역으로 같이 본다."""
    cw, others = site["crosswalk"], site.setdefault("other_crosswalks_px", {})
    polys = {"exit": cw.get("area_px") or cw["corners_px"], **others}
    polys, corners = split_corner(polys)
    if not corners:
        return
    if polys["exit"] is not (cw.get("area_px") or cw["corners_px"]):
        cw["area_px"] = polys["exit"]  # 네 모서리(homography)는 그대로, 판정 영역만 줄인다
    others.update({n: p for n, p in polys.items() if n != "exit"})
    for poly in corners:
        name = next(f"corner{i}" if i > 1 else "corner" for i in range(1, 100) if (f"corner{i}" if i > 1 else "corner") not in others)
        others[name] = poly
    print(f"안내: 횡단보도 영역끼리 겹친 자리 {len(corners)}곳을 corner 영역으로 뗐다")


def build_site(video: str, points: dict, width_m: float, length_m: float) -> dict:
    """site_example.yaml에 클릭한 점과 실측치만 채운다.
    camera_moved_s·right_turn_signal·판정 기준값은 example 값 그대로 둔다(사람이 영상을 보고 고친다)."""
    with open(EXAMPLE, encoding="utf-8") as f:
        site = yaml.safe_load(f)
    site["site"] = Path(video).stem.split("_")[0]  # A_20261107_0800 → A
    site["crosswalk"] = {"corners_px": points["corners_px"], "area_px": points.get("area_px", []),
                         "width_m": width_m, "length_m": length_m}
    site["other_crosswalks_px"] = {name: points[key] for key, name in OTHER_KEYS.items() if points.get(key)}
    site.update({key: points[key] for key in ONLY.values()
                 if key not in CROSSWALK_KEYS and key not in OTHER_KEYS and key != CURB_KEY})
    apply_curbs(site, points.get(CURB_KEY))
    apply_corners(site)
    check_site(site)  # 모서리·실측치가 잘못되면 저장 전에 에러
    return site


def update_site(site: dict, points: dict, width_m: float | None, length_m: float | None) -> dict:
    """--only: 다시 찍은 영역과 새로 준 실측치만 바꾼다. 나머지는 그대로."""
    for key, pts in points.items():
        if key == CURB_KEY:
            continue  # 횡단보도를 다 바꾼 뒤 합친다
        if key in CROSSWALK_KEYS:
            site["crosswalk"][key] = pts
        elif key in OTHER_KEYS:  # 비우면(s) 그 이름을 지운다
            others = site.setdefault("other_crosswalks_px", {})
            if pts:
                others[OTHER_KEYS[key]] = pts
            else:
                others.pop(OTHER_KEYS[key], None)
        else:
            site[key] = pts
    apply_curbs(site, points.get(CURB_KEY))
    if any(key in points for key in (*CROSSWALK_KEYS, *OTHER_KEYS, CURB_KEY)):  # 횡단보도를 바꿨을 때만
        apply_corners(site)
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
        steps = [s for s in STEPS if s.key in keys]
        others = site.get("other_crosswalks_px") or {}
        existing = {**{k: site["crosswalk"].get(k, []) for k in CROSSWALK_KEYS},
                    **{k: others.get(name, []) for k, name in OTHER_KEYS.items()},
                    **{k: site.get(k, []) for k in ONLY.values() if k not in CROSSWALK_KEYS and k not in OTHER_KEYS}}
        existing = {k: v for k, v in existing.items() if k not in keys}  # 다시 찍을 영역의 예전 모양은 안 보여 준다
        # --only curb: 채운 빈 차도는 이미 있는 횡단보도 영역에 더 합친다(예전에 합친 것을 빼려면 그 횡단보도를 다시 찍는다)
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
