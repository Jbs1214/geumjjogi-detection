# 담당: 장범수
"""분석 리포트 PDF. 판정 표(outputs/judged 모양)의 통계만 담는다(영상·프레임·클립은 넣지 않는다).
matplotlib(이미 쓰는 라이브러리)의 PdfPages로 A4 세로 쪽을 만든다. 글꼴은 맑은 고딕(eval/overlay.py와 같음).
배치·색은 대시보드(dashboard/app.py) 화면을 따른다: 요약 카드 → 단속 우선 검토 시간대 TOP 3 → 표, 남색 제목·빨강 강조.

1쪽 요약: 제목·자료 기간·"위반 의심 통계" 문구, 위반 유형별 요약 카드(판정 대상·위반 의심·위반율과 Wilson 95% 신뢰구간),
  단속 우선 검토 시간대 TOP 3, 판정 구성(양보·미양보·제외 / 준수·경미·미정지·제외·적색 아님), 차종별 표, 읽는 법
다음 쪽(위반 유형마다): 교차로 x 시간대 위반율 표(판정 대상 min_n 미만은 참고용)와 막대그래프(신뢰구간 표시),
  시간대별 판정 구성, 제외 사유, 미양보가 있던 횡단보도(보행자). 제외 사유·횡단보도 칸이 없는 표(DB에서 읽은 것)는 그 항목만 뺀다
위반율·TOP 3 계산은 대시보드와 같은 eval.report.rate_table·wilson_ci·top_slots를 쓴다.
"""
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 화면 없이 파일로만
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle

from eval.report import RED_MODES, apply_red_mode, rate_table, top_slots, wilson_ci

NOTICE = "위반 의심 통계이며 실제 단속은 별도 확인이 필요하다"
KINDS = {  # 표시 이름: (판정 컬럼, 위반 아님, 위반 의심). dashboard/app.py KINDS와 같아야 한다(테스트로 확인)
    "보행자 미양보": ("ped_result", "yield", "no_yield"),
    "적색 미정지": ("red_result", "stopped", "not_stopped"),
}
EXCLUDE = "exclude"
TOP_N = 3  # 단속 우선 검토 시간대 개수(대시보드와 같이 쓴다)
A4 = (8.27, 11.69)  # 인치
LEFT, WIDTH = 0.08, 0.84  # 본문 왼쪽 끝과 너비(쪽 너비 대비)
LABEL_W = 0.16      # 막대·카드 왼쪽 이름 칸 너비
ROW_H = 0.021       # 표 한 줄 높이(쪽 높이 대비)
BAND_H = 0.024      # 막대 한 줄 높이
BULLET_CHARS = 72   # 작은 글 한 줄 글자 수(쪽 너비에 맞춘 값)
ROWS_PER_PAGE = 6   # 위반율 표 한 쪽 줄 수. 이 수까지는 표·막대·판정 구성·제외 사유·판정 기준이 한 쪽에 들어간다
FONT = "Malgun Gothic"
NAVY, RED, GRAY, INK, LINE = "#1f3a68", "#c62828", "#6b7280", "#111827", "#d0d5dd"
PINK = "#fdf1f0"    # TOP 3 카드 바탕(대시보드와 같음)
BLUE, YELLOW, MID, LIGHT = "#2a78d6", "#eda100", "#9aa1ab", "#d3d7dd"
# 판정 구성 막대의 조각: 컬럼 → [(값, 표시 이름, 색)]. 파랑·노랑·빨강은 색각 이상에서도 구분되는 조합이고 판정 대상이 아닌 값은 회색
SEGMENTS = {
    "ped_result": [("yield", "양보", BLUE), ("no_yield", "미양보", RED), (EXCLUDE, "제외", MID)],
    "red_result": [("stopped", "준수", BLUE), ("stopped_minor", "경미", YELLOW), ("not_stopped", "미정지", RED),
                   (EXCLUDE, "제외", MID), ("na", "적색 아님", LIGHT)],
}
REASON_COLUMNS = {"ped_result": "ped_exclude_reason", "red_result": "red_exclude_reason"}
REASONS = {  # 제외 사유(docs/csv_schema.md 5절)
    "camera_moved": "카메라가 움직인 구간", "occluded": "가림 의심", "not_right_turn": "우회전 아님",
    "signal_unknown": "신호 불명", "stopline_hidden": "정지선 가려짐", "zone_unseen": "정지 구간 진입을 못 봄",
    "rturn_signal": "우회전 전용 신호",
}
CROSSWALKS = {"exit": "출구 쪽 횡단보도", "near": "진입 쪽 횡단보도", "diag": "중간 횡단보도", "corner": "횡단보도가 겹친 모퉁이"}
CRITERIA = {  # 위반 유형별 판정 기준 요약(docs/judgment_criteria.md, 기술서 2.2·2.3). 기준값은 적지 않는다(설정 파일이 기준)
    "ped_result": [
        "우회전 차량이 경로상의 횡단보도(진입 쪽·중간·출구 쪽)에 들어서는 순간, 그 횡단보도 위에 보행자가 한 명이라도 있으면 미양보(위반 의심)입니다.",
        "지나는 모든 횡단보도에 보행자가 없으면 양보(위반 아님)입니다. 먼 쪽 차로의 보행자도 셉니다.",
        "보행자가 가려졌을 수 있거나 카메라가 움직인 구간의 우회전은 제외합니다.",
        "화면에 보이는 부분만으로 판정하므로, 카메라에 담기지 않은 곳의 보행자는 세지 못합니다(한계).",
    ],
    "red_result": [
        "우회전 방향의 직진 신호가 빨간불일 때 정지선을 지난 우회전만 봅니다(녹색·황색은 '적색 아님').",
        "정지 구간 안에서 차 앞 끝이 멈추면 정지입니다. 정지선을 넘기 전에 멈추면 준수, 정지선은 넘었지만 횡단보도 전에 멈추면 경미입니다.",
        "정지 구간을 멈추지 않고 지나가면 미정지(위반 의심)입니다.",
        "신호 색을 알 수 없거나 정지선·정지 구간이 화면에서 확인되지 않으면 제외합니다.",
    ],
}


def pct(x: float) -> str:
    return "-" if pd.isna(x) else ("1% 미만" if 0 < x < 0.005 else f"{x:.0%}")


def counts(judged: pd.DataFrame, col: str, ok: str, bad: str) -> tuple[int, int, int]:
    """(판정 대상, 위반 의심, 제외) 건수. 판정 컬럼이 없으면 모두 0."""
    if col not in judged:
        return 0, 0, 0
    return int(judged[col].isin([ok, bad]).sum()), int((judged[col] == bad).sum()), int((judged[col] == EXCLUDE).sum())


def heading(fig, y: float, text: str, note: str = "") -> None:
    fig.text(LEFT, y, text, fontsize=13, color=NAVY, weight="bold")
    if note:
        fig.text(LEFT + WIDTH, y, note, fontsize=8, color=GRAY, ha="right")


def card(fig, x: float, y: float, w: float, h: float, label: str, value: str, note: str = "",
         color: str = INK, accent: bool = False) -> None:
    """카드 한 장: 위에 작은 이름(오른쪽에 보조 글), 아래에 값. accent면 분홍 바탕에 왼쪽 빨간 띠(TOP 3 카드)."""
    face = PINK if accent else "white"
    fig.add_artist(Rectangle((x, y), w, h, transform=fig.transFigure, facecolor=face, edgecolor=face if accent else LINE, linewidth=0.8))
    if accent:
        fig.add_artist(Rectangle((x, y), 0.006, h, transform=fig.transFigure, facecolor=RED, edgecolor="none"))
    pad = 0.018 if accent else 0.012
    fig.text(x + pad, y + h - 0.019, label, fontsize=8.5, color=RED if accent else GRAY, weight="bold" if accent else "normal")
    fig.text(x + pad, y + 0.013, value, fontsize=9.5 if accent else 17, color=color, weight="normal" if accent else "bold")
    if note:
        fig.text(x + w - 0.012, y + h - 0.019, note, fontsize=8.5, color=GRAY, ha="right")


def table(fig, top: float, rows: list[list[str]], columns: list[str], widths: list[float],
          x: float = LEFT, w: float = WIDTH) -> float:
    """top에서 아래로 표를 그리고 표 아래 끝의 y를 돌려준다. 줄 높이가 고정이라 줄 수만큼만 자리를 쓴다."""
    h = (len(rows) + 1) * ROW_H
    ax = fig.add_axes([x, top - h, w, h])
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=columns, colWidths=widths, cellLoc="center", bbox=[0, 0, 1, 1])
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    for (r, _), cell in t.get_celld().items():
        cell.set_edgecolor(LINE)
        if r == 0:
            cell.set_facecolor("#e8eef7")
            cell.set_text_props(weight="bold", color=NAVY)
    return top - h


def legend(fig, y: float, items: list[tuple[str, str]]) -> None:
    """색 네모와 이름을 한 줄에 같은 간격으로 놓는다. items = [(이름, 색)]"""
    for i, (label, color) in enumerate(items):
        x = LEFT + LABEL_W + i * (WIDTH - LABEL_W) / len(items)
        fig.add_artist(Rectangle((x, y), 0.011, 0.0078, transform=fig.transFigure, facecolor=color, edgecolor="none"))
        fig.text(x + 0.016, y - 0.0008, label, fontsize=8.5, color=INK)


def inside_color(color: str) -> str:
    """색 조각 안에 쓰는 글자색: 밝은 조각은 검정에 가깝게, 어두운 조각은 흰색."""
    r, g, b = to_rgb(color)
    return INK if 0.299 * r + 0.587 * g + 0.114 * b > 0.6 else "white"


def composition(fig, y: float, name: str, raw: pd.DataFrame, col: str) -> None:
    """판정 구성 한 줄: 전체를 나눈 누적 막대(조각이 넓으면 비율을 안에 적음)와 아래 범례(이름·건수)."""
    parts = [(label, int((raw[col] == value).sum()), color) for value, label, color in SEGMENTS[col]] if col in raw else []
    total = sum(n for _, n, _ in parts)
    fig.text(LEFT, y + 0.006, name, fontsize=10.5, color=INK, weight="bold")
    if not total:
        fig.text(LEFT + LABEL_W, y + 0.006, "판정 결과 없음", fontsize=9, color=GRAY)
        return
    ax = fig.add_axes([LEFT + LABEL_W, y, WIDTH - LABEL_W, 0.022])
    ax.axis("off")
    ax.set_xlim(0, total)
    ax.set_ylim(0, 1)
    left = 0
    for _, n, color in parts:
        if n:
            ax.barh(0.5, n, left=left, height=1, color=color, edgecolor="white", linewidth=2)  # 흰 테두리 = 조각 사이 틈
            if n / total >= 0.07:  # 글자가 들어갈 만큼 넓은 조각만
                ax.text(left + total * 0.008, 0.48, f"{n / total:.0%}", fontsize=8.5, color=inside_color(color), va="center", weight="bold")
            left += n
    legend(fig, y - 0.016, [(f"{label} {n}건", color) for label, n, color in parts])


def bullets(fig, y: float, lines: list[str]) -> float:
    """작은 회색 글 여러 줄(쪽 너비를 넘는 줄은 접는다). 아래 끝 y를 돌려준다."""
    for line in lines:
        for i, part in enumerate(textwrap.wrap(line, BULLET_CHARS)):
            fig.text(LEFT, y, f"· {part}" if i == 0 else f"  {part}", fontsize=8.5, color=GRAY, va="top")
            y -= 0.017
    return y


def kind_line(fig, y: float, name: str, note: str = "") -> None:
    """카드 줄 위의 위반 유형 이름(오른쪽에 작은 보조 글)."""
    fig.text(LEFT, y, name, fontsize=10.5, color=INK, weight="bold")
    if note:
        fig.text(LEFT + WIDTH, y, note, fontsize=8.5, color=GRAY, ha="right")


def notes(fig, y: float, min_n: int, red_mode: str) -> None:
    minor = "위반 의심으로" if RED_MODES[red_mode] == "not_stopped" else "정지로"
    bullets(fig, y, [
        f"위반율 = 위반 의심 / 판정 대상(제외 뺀 것). 신뢰구간은 Wilson 95%. 판정 대상 {min_n}건 미만인 칸은 '참고용'.",
        "단속 우선 검토 시간대는 표본이 충분한(참고용이 아닌) 시간대 중 위반율이 높은 순서입니다.",
        "적색 미정지는 우회전 방향 신호가 빨간불일 때만 봅니다. 경미 정지 = 정지선은 넘었지만 횡단보도 전에 선 경우.",
        f"적색 판정 모드: {red_mode} (경미 정지를 {minor} 봅니다).",
        "자동 판정 결과(사람 확인 전)입니다. 판정 기준은 다음 쪽부터 위반 유형별로 적었습니다.",
    ])


def vehicle_rows(judged: pd.DataFrame) -> list[list[str]]:
    """차종마다 우회전 건수와 위반 유형별 '위반 의심 / 판정 대상', 위반율. 차종 칸이 없으면 빈 목록."""
    if "vehicle_type" not in judged or judged.empty:
        return []
    rows = []
    for name, g in sorted(judged.groupby("vehicle_type"), key=lambda item: -len(item[1])):
        cells = [str(name), f"{len(g)}건"]
        for col, ok, bad in KINDS.values():
            n, k, _ = counts(g, col, ok, bad)
            cells += [f"{k} / {n}건" if n else "-", pct(k / n) if n else "-"]
        rows.append(cells)
    return rows


def cover(pdf: PdfPages, judged: pd.DataFrame, raw: pd.DataFrame, title: str, min_n: int, videos: dict | None, red_mode: str) -> None:
    fig = plt.figure(figsize=A4)
    wall = pd.to_datetime(judged["wall_time"])
    videos = videos or dict(zip(judged["video"], judged["intersection"]))  # 이벤트 0건인 영상도 세려면 넘긴다
    sites = ", ".join(sorted({str(s) for s in videos.values()})) or "-"
    period = f"{wall.min():%Y-%m-%d %H:%M} ~ {wall.max():%Y-%m-%d %H:%M}" if len(judged) else "- (우회전 이벤트 없음)"
    fig.text(LEFT, 0.950, title, fontsize=20, color=NAVY, weight="bold")
    fig.text(LEFT + WIDTH, 0.955, f"만든 시각 {datetime.now():%Y-%m-%d %H:%M}", fontsize=8.5, color=GRAY, ha="right")
    fig.text(LEFT, 0.927, f"자료 기간  {period}   ·   영상 {len(videos)}개   ·   우회전 이벤트 {len(judged)}건", fontsize=10, color=INK)
    fig.text(LEFT, 0.910, f"교차로  {textwrap.shorten(sites, 80, placeholder=' …')}", fontsize=10, color=INK)
    fig.add_artist(Rectangle((LEFT, 0.870), WIDTH, 0.028, transform=fig.transFigure, facecolor=PINK, edgecolor=RED, linewidth=0.8))
    fig.text(LEFT + 0.012, 0.884, f"※ {NOTICE}", fontsize=10.5, color=RED, weight="bold", va="center")

    heading(fig, 0.845, "요약")
    gap = 0.012
    w = (WIDTH - 2 * gap) / 3
    y = 0.822
    for name, (col, ok, bad) in KINDS.items():
        n, k, excluded = counts(judged, col, ok, bad)
        low, high = wilson_ci(k, n)
        kind_line(fig, y, name, f"제외 {excluded}건" if n else "판정 결과 없음")
        card(fig, LEFT, y - 0.064, w, 0.056, "판정 대상 우회전", f"{n}건" if n else "-")
        card(fig, LEFT + w + gap, y - 0.064, w, 0.056, "위반 의심", f"{k}건" if n else "-", color=RED)
        card(fig, LEFT + 2 * (w + gap), y - 0.064, w, 0.056, "위반율", pct(k / n) if n else "-",
             f"95% 신뢰구간 {low:.0%}~{high:.0%}" if n else "", color=RED)
        y -= 0.086

    heading(fig, 0.640, f"단속 우선 검토 시간대 TOP {TOP_N}", "표본이 충분한 시간대 중 위반율 높은 순")
    y = 0.617
    for name, (col, ok, bad) in KINDS.items():
        kind_line(fig, y, name)
        top = top_slots(rate_table(judged, min_n, col, ok, bad), TOP_N) if counts(judged, col, ok, bad)[0] else pd.DataFrame()
        if top.empty:
            fig.text(LEFT, y - 0.034, f"판정 대상이 {min_n}건 이상이면서 위반 의심이 있는 시간대가 아직 없습니다.", fontsize=9, color=GRAY)
        for i, r in enumerate(top.itertuples()):
            card(fig, LEFT + i * (w + gap), y - 0.056, w, 0.048, f"{i + 1}. 교차로 {r.intersection} {r.slot}",
                 f"위반율 {pct(r.violation_rate)} · 위반 의심 {getattr(r, f'n_{bad}')}건", accent=True)
        y -= 0.076

    heading(fig, 0.456, "판정 구성", "우회전 이벤트 전체를 판정 결과로 나눈 것")
    composition(fig, 0.414, "보행자", raw, "ped_result")
    composition(fig, 0.362, "적색", raw, "red_result")

    y = 0.314
    rows = vehicle_rows(judged)
    if rows:
        heading(fig, y, "차종별", "위반 의심 / 판정 대상")
        y = table(fig, y - 0.014, rows, ["차종", "우회전", "보행자 미양보", "위반율", "적색 미정지", "위반율"],
                  [0.16, 0.14, 0.2, 0.15, 0.2, 0.15]) - 0.040
    heading(fig, y, "읽는 법")
    notes(fig, y - 0.014, min_n, red_mode)
    pdf.savefig(fig)
    plt.close(fig)


def band_axes(fig, top: float, chunk: pd.DataFrame):
    """교차로·시간대 한 줄에 가로 막대 하나를 놓는 축과 그 아래 끝 y. 줄 높이가 고정이라 줄 수만큼만 자리를 쓴다."""
    labels = (chunk["intersection"].astype(str) + " " + chunk["slot"]).tolist()
    h = len(labels) * BAND_H
    ax = fig.add_axes([LEFT + LABEL_W, top - h, WIDTH - LABEL_W - 0.07, h])
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(length=0, labelsize=8.5)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.grid(axis="x", color="#eceff3", linewidth=0.8)
    ax.set_axisbelow(True)
    return ax, top - h - 0.022  # 눈금 글자 자리까지


def rate_chart(fig, top: float, chunk: pd.DataFrame) -> float:
    """위반율 막대(선 = 95% 신뢰구간, 옅은 막대 = 참고용). 신뢰구간 선 끝 옆에 값을 적는다. 아래 끝 y를 돌려준다."""
    ax, bottom = band_axes(fig, top, chunk)
    rate = chunk["violation_rate"].fillna(0).to_numpy() * 100
    low = (chunk["violation_rate"] - chunk["ci_low"]).fillna(0).to_numpy() * 100
    high = (chunk["ci_high"] - chunk["violation_rate"]).fillna(0).to_numpy() * 100
    colors = [RED if note == "" else "#f2a7a7" for note in chunk["note"]]
    ax.barh(range(len(chunk)), rate, height=0.5, color=colors, xerr=[low, high], error_kw={"ecolor": GRAY, "elinewidth": 1, "capsize": 2})
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    for i, (r, h, value) in enumerate(zip(rate, high, chunk["violation_rate"])):
        ax.text(min(r + h, 100) + 1.5, i, pct(value), fontsize=8.5, color=INK, va="center", clip_on=False)
    return bottom


def composition_chart(fig, top: float, chunk: pd.DataFrame, raw: pd.DataFrame, col: str) -> float:
    """시간대별 판정 구성(건수 누적 막대, 끝에 합계). 아래 끝 y를 돌려준다."""
    legend(fig, top, [(label, color) for _, label, color in SEGMENTS[col]])
    ax, bottom = band_axes(fig, top - 0.014, chunk)
    n_by = raw.groupby(["intersection", "slot"])[col].value_counts()
    totals = []
    for i, r in enumerate(chunk.itertuples()):
        left = 0
        for value, _, color in SEGMENTS[col]:
            n = int(n_by.get((r.intersection, r.slot, value), 0))
            if n:
                ax.barh(i, n, left=left, height=0.5, color=color, edgecolor="white", linewidth=1.5)
            left += n
        totals.append(left)
    longest = max(totals + [1])
    ax.set_xlim(0, longest)
    for i, total in enumerate(totals):
        ax.text(total + longest * 0.015, i, f"{total}건", fontsize=8.5, color=INK, va="center", clip_on=False)
    return bottom


def reason_rows(raw: pd.DataFrame, col: str) -> list[list[str]]:
    """제외 사유별 건수(많은 순). 사유 칸이 없는 표(DB에서 읽은 것)면 빈 목록."""
    column = REASON_COLUMNS[col]
    if column not in raw:
        return []
    reasons = raw.loc[raw[col] == EXCLUDE, column].fillna("사유 없음").map(lambda v: REASONS.get(v, v))
    return [[name, f"{n}건"] for name, n in reasons.value_counts().items()]


def crosswalk_rows(raw: pd.DataFrame) -> list[list[str]]:
    """미양보 이벤트에서 보행자가 있던 횡단보도별 건수(한 이벤트에 여러 곳이면 각각 센다)."""
    if "no_yield_at" not in raw:
        return []
    names = raw.loc[raw["ped_result"] == "no_yield", "no_yield_at"].dropna().astype(str).str.split(";").explode()
    return [[CROSSWALKS.get(name, name), f"{n}건"] for name, n in names.value_counts().items()]


def detail_blocks(raw: pd.DataFrame, col: str) -> list[tuple[str, list[str], list[list[str]]]]:
    """표 아래에 붙일 작은 표들 [(제목, 칸 이름, 줄)]: 제외 사유와, 보행자면 미양보가 있던 횡단보도. 줄이 없는 것은 뺀다."""
    blocks = [("제외 사유", ["사유", "건수"], reason_rows(raw, col))]
    if col == "ped_result":
        blocks.append(("미양보 때 보행자가 있던 횡단보도", ["횡단보도", "건수"], crosswalk_rows(raw)))
    return [b for b in blocks if b[2]]


def details(fig, top: float, blocks: list, col: str) -> None:
    """작은 표들을 왼쪽·오른쪽 반씩 나란히 그리고, 그 아래에 판정 기준 요약을 적는다."""
    half = (WIDTH - 0.04) / 2
    bottom = top
    for i, (name, columns, rows) in enumerate(blocks):
        x = LEFT + i * (half + 0.04)
        fig.text(x, top, name, fontsize=13, color=NAVY, weight="bold")
        bottom = min(bottom, table(fig, top - 0.014, rows, columns, [0.68, 0.32], x, half) - 0.040)
    heading(fig, bottom, "판정 기준")
    bullets(fig, bottom - 0.014, CRITERIA[col])


def kind_pages(pdf: PdfPages, judged: pd.DataFrame, raw: pd.DataFrame, name: str, min_n: int) -> None:
    col, ok, bad = KINDS[name]
    if not counts(judged, col, ok, bad)[0]:
        return
    t = rate_table(judged, min_n, col, ok, bad).sort_values(["intersection", "slot"])
    cells = [[str(r.intersection), r.slot, f"{r.n_target}", f"{getattr(r, f'n_{bad}')}", pct(r.violation_rate),
              "-" if pd.isna(r.ci_low) else f"{r.ci_low:.0%}~{r.ci_high:.0%}", r.note or "표본 충분"] for r in t.itertuples()]
    blocks = detail_blocks(raw, col)
    rows = max((len(b[2]) + 1 for b in blocks), default=0)
    need = rows * ROW_H + (0.054 if blocks else 0) + 0.014 + len(CRITERIA[col]) * 0.017  # 작은 표와 판정 기준이 쓰는 높이
    done = False
    starts = list(range(0, len(t), ROWS_PER_PAGE))
    for start in starts:
        chunk = t.iloc[start:start + ROWS_PER_PAGE]
        fig = plt.figure(figsize=A4)
        fig.text(LEFT, 0.945, f"{name} · 교차로 x 시간대", fontsize=16, color=NAVY, weight="bold")
        fig.text(LEFT, 0.922, f"※ {NOTICE}", fontsize=9, color=RED)
        y = table(fig, 0.895, cells[start:start + ROWS_PER_PAGE],
                  ["교차로", "시간대", "판정 대상", "위반 의심", "위반율", "95% 신뢰구간", "신뢰도"], [0.17, 0.14, 0.12, 0.12, 0.12, 0.17, 0.16])
        heading(fig, y - 0.042, "시간대별 위반율", "선 = 95% 신뢰구간 · 옅은 막대 = 참고용")
        y = rate_chart(fig, y - 0.056, chunk)
        heading(fig, y - 0.036, "시간대별 판정 구성", "우회전 이벤트 건수")
        y = composition_chart(fig, y - 0.060, chunk, raw, col)
        if start == starts[-1] and y - 0.036 - need > 0.04:  # 마지막 쪽에 자리가 있으면 제외 사유·판정 기준까지
            details(fig, y - 0.036, blocks, col)
            done = True
        pdf.savefig(fig)
        plt.close(fig)
    if not done:  # 자리가 없었으면 따로 한 쪽
        fig = plt.figure(figsize=A4)
        fig.text(LEFT, 0.945, f"{name} · 제외 사유와 판정 기준", fontsize=16, color=NAVY, weight="bold")
        details(fig, 0.895, blocks, col)
        pdf.savefig(fig)
        plt.close(fig)


def build_report(judged: pd.DataFrame, out: Path, min_n: int, title: str = "우회전 위반 의심 분석 리포트",
                 videos: dict | None = None, red_mode: str = "보통") -> Path:
    """판정 표로 리포트 PDF를 만들고 경로를 돌려준다. videos = {영상 이름: 교차로}(이벤트가 0건인 영상까지 세려면 준다).
    red_mode = 적색 판정 모드(기술서 3.1, 기본 보통). 판정 구성과 제외 사유는 모드를 적용하기 전 값(경미 정지를 따로)으로 보여 준다."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw, judged = judged, apply_red_mode(judged, red_mode)
    with plt.rc_context({"font.family": FONT, "axes.unicode_minus": False}), PdfPages(out) as pdf:
        cover(pdf, judged, raw, title, min_n, videos, red_mode)
        for name in KINDS:
            kind_pages(pdf, judged, raw, name, min_n)
    return out
