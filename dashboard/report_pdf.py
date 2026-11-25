# 담당: 장범수
"""분석 보고서 PDF. 판정 표(outputs/judged 모양)의 통계만 담는다(영상·프레임·클립은 넣지 않는다).
matplotlib(이미 쓰는 라이브러리)의 PdfPages로 A4 세로 쪽을 만든다. 모양은 공문서 보고서 양식을 따른다:
제목 괘선 → 요지 상자 → Ⅰ·Ⅱ… 장 제목 → □·○ 개조식 문장과 괘선 표(흑백 위주), 쪽 아래 "- 1 -", 맨 끝 "끝."

요지 상자: 종합 단속 우선 검토 대상(어느 교차로를 몇 시에)과 유형별 위반 의심 건수
Ⅰ 분석 개요: 목적·기간·대상·자료·방법
Ⅱ 분석 결과 총괄: 종합 단속 우선 검토 시간대(기술서 3.1), 위반 유형별 현황(판정 대상·위반 의심·위반율과 Wilson 95% 신뢰구간), 유형별 상위 3개, 차종별 현황
Ⅲ 판정 기준(판정 결과가 있는 유형만)
Ⅳ~ 위반 유형별 세부 현황: 교차로·시간대별 위반율 표(판정 대상 min_n 미만은 참고용)와 막대그래프(신뢰구간 표시),
  판정 결과별 건수, 제외 사유, 미양보가 있던 횡단보도(보행자). 제외 사유·횡단보도 칸이 없는 표(DB에서 읽은 것)는 그 항목만 뺀다
마지막 장 자료 이용 시 유의사항
위반율·상위 시간대 계산은 대시보드와 같은 eval.report.rate_table·wilson_ci·top_slots를 쓴다.
"""
import logging
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 화면 없이 파일로만
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from eval.report import RED_MODES, apply_red_mode, rate_table, top_slots, wilson_ci

logging.getLogger("fontTools").setLevel(logging.ERROR)  # PDF에 글꼴을 넣을 때 나오는 안내가 작업 로그에 쌓이지 않게

NOTICE = "위반 의심 통계이며 실제 단속은 별도 확인이 필요하다"
KINDS = {  # 표시 이름: (판정 컬럼, 위반 아님, 위반 의심). dashboard/app.py KINDS와 같아야 한다(테스트로 확인)
    "보행자 미양보": ("ped_result", "yield", "no_yield"),
    "적색 미정지": ("red_result", "stopped", "not_stopped"),
}
EXCLUDE = "exclude"
TOP_N = 3  # 단속 우선 검토 시간대 개수(대시보드와 같이 쓴다)

# 쪽 배치(단위 mm, 위에서 아래로)
PAGE_W, PAGE_H = 210, 297
LEFT, WIDTH, TOP, BOTTOM = 20, 170, 20, 280
ROW_H = 6.6         # 표 한 줄 높이
LINE_H = 6.6        # 본문 한 줄 높이
BAND_H = 5.2        # 막대 한 줄 높이
CHART_LABEL_W = 34  # 막대 왼쪽 이름 칸 너비
MAX_BANDS = 25      # 그래프 하나에 넣는 막대 수(넘으면 나눠 그린다)
NAVY, INK, GRAY, HEAD, TOTAL, LINE, BAR = "#1f3864", "#000000", "#595959", "#e7e6e6", "#f5f5f5", "#7f7f7f", "#44546a"
# 글꼴 후보: 앞에서부터 이 PC에 있는 것을 쓴다(없으면 맑은 고딕)
FONTS = {"body": ["HCR Batang", "Batang", "Malgun Gothic"], "title": ["HYHeadLine-Medium", "Malgun Gothic"], "table": ["Malgun Gothic"]}

SEGMENTS = {  # 판정 결과별 건수 표의 칸: 컬럼 → [(값, 표시 이름)]
    "ped_result": [("yield", "양보"), ("no_yield", "미양보"), (EXCLUDE, "제외")],
    "red_result": [("stopped", "준수"), ("stopped_minor", "경미"), ("not_stopped", "미정지"), (EXCLUDE, "제외"), ("na", "적색 아님")],
}
REASON_COLUMNS = {"ped_result": "ped_exclude_reason", "red_result": "red_exclude_reason"}
REASONS = {  # 제외 사유(docs/csv_schema.md 5절)
    "camera_moved": "카메라가 움직인 구간", "occluded": "가림 의심", "not_right_turn": "우회전 아님",
    "signal_unknown": "신호 불명", "stopline_hidden": "정지선 가려짐", "zone_unseen": "정지 구간 진입을 못 봄",
    "rturn_signal": "우회전 전용 신호",
}
CROSSWALKS = {"exit": "출구 쪽 횡단보도", "near": "진입 쪽 횡단보도", "diag": "중간 횡단보도", "corner": "횡단보도가 겹친 모퉁이"}
VEHICLES = {"차": "승용·승합차"}  # 표시 이름만 바꾼다(car는 승합차를 포함)
CRITERIA = {  # 위반 유형별 판정 기준 요약(docs/judgment_criteria.md, 기술서 2.2·2.3). 기준값은 적지 않는다(설정 파일이 기준)
    "ped_result": [
        "(미양보) 우회전 차량이 경로상 횡단보도(진입 쪽·중간·출구 쪽)에 진입하는 순간 해당 횡단보도에 보행자가 1명 이상 있는 경우",
        "(양보) 통과한 모든 횡단보도에 보행자가 없는 경우(먼 쪽 차로의 보행자도 포함하여 판단)",
        "(제외) 보행자가 가려졌을 가능성이 있거나 카메라가 움직인 구간의 우회전",
        "(한계) 화면에 보이는 범위만으로 판정하므로 촬영 범위 밖의 보행자는 반영되지 않음",
    ],
    "red_result": [
        "(대상) 우회전 방향 직진 신호가 적색일 때 정지선을 통과한 우회전(녹색·황색은 '적색 아님'으로 분류)",
        "(준수·경미) 정지 구간 안에서 차량 앞 끝이 멈춘 경우. 정지선을 넘기 전 정지는 준수, 정지선을 넘었으나 횡단보도 전 정지는 경미",
        "(미정지) 정지 구간을 멈추지 않고 통과한 경우",
        "(제외) 신호 색을 알 수 없거나 정지선·정지 구간이 화면에서 확인되지 않는 경우",
    ],
}


def pct(x: float) -> str:
    return "-" if pd.isna(x) else f"{x:.1%}"


def ci(low: float, high: float) -> str:
    return "-" if pd.isna(low) else f"{low:.1%}~{high:.1%}"


def counts(judged: pd.DataFrame, col: str, ok: str, bad: str) -> tuple[int, int, int]:
    """(판정 대상, 위반 의심, 제외) 건수. 판정 컬럼이 없으면 모두 0."""
    if col not in judged:
        return 0, 0, 0
    return int(judged[col].isin([ok, bad]).sum()), int((judged[col] == bad).sum()), int((judged[col] == EXCLUDE).sum())


def table_h(n_rows: int) -> float:
    """머리글까지 합친 표 높이(mm). 한 쪽을 넘는 표는 어차피 나뉘므로 첫 두 줄 높이만."""
    h = (n_rows + 1) * ROW_H
    return h if h <= BOTTOM - TOP - 20 else 2 * ROW_H


def korean_date(t: datetime) -> str:
    return f"{t.year}. {t.month}. {t.day}.({'월화수목금토일'[t.weekday()]})"


def period_text(wall: pd.Series) -> str:
    """공문서 날짜 표기. 하루 안이면 날짜를 한 번만 적는다."""
    if wall.empty:
        return "해당 없음(우회전 차량 없음)"
    a, b = wall.min(), wall.max()
    end = f"{b:%H:%M}" if a.date() == b.date() else f"{korean_date(b)} {b:%H:%M}"
    return f"{korean_date(a)} {a:%H:%M} ~ {end}"


def overall_slots(judged: pd.DataFrame, min_n: int) -> pd.DataFrame:
    """종합 단속 우선 검토 시간대(기술서 3.1): 두 유형 모두 표본이 충분한 칸에서 유형별 위반율 순위의 합이 작은 순 TOP_N개.
    컬럼: intersection, slot, 판정 컬럼마다 rate_·n_(위반 의심 건수)·rank_, rank_sum. 한 유형이라도 판정이 없으면 빈 표."""
    if any(not counts(judged, *kind)[0] for kind in KINDS.values()):
        return pd.DataFrame()
    merged = None
    for col, ok, bad in KINDS.values():
        t = rate_table(judged, min_n, col, ok, bad)
        t = t[t["note"] == ""].rename(columns={"violation_rate": f"rate_{col}", f"n_{bad}": f"n_{col}"})
        t = t[["intersection", "slot", f"rate_{col}", f"n_{col}"]]
        merged = t if merged is None else merged.merge(t, on=["intersection", "slot"])
    cols = [col for col, _, _ in KINDS.values()]
    merged = merged[merged[[f"n_{c}" for c in cols]].sum(axis=1) > 0].copy()
    for c in cols:
        merged[f"rank_{c}"] = merged[f"rate_{c}"].rank(method="min", ascending=False).astype(int)
    merged["rank_sum"] = merged[[f"rank_{c}" for c in cols]].sum(axis=1)
    return merged.sort_values(["rank_sum", "rate_ped_result"], ascending=[True, False]).head(TOP_N)


class Document:
    """A4 쪽을 위에서 아래로 채운다. 자리가 모자라면 다음 쪽으로 넘긴다. 좌표는 mm."""

    def __init__(self, pdf: PdfPages):
        installed = {f.name for f in font_manager.fontManager.ttflist}
        self.fonts = {k: next((n for n in names if n in installed), names[-1]) for k, names in FONTS.items()}
        self.pdf, self.page, self.chapter = pdf, 0, 0
        self.new_page()

    def new_page(self) -> None:
        self.close()
        self.fig = plt.figure(figsize=(PAGE_W / 25.4, PAGE_H / 25.4))
        self.page += 1
        self.y = TOP
        self.text(PAGE_W / 2, 286, f"- {self.page} -", 10, ha="center")

    def close(self) -> None:
        if self.page:
            self.pdf.savefig(self.fig)
            plt.close(self.fig)

    def need(self, h: float) -> None:
        if self.y + h > BOTTOM:
            self.new_page()

    def text(self, x: float, y: float, s: str, size: float, font: str = "body", **kw):
        style = {"color": INK, "va": "center", **kw}
        return self.fig.text(x / PAGE_W, 1 - y / PAGE_H, s, fontsize=size, family=self.fonts[font], **style)

    def line(self, x1: float, y1: float, x2: float, y2: float, width: float, color: str = INK) -> None:
        self.fig.add_artist(Line2D([x1 / PAGE_W, x2 / PAGE_W], [1 - y1 / PAGE_H, 1 - y2 / PAGE_H], linewidth=width, color=color))

    def box(self, x: float, y: float, w: float, h: float, face: str, edge: str = "none", width: float = 0) -> None:
        self.fig.add_artist(Rectangle((x / PAGE_W, 1 - (y + h) / PAGE_H), w / PAGE_W, h / PAGE_H, transform=self.fig.transFigure,
                                      facecolor=face, edgecolor=edge, linewidth=width))

    def width_of(self, s: str, size: float, font: str = "body", **kw) -> float:
        t = self.text(0, 0, s, size, font, **kw)
        w = t.get_window_extent(self.fig.canvas.get_renderer()).width / self.fig.dpi * 25.4
        t.remove()
        return w

    def wrap(self, s: str, size: float, width: float) -> list[str]:
        """낱말 단위로 접는다. 한 낱말이 너비보다 길면(공백 없는 교차로 목록 등) 글자 단위로 자른다."""
        lines, current = [], ""
        for word in s.split(" "):
            candidate = f"{current} {word}" if current else word
            if self.width_of(candidate, size) <= width:
                current = candidate
                continue
            if current:
                lines.append(current)
            while self.width_of(word, size) > width:
                cut = len(word) - 1
                while self.width_of(word[:cut], size) > width:
                    cut -= 1
                lines.append(word[:cut])
                word = word[cut:]
            current = word
        return lines + [current]

    def paragraph(self, mark: str, s: str, indent: float, size: float = 11.5) -> None:
        """기호(□·○)와 문장. 둘째 줄부터는 글자 시작 위치에 맞춘다."""
        x = LEFT + indent
        tx = x + self.width_of(mark, size) + 2
        lines = self.wrap(s, size, LEFT + WIDTH - tx)
        self.need(len(lines) * LINE_H)
        self.text(x, self.y + LINE_H / 2, mark, size)
        for part in lines:
            self.text(tx, self.y + LINE_H / 2, part, size)
            self.y += LINE_H
        self.y += 0.6

    def heading(self, s: str, keep: float = 30) -> None:
        """장 제목: 남색 칸의 로마 숫자 + 제목 + 밑줄. keep = 제목과 같은 쪽에 있어야 할 높이."""
        self.chapter += 1
        if self.y > TOP:
            self.y += 5
        self.need(12 + keep)
        self.box(LEFT, self.y, 10, 8.5, NAVY)
        self.text(LEFT + 5, self.y + 4.4, "ⅠⅡⅢⅣⅤⅥ"[self.chapter - 1], 13, "table", color="white", ha="center", weight="bold")
        self.text(LEFT + 13, self.y + 4.4, s, 15, "title")
        self.line(LEFT, self.y + 8.5, LEFT + WIDTH, self.y + 8.5, 1.2, NAVY)
        self.y += 12

    def item(self, s: str, unit: str = "", keep: float = 2 * ROW_H, x: float = LEFT) -> None:
        """□ 항목 제목(오른쪽에 단위). keep = 제목과 같은 쪽에 있어야 할 높이."""
        self.need(8 + keep)
        self.text(x, self.y + 3.5, f"□ {s}", 12, weight="bold")
        if unit:
            self.text(LEFT + WIDTH, self.y + 4.2, f"({unit})", 9, ha="right", color=GRAY)
        self.y += 8

    def cell(self, x: float, y: float, w: float, h: float, s: str, weight: str = "normal") -> None:
        """표 칸 가운데에 쓴다. 칸보다 긴 글(교차로 이름 등)은 글자를 줄인다."""
        size = 9.5
        used = self.width_of(s, size, "table", weight=weight) if s else 0
        self.text(x + w / 2, y + h / 2, s, size * min(1, (w - 1.5) / used) if used else size, "table", ha="center", weight=weight)

    def table(self, columns: list[str], rows: list[list[str]], widths: list[float], total: bool = False, merge: bool = False,
              x: float = LEFT, w: float = WIDTH) -> None:
        """괘선 표. total이면 첫 줄이 합계(옅은 바탕), merge면 첫 칸의 같은 값이 이어지는 줄을 한 칸으로 묶는다.
        쪽을 넘으면 다음 쪽에 머리글을 다시 그린다."""
        edges = [x]
        for part in widths:
            edges.append(edges[-1] + w * part / sum(widths))
        right = x + w
        start = 0
        while start < len(rows):
            self.need(2 * ROW_H)
            chunk = rows[start:start + int((BOTTOM - self.y) // ROW_H) - 1]
            top, bottom = self.y, self.y + (len(chunk) + 1) * ROW_H
            self.box(x, top, w, ROW_H, HEAD)
            for c, name in enumerate(columns):
                self.cell(edges[c], top, edges[c + 1] - edges[c], ROW_H, name, "bold")
            for i, row in enumerate(chunk):
                y = top + (i + 1) * ROW_H
                is_total = total and start + i == 0
                if is_total:
                    self.box(x, y, w, ROW_H, TOTAL)
                joined = is_total and row[1] == ""  # 합계 줄은 이름 칸 둘(교차로·시간대)을 한 칸으로
                same = merge and i > 0 and chunk[i - 1][0] == row[0]  # 위 줄과 묶이는 줄
                span = 1
                while merge and i + span < len(chunk) and chunk[i + span][0] == row[0]:
                    span += 1
                for c, value in enumerate(row):
                    if not (c == 0 and same):
                        end = edges[2] if c == 0 and joined else edges[c + 1]
                        self.cell(edges[c], y, end - edges[c], ROW_H * (span if c == 0 else 1), value, "bold" if is_total else "normal")
                for edge in edges[2 if joined else 1:-1]:
                    self.line(edge, y, edge, y + ROW_H, 0.4, LINE)
                if i:
                    self.line(edges[1] if same else x, y, right, y, 0.4, LINE)
            for edge in edges[1:-1]:
                self.line(edge, top, edge, top + ROW_H, 0.4, LINE)
            self.line(x, top + ROW_H, right, top + ROW_H, 0.8)
            for y in (top, bottom):
                self.line(x, y, right, y, 1.2)
            self.y = bottom
            start += len(chunk)
            if start < len(rows):
                self.new_page()
        self.y += 5


def vehicle_rows(judged: pd.DataFrame) -> list[list[str]]:
    """합계와 차종마다 우회전 건수, 위반 유형별 판정 대상·위반 의심·위반율. 차종 칸이 없으면 빈 목록."""
    if "vehicle_type" not in judged or judged.empty:
        return []
    groups = sorted(judged.groupby("vehicle_type"), key=lambda item: -len(item[1]))
    rows = []
    for name, g in [("계", judged), *groups]:
        cells = [VEHICLES.get(name, str(name)), f"{len(g):,}"]
        for col, ok, bad in KINDS.values():
            n, k, _ = counts(g, col, ok, bad)
            cells += [f"{n:,}", f"{k:,}", pct(k / n)] if n else ["-", "-", "-"]
        rows.append(cells)
    return rows


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
    """세부 현황에 붙일 작은 표들 [(제목, 칸 이름, 줄)]: 제외 사유와, 보행자면 미양보가 있던 횡단보도. 줄이 없는 것은 뺀다."""
    blocks = [("제외 사유", ["사유", "건수"], reason_rows(raw, col))]
    if col == "ped_result":
        blocks.append(("미양보 시 보행자 위치(중복 포함)", ["횡단보도", "건수"], crosswalk_rows(raw)))
    return [b for b in blocks if b[2]]


def title_block(doc: Document, judged: pd.DataFrame, title: str, min_n: int) -> None:
    """제목(위아래 겹줄) → 작성일 → 요지 상자."""
    for dy, width in ((0, 2.4), (1.3, 0.6), (20.7, 0.6), (22, 2.4)):
        doc.line(LEFT, doc.y + dy, LEFT + WIDTH, doc.y + dy, width, NAVY)
    doc.text(PAGE_W / 2, doc.y + 11, title, 22, "title", ha="center")
    doc.text(LEFT + WIDTH, doc.y + 27.5, korean_date(datetime.now()), 11, ha="right")
    doc.y += 33

    places = [f"{r.intersection} {r.slot}" for r in overall_slots(judged, min_n).itertuples()]
    head = f"종합 단속 우선 검토 1순위: {places[0]}" if places else f"우회전 차량 {len(judged):,}건을 분석한 결과, 위반 의심 현황은 다음과 같음"
    lines = [head]
    if places[1:]:
        lines.append("  - " + ", ".join(f"{i}순위 {p}" for i, p in enumerate(places[1:], 2)))
    for name, (col, ok, bad) in KINDS.items():
        n, k, _ = counts(judged, col, ok, bad)
        lines.append(f"  - {name} 의심 {k:,}건(판정 대상 {n:,}건의 {pct(k / n)})" if n else f"  - {name}: 판정 결과 없음")
    lines = [part for s in lines for part in doc.wrap(s, 11.5, WIDTH - 14)]
    h = len(lines) * LINE_H + 13
    doc.box(LEFT, doc.y, WIDTH, h, "white", INK, 0.8)
    for i, s in enumerate(lines):
        doc.text(LEFT + 5, doc.y + 3 + (i + 0.5) * LINE_H, s, 11.5, weight="bold" if i == 0 else "normal")
    doc.text(LEFT + 5, doc.y + h - 5, f"※ {NOTICE}", 10.5, weight="bold")
    doc.y += h + 2


def overview(doc: Document, judged: pd.DataFrame, videos: dict, red_mode: str) -> None:
    doc.heading("분석 개요", keep=8 * LINE_H)
    sites = ", ".join(sorted({str(s) for s in videos.values()})) or "-"
    minor = "위반 의심으로" if RED_MODES[red_mode] == "not_stopped" else "정지로"
    for name, s in [
        ("분석 목적", "우회전 차량의 보행자 미양보 및 적색신호 미정지 의심 현황을 파악하여 단속 우선 검토 시간대 선정에 활용"),
        ("분석 기간", period_text(pd.to_datetime(judged["wall_time"]))),
        ("분석 대상", f"교차로 {len(set(videos.values()))}개소({sites}), 우회전 차량 {len(judged):,}건"),
        ("분석 자료", f"고정 카메라 촬영 영상 {len(videos)}개"),
        ("분석 방법", "영상 자동 검출 후 판정 기준에 따라 자동 판정(사람 확인 전 결과)"),
        ("적색 판정", f"{red_mode} 모드 적용(경미 정지를 {minor} 집계)"),
    ]:
        doc.paragraph("□", f"({name}) {s}", 0)


def criteria(doc: Document, judged: pd.DataFrame) -> None:
    kinds = [(name, col) for name, (col, ok, bad) in KINDS.items() if counts(judged, col, ok, bad)[0]]
    if kinds:
        doc.heading("판정 기준", keep=8 * LINE_H)
    for name, col in kinds:
        doc.item(name, keep=len(CRITERIA[col]) * 1.5 * LINE_H)
        for s in CRITERIA[col]:
            doc.paragraph("○", s, 4, 11)
        doc.y += 3


def summary(doc: Document, judged: pd.DataFrame, min_n: int) -> None:
    doc.heading("분석 결과 총괄")
    rows, top_rows = [], []
    for name, (col, ok, bad) in KINDS.items():
        n, k, excluded = counts(judged, col, ok, bad)
        rows.append([name, *([f"{n:,}", f"{k:,}", pct(k / n), ci(*wilson_ci(k, n))] if n else ["-"] * 4), f"{excluded:,}"])
        top = top_slots(rate_table(judged, min_n, col, ok, bad), TOP_N) if n else pd.DataFrame()
        top_rows += [[name, str(i + 1), str(r.intersection), r.slot, f"{r.n_target:,}", f"{getattr(r, f'n_{bad}'):,}", pct(r.violation_rate)]
                     for i, r in enumerate(top.itertuples())]
    best = overall_slots(judged, min_n)
    doc.item("종합 단속 우선 검토 시간대", "단위: 건, %", keep=table_h(len(best)) + 10)
    if len(best):
        cells = [[f"{i + 1}순위", str(r.intersection), r.slot, f"{pct(r.rate_ped_result)} ({r.rank_ped_result}위)", f"{r.n_ped_result:,}",
                  f"{pct(r.rate_red_result)} ({r.rank_red_result}위)", f"{r.n_red_result:,}"] for i, r in enumerate(best.itertuples())]
        doc.table(["종합 순위", "교차로", "시간대", "미양보 위반율(순위)", "의심", "미정지 위반율(순위)", "의심"], cells, [20, 34, 24, 34, 12, 34, 12])
        doc.y -= 4
        doc.paragraph("※", f"유형별 위반율 순위의 합이 작은 순(동점은 미양보 위반율 높은 순). 한 유형이라도 {min_n}건 미만이면 제외", 2, 9)
        doc.y += 2
    else:
        doc.paragraph("○", f"두 유형 모두 판정 대상이 {min_n}건 이상이면서 위반 의심이 있는 시간대 없음", 4)

    doc.item("위반 유형별 현황", "단위: 건, %", keep=table_h(len(rows)))
    doc.table(["구분", "판정 대상", "위반 의심", "위반율", "95% 신뢰구간", "판정 제외"], rows, [34, 26, 26, 24, 36, 24])

    doc.item(f"유형별 단속 우선 검토 시간대(위반율 상위 {TOP_N}개 이내)", "단위: 건, %", keep=table_h(len(top_rows)))
    if top_rows:
        doc.table(["구분", "순위", "교차로", "시간대", "판정 대상", "위반 의심", "위반율"], top_rows, [34, 14, 34, 26, 22, 22, 20], merge=True)
    else:
        doc.paragraph("○", f"판정 대상이 {min_n}건 이상이면서 위반 의심이 있는 시간대 없음", 4)

    rows = vehicle_rows(judged)
    if rows:
        doc.item("차종별 현황", "단위: 건, %", keep=table_h(len(rows)))
        doc.table(["차종", "우회전", "미양보 대상", "미양보 의심", "위반율", "미정지 대상", "미정지 의심", "위반율"], rows,
                  [30, 18, 22, 22, 18, 22, 22, 18], total=True)


def rate_chart(doc: Document, chunk: pd.DataFrame) -> None:
    """위반율 막대(선 = 95% 신뢰구간, 빗금 막대 = 참고용). 신뢰구간 선 끝 옆에 값을 적는다."""
    h = len(chunk) * BAND_H
    ax = doc.fig.add_axes([(LEFT + CHART_LABEL_W) / PAGE_W, 1 - (doc.y + h) / PAGE_H, (WIDTH - CHART_LABEL_W - 16) / PAGE_W, h / PAGE_H])
    rate = chunk["violation_rate"].fillna(0).to_numpy() * 100
    low = (chunk["violation_rate"] - chunk["ci_low"]).fillna(0).to_numpy() * 100
    high = (chunk["ci_high"] - chunk["violation_rate"]).fillna(0).to_numpy() * 100
    bars = ax.barh(range(len(chunk)), rate, height=0.5, color=BAR, edgecolor=BAR, linewidth=0.6, xerr=[low, high],
                   error_kw={"ecolor": INK, "elinewidth": 0.8, "capsize": 2})
    for bar, note in zip(bars, chunk["note"]):
        if note:
            bar.set_facecolor("white")
            bar.set_hatch("/////")
    ax.set_ylim(len(chunk) - 0.5, -0.5)
    ax.set_yticks([])
    for i, r in enumerate(chunk.itertuples()):
        doc.cell(LEFT, doc.y + i * BAND_H, CHART_LABEL_W, BAND_H, f"{r.intersection} {r.slot}")
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100], ["0%", "20%", "40%", "60%", "80%", "100%"])
    ax.tick_params(length=0, labelsize=9)
    for label in ax.get_xticklabels():
        label.set_family(doc.fonts["table"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(LINE)
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.5)
    ax.set_axisbelow(True)
    for i, (r, h_err, value) in enumerate(zip(rate, high, chunk["violation_rate"])):
        ax.text(min(r + h_err, 100) + 1.5, i, pct(value), fontsize=9, family=doc.fonts["table"], color=INK, va="center", clip_on=False)
    doc.y += h + 10


def kind_section(doc: Document, judged: pd.DataFrame, raw: pd.DataFrame, name: str, min_n: int) -> None:
    col, ok, bad = KINDS[name]
    n, k, _ = counts(judged, col, ok, bad)
    if not n:
        return
    t = rate_table(judged, min_n, col, ok, bad).sort_values(["intersection", "slot"])
    doc.heading(f"{name} 의심 세부 현황", keep=8 + table_h(len(t) + 1))

    doc.item("교차로·시간대별 위반율", "단위: 건, %", keep=table_h(len(t) + 1))
    rows = [["계", "", f"{n:,}", f"{k:,}", pct(k / n), ci(*wilson_ci(k, n)), ""]]
    rows += [[str(r.intersection), r.slot, f"{r.n_target:,}", f"{getattr(r, f'n_{bad}'):,}", pct(r.violation_rate), ci(r.ci_low, r.ci_high), r.note]
             for r in t.itertuples()]
    doc.table(["교차로", "시간대", "판정 대상", "위반 의심", "위반율", "95% 신뢰구간", "비고"], rows, [34, 26, 24, 24, 20, 32, 20], total=True, merge=True)

    for start in range(0, len(t), MAX_BANDS):
        chunk = t.iloc[start:start + MAX_BANDS]
        doc.item("시간대별 위반율 비교", "가는 선: 95% 신뢰구간, 빗금: 참고용", keep=len(chunk) * BAND_H + 10)
        rate_chart(doc, chunk)

    doc.item("판정 결과별 건수", "단위: 건", keep=table_h(len(t) + 1))
    values = [value for value, _ in SEGMENTS[col]]
    by_slot = [("계", "", raw)] + [(str(site), slot, g) for (site, slot), g in raw.groupby(["intersection", "slot"])]
    rows = [[site, slot, f"{len(g):,}", *[f"{int((g[col] == value).sum()):,}" for value in values]] for site, slot, g in by_slot]
    doc.table(["교차로", "시간대", "우회전", *[label for _, label in SEGMENTS[col]]], rows, [34, 26, 22, *[22] * len(values)], total=True, merge=True)

    blocks = detail_blocks(raw, col)  # 왼쪽·오른쪽 반씩 나란히
    doc.need(8 + table_h(max((len(rows) for _, _, rows in blocks), default=0)))
    top, bottom, half = doc.y, doc.y, (WIDTH - 6) / 2
    for i, (block, columns, rows) in enumerate(blocks):
        doc.y, x = top, LEFT + i * (half + 6)
        doc.item(block, x=x)
        doc.table(columns, rows, [60, 22], x=x, w=half)
        bottom = max(bottom, doc.y)
    doc.y = bottom


def cautions(doc: Document, min_n: int) -> None:
    lines = [
        f"본 자료는 {NOTICE}.",
        "위반율은 위반 의심 건수를 판정 대상 건수(제외를 뺀 건수)로 나눈 값이며, 신뢰구간은 Wilson 95% 신뢰구간임",
        f"판정 대상이 {min_n}건 미만인 시간대는 '참고용'으로 표시하고 단속 우선 검토 시간대에서 제외함",
        "적색 미정지는 우회전 방향 신호가 적색일 때만 판정하며, 경미는 정지선을 넘었으나 횡단보도 전에 정지한 경우임.  끝.",
    ]
    doc.heading("자료 이용 시 유의사항", keep=len(lines) * 2 * LINE_H)
    for s in lines:
        doc.paragraph("○", s, 0, 11)


def build_report(judged: pd.DataFrame, out: Path, min_n: int, title: str = "우회전 차량 위반 의심 분석 결과 보고",
                 videos: dict | None = None, red_mode: str = "보통") -> Path:
    """판정 표로 보고서 PDF를 만들고 경로를 돌려준다. videos = {영상 이름: 교차로}(이벤트가 0건인 영상까지 세려면 준다).
    red_mode = 적색 판정 모드(기술서 3.1, 기본 보통). 판정 결과별 건수와 제외 사유는 모드를 적용하기 전 값(경미 정지를 따로)으로 보여 준다."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw, judged = judged, apply_red_mode(judged, red_mode)
    videos = videos or dict(zip(judged["video"], judged["intersection"]))  # 이벤트 0건인 영상도 세려면 넘긴다
    # fonttype 42 = 글꼴을 트루타입으로 넣어 PDF에서 글자를 찾고 복사할 수 있게 한다
    with plt.rc_context({"axes.unicode_minus": False, "pdf.fonttype": 42, "hatch.linewidth": 0.6}), \
            PdfPages(out, metadata={"Title": title, "Creator": None, "Producer": None}) as pdf:
        doc = Document(pdf)
        title_block(doc, judged, title, min_n)
        overview(doc, judged, videos, red_mode)
        summary(doc, judged, min_n)
        criteria(doc, judged)
        for name in KINDS:
            kind_section(doc, judged, raw, name, min_n)
        cautions(doc, min_n)
        doc.close()
    return out
