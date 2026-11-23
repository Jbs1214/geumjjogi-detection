# 담당: 장범수
"""분석 리포트 PDF. 판정 표(outputs/judged 모양)의 통계만 담는다(영상·프레임·클립은 넣지 않는다).
matplotlib(이미 쓰는 라이브러리)의 PdfPages로 A4 세로 쪽을 만든다. 글꼴은 맑은 고딕(eval/overlay.py와 같음).

1쪽: 제목, 만든 시각, 자료 기간·교차로·영상 수, "위반 의심 통계" 문구, 위반 유형별 요약(판정 대상·위반 의심·위반율·Wilson 95% 신뢰구간·제외)
다음 쪽: 위반 유형마다 교차로 x 시간대 위반율 표(판정 대상 min_n 미만은 참고용)와 막대그래프(신뢰구간 표시)
위반율 계산은 대시보드(dashboard/app.py)와 같은 eval.report.rate_table·wilson_ci를 쓴다.
"""
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 화면 없이 파일로만
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from eval.report import RED_MODES, apply_red_mode, rate_table, wilson_ci

NOTICE = "위반 의심 통계이며 실제 단속은 별도 확인이 필요하다"
KINDS = {  # 표시 이름: (판정 컬럼, 위반 아님, 위반 의심). dashboard/app.py KINDS와 같아야 한다(테스트로 확인)
    "보행자 미양보": ("ped_result", "yield", "no_yield"),
    "적색 미정지": ("red_result", "stopped", "not_stopped"),
}
EXCLUDE = "exclude"
A4 = (8.27, 11.69)  # 인치
ROWS_PER_PAGE = 22  # 위반율 표 한 쪽 줄 수
FONT = "Malgun Gothic"
NAVY, RED, GRAY = "#1f3a68", "#c62828", "#6b7280"


def pct(x: float) -> str:
    return "-" if pd.isna(x) else ("1% 미만" if 0 < x < 0.005 else f"{x:.0%}")


def summary_rows(judged: pd.DataFrame) -> list[list[str]]:
    rows = []
    for name, (col, ok, bad) in KINDS.items():
        if col not in judged or not judged[col].isin([ok, bad]).any():
            rows.append([name, "-", "-", "판정 결과 없음", "-"])
            continue
        n, k = int(judged[col].isin([ok, bad]).sum()), int((judged[col] == bad).sum())
        low, high = wilson_ci(k, n)
        rows.append([name, f"{n}건", f"{k}건", f"{pct(k / n)} ({low:.0%}~{high:.0%})", f"{int((judged[col] == EXCLUDE).sum())}건"])
    return rows


def table(ax, rows: list[list[str]], columns: list[str], widths: list[float]) -> None:
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=columns, colWidths=widths, loc="upper center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.5)
    for (r, _), cell in t.get_celld().items():
        cell.set_edgecolor("#d0d5dd")
        if r == 0:
            cell.set_facecolor("#e8eef7")
            cell.set_text_props(weight="bold", color=NAVY)


def cover(pdf: PdfPages, judged: pd.DataFrame, title: str, min_n: int, videos: dict | None, red_mode: str) -> None:
    fig = plt.figure(figsize=A4)
    wall = pd.to_datetime(judged["wall_time"])
    videos = videos or dict(zip(judged["video"], judged["intersection"]))  # 이벤트 0건인 영상도 세려면 넘긴다
    sites = ", ".join(sorted({str(s) for s in videos.values()})) or "-"
    period = f"{wall.min():%Y-%m-%d %H:%M} ~ {wall.max():%Y-%m-%d %H:%M}" if len(judged) else "- (우회전 이벤트 없음)"
    lines = [
        (title, 20, NAVY, "bold"),
        (f"만든 시각 {datetime.now():%Y-%m-%d %H:%M}", 10, GRAY, "normal"),
        ("", 6, GRAY, "normal"),
        (f"자료 기간  {period}", 11, "black", "normal"),
        (f"교차로  {sites}", 11, "black", "normal"),
        (f"영상 {len(videos)}개 · 우회전 이벤트 {len(judged)}건", 11, "black", "normal"),
    ]
    y = 0.94
    for text, size, color, weight in lines:
        fig.text(0.08, y, text, fontsize=size, color=color, weight=weight)
        y -= 0.012 + size / 520
    fig.text(0.08, y - 0.02, f"※ {NOTICE}", fontsize=11, color=RED, weight="bold",
             bbox={"boxstyle": "round,pad=0.6", "facecolor": "#fdf1f0", "edgecolor": RED})
    fig.text(0.08, y - 0.08, "요약", fontsize=14, color=NAVY, weight="bold")
    ax = fig.add_axes([0.08, y - 0.30, 0.84, 0.2])
    table(ax, summary_rows(judged), ["위반 유형", "판정 대상", "위반 의심", "위반율 (95% 신뢰구간)", "제외"], [0.2, 0.15, 0.15, 0.32, 0.12])
    fig.text(0.08, y - 0.33,
             f"위반율 = 위반 의심 / 판정 대상(제외 뺀 것). 신뢰구간은 Wilson 95%. 교차로·시간대별 판정 대상 {min_n}건 미만은 '참고용'.\n"
             "자동 판정 결과(사람 확인 전)입니다. 적색 미정지는 우회전 방향 신호가 빨간불일 때만 봅니다.\n"
             f"적색 판정 모드: {red_mode} (정지선은 넘었지만 횡단보도 전에 선 경미 정지를 "
             f"{'위반 의심으로' if RED_MODES[red_mode] == 'not_stopped' else '정지로'} 봅니다).",
             fontsize=8.5, color=GRAY, va="top", linespacing=1.6)
    pdf.savefig(fig)
    plt.close(fig)


def kind_pages(pdf: PdfPages, judged: pd.DataFrame, name: str, min_n: int) -> None:
    col, ok, bad = KINDS[name]
    if col not in judged or not judged[col].isin([ok, bad]).any():
        return
    t = rate_table(judged, min_n, col, ok, bad).sort_values(["intersection", "slot"])
    rows = [[str(r.intersection), r.slot, f"{r.n_target}", f"{getattr(r, f'n_{bad}')}", pct(r.violation_rate),
             "-" if pd.isna(r.ci_low) else f"{r.ci_low:.0%}~{r.ci_high:.0%}", r.note or "표본 충분"] for r in t.itertuples()]
    for start in range(0, len(rows), ROWS_PER_PAGE):
        chunk, part = t.iloc[start:start + ROWS_PER_PAGE], rows[start:start + ROWS_PER_PAGE]
        fig = plt.figure(figsize=A4)
        fig.text(0.08, 0.95, f"{name} · 교차로 x 시간대", fontsize=15, color=NAVY, weight="bold")
        fig.text(0.08, 0.925, NOTICE, fontsize=9, color=RED)
        ax = fig.add_axes([0.08, 0.52, 0.84, 0.38])
        table(ax, part, ["교차로", "시간대", "판정 대상", "위반 의심", "위반율", "95% 신뢰구간", "신뢰도"],
              [0.17, 0.14, 0.12, 0.12, 0.12, 0.17, 0.16])
        bar = fig.add_axes([0.25, 0.07, 0.67, 0.38])
        labels = (chunk["intersection"].astype(str) + " " + chunk["slot"]).tolist()
        rate = chunk["violation_rate"].fillna(0).to_numpy() * 100
        low = (chunk["violation_rate"] - chunk["ci_low"]).fillna(0).to_numpy() * 100
        high = (chunk["ci_high"] - chunk["violation_rate"]).fillna(0).to_numpy() * 100
        colors = [RED if note == "" else "#f2a7a7" for note in chunk["note"]]
        bar.barh(labels, rate, xerr=[low, high], color=colors, error_kw={"ecolor": GRAY, "capsize": 3})
        bar.invert_yaxis()
        bar.set_xlim(0, 100)
        bar.set_xlabel("위반율 (%) · 선 = 95% 신뢰구간 · 옅은 막대 = 참고용")
        bar.tick_params(labelsize=8)
        pdf.savefig(fig)
        plt.close(fig)


def build_report(judged: pd.DataFrame, out: Path, min_n: int, title: str = "우회전 위반 의심 분석 리포트",
                 videos: dict | None = None, red_mode: str = "보통") -> Path:
    """판정 표로 리포트 PDF를 만들고 경로를 돌려준다. videos = {영상 이름: 교차로}(이벤트가 0건인 영상까지 세려면 준다).
    red_mode = 적색 판정 모드(기술서 3.1, 기본 보통)."""
    out = Path(out)
    judged = apply_red_mode(judged, red_mode)
    out.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": FONT, "axes.unicode_minus": False}), PdfPages(out) as pdf:
        cover(pdf, judged, title, min_n, videos, red_mode)
        for name in KINDS:
            kind_pages(pdf, judged, name, min_n)
    return out
