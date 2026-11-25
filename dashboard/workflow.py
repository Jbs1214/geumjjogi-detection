# 담당: 장범수
"""대시보드 "영상 분석" 페이지: 1. 영상 등록 → 2. 영역 설정 → 3. 분석(메일 선택) → 4. 보고서.
여러 교차로·영상을 등록해 두고 영역 설정을 누르면 영역이 없는 영상마다 영역 지정 창(tracking.roi_tool)을 차례로 띄운다.
영역 지정 창은 대시보드를 실행한 PC 화면에 뜬다(로컬 PC에서 쓰는 것을 전제).
영상을 등록할 때 날씨와 신호 조건을 고르고, 낮·야간은 촬영 시작 시각으로 정한다(기술서 11.1).
사이드바: 조회 조건(교차로·촬영 기간·낮/야간·날씨, 등록된 영상 목록과 처리 현황에만 적용)과 처리 현황.
화면 없는 처리(목록 저장, 영역 지정 실행, 분석 작업)는 dashboard/pipeline.py.
"""
import re
from datetime import datetime, time
from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard import mailer, pipeline, ui
from dashboard.auth import EMAIL_PATTERN

REFRESH_SEC = 5  # 분석 진행 상황을 다시 읽는 간격(초)
NEW_SITE = "+ 새 교차로"
ALL = "전체"
NO_REDO = "(선택 안 함)"
BADGES = {"분석 완료": "green", "영역 완료": "blue", "영역 미설정": "orange"}  # 상태: 배지 색
TABLE_WIDTHS = [2, 1, 2, 1.6, 2.4, 1]


def start_time(video: dict) -> datetime:
    return datetime.strptime(video["start"], "%Y-%m-%d %H:%M")


def sidebar_filters(videos: list[dict]) -> list[dict]:
    """사이드바 조회 조건을 그리고, 걸러 낸 영상 목록을 돌려준다."""
    side = st.sidebar
    side.subheader("조회 조건")
    if not videos:
        side.caption("등록한 영상이 없습니다.")
        return []
    site = side.selectbox("교차로", [ALL, *sorted({v["intersection"] for v in videos})])
    days = [start_time(v).date() for v in videos]
    period = side.date_input("촬영 기간", (min(days), max(days)))
    when = side.segmented_control("낮 / 야간", [ALL, "낮", "야간"], default=ALL, required=True)
    weather = side.segmented_control("날씨", [ALL, *pipeline.WEATHERS], default=ALL, required=True, key="filter-weather")
    if len(period) != 2:  # 날짜 범위를 고르는 중
        return videos
    return [v for v in videos
            if site in (ALL, v["intersection"]) and period[0] <= start_time(v).date() <= period[1]
            and when in (ALL, pipeline.day_night(start_time(v))) and weather in (ALL, v.get("weather"))]


def sidebar_status(videos: list[dict]) -> None:
    """사이드바 처리 현황."""
    st.sidebar.subheader("처리 현황")
    for name, n in pipeline.status_counts(videos).items():
        st.sidebar.text(f"{name} {n}")


def register_section() -> None:
    st.subheader("1. 영상 등록")
    st.caption("교차로·영상을 여러 개 등록해 둔 뒤 '영역 설정 시작'을 누르면 한 번에 차례로 설정합니다. 영상은 PC에 있는 그대로 쓰고 복사하지 않습니다.")
    with st.container(key="register"):  # 폼을 쓰지 않는다: 새 교차로 입력 칸과 낮·야간 표시가 고르는 즉시 바뀌어야 한다
        known = sorted({v["intersection"] for v in pipeline.load_videos()})
        c1, c2 = st.columns(2)
        name = c1.selectbox("교차로 이름", [*known, NEW_SITE])
        if name == NEW_SITE:
            name = c1.text_input("새 교차로 이름", placeholder="예: 강남역 (공백·밑줄 없이)")
        direction = c2.selectbox("우회전 방향", list(pipeline.DIRECTIONS), help="들어오는 쪽 → 꺾어 나가는 쪽")
        c3, c4 = st.columns(2)
        day = c3.date_input("촬영 날짜")
        start = datetime.combine(day, c4.time_input("촬영 시작 시각", time(8, 0), step=60))
        c5, c6 = st.columns(2)
        weather = c5.segmented_control("날씨", pipeline.WEATHERS, default=pipeline.WEATHERS[0], required=True)
        c6.text_input("낮·야간 (자동)", pipeline.day_night(start), disabled=True)
        signal = st.segmented_control("신호 조건", list(pipeline.SIGNALS), default=next(iter(pipeline.SIGNALS)), required=True)
        path = st.text_input("영상 파일 경로", placeholder=r"예: D:\영상\강남역_1122_0800.mp4  (탐색기에서 Shift+우클릭 → 경로로 복사)")
    if st.container(horizontal=True, horizontal_alignment="right").button("목록에 추가"):
        error = pipeline.add_video(name, direction, start, path, weather, pipeline.SIGNALS[signal])
        if error:
            st.error(error)
        else:
            st.rerun()  # 사이드바의 조회 조건·처리 현황까지 새 목록으로 다시 그린다


def video_table(videos: list[dict], duplicated: set[str]) -> None:
    st.markdown(f"**등록된 영상** (총 {len(videos)}건)")
    if not videos:
        st.info("등록한 영상이 없거나 조회 조건에 맞는 영상이 없습니다.")
        return
    busy = pipeline.running_job() is not None
    rows = st.container(key="video-list")
    for col, label in zip(rows.columns(TABLE_WIDTHS), ["교차로", "방향", "촬영 시작", "날씨·주야", "상태", "관리"]):
        col.markdown(f"**{label}**")
    for v in videos:
        c = rows.columns(TABLE_WIDTHS, vertical_alignment="center")
        c[0].text(v["intersection"])
        c[1].text(v["direction"])
        c[2].text(v["start"])
        c[3].text("·".join(filter(None, [v.get("weather"), pipeline.day_night(start_time(v))])))  # 예전에 등록한 영상은 날씨가 없다
        status = pipeline.video_status(v)
        c[4].markdown(f":{BADGES[status]}-badge[{status}]" + (" :red-badge[중복]" if v["id"] in duplicated else ""))
        if c[5].button("빼기", key=f"remove-{v['id']}", disabled=busy, help="목록에서만 뺍니다(영상·결과는 그대로)"):
            pipeline.remove_video(v["id"])
            st.rerun()
    st.caption('같은 파일명이 두 번 등록되면 "중복"으로 표시합니다.')


def roi_section(videos: list[dict]) -> None:
    """영역 설정 시작 = 조회 조건에 맞는 영상 중 영역이 없는 것 전부. 이미 설정한 영상은 하나씩 다시 한다."""
    st.subheader("2. 영역 설정")
    todo = [v for v in videos if pipeline.video_status(v) == "영역 미설정"]
    st.caption("'영역 설정 시작'을 누르면 이 PC 화면에 영역 지정 창이 영상마다 차례로 뜹니다. 창의 안내대로 찍고 마지막 화면에서 Enter로 저장하면 다음 영상으로 넘어갑니다(Esc = 그 영상 건너뛰기).")
    busy = pipeline.running_job() is not None
    start = st.button(f"영역 설정 시작 ({len(todo)}개)", type="primary", disabled=not todo or busy)
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    again = c1.selectbox("이미 설정한 영상 다시 설정", [NO_REDO, *[v["id"] for v in videos if v not in todo]])
    redo = c2.button("다시 설정", disabled=again == NO_REDO or busy)
    targets = [v for v in videos if v["id"] == again] if redo else todo if start else []
    if targets:
        with st.status("영역 지정 창을 띄웁니다…", expanded=True) as box:
            saved = 0
            for i, v in enumerate(targets, 1):
                box.write(f"{i}/{len(targets)} **{v['intersection']} {v['direction']} {v['start']}**: PC 화면의 창에서 영역을 찍으세요.")
                if pipeline.run_roi_tool(v):
                    saved += 1
                    box.write("　→ 저장했습니다.")
                else:
                    box.write("　→ 저장하지 않았습니다(Esc). 나중에 다시 설정하세요.")
            box.update(label=f"영역 설정 끝: {saved}/{len(targets)}개 저장", state="complete")


def analyze_section(red_mode: str) -> None:
    st.subheader("3. 분석")
    ready = [v for v in pipeline.load_videos() if pipeline.video_status(v) == "영역 완료"]
    running = pipeline.running_job()
    st.caption("영역을 설정한 영상을 분석합니다(차량·보행자 검출과 추적이라 영상 길이만큼 이상 걸립니다). "
               f"메일 주소를 적으면 끝난 뒤 보고서 PDF를 보내고, 이어서 위반 의심 클립(블러)을 만듭니다. 화면을 닫아도 분석은 계속됩니다. 보고서는 왼쪽의 판정 모드({red_mode})로 만듭니다.")
    email = st.text_input("보고서 받을 메일 (선택)", placeholder="name@example.com")
    if email and not re.fullmatch(EMAIL_PATTERN, email.strip()):
        st.caption(":red[메일 주소 형식이 아닙니다.]")
    elif email and not mailer.configured():
        st.caption(":orange[보내는 메일 계정 설정(.env의 SMTP_*)이 없어 메일은 보내지 못합니다. 보고서는 아래에서 받을 수 있습니다.]")
    bad_email = bool(email) and not re.fullmatch(EMAIL_PATTERN, email.strip())
    if st.button(f"분석 시작 ({len(ready)}개 영상)", type="primary", disabled=not ready or running is not None or bad_email):
        pipeline.start_job(ready, email, red_mode)
        st.rerun()
    if running is not None:
        progress(running)


@st.fragment(run_every=REFRESH_SEC)
def progress(job: Path) -> None:
    """진행 중인 작업을 REFRESH_SEC마다 다시 읽어 보여 준다. 끝나면 페이지 전체를 다시 그린다."""
    status = pipeline.read_status(job)
    if status["state"] not in ("대기", "진행 중"):
        st.rerun()
    names = {"대기": "대기", "track": "검출·추적", "events": "우회전 찾기", "signal": "신호등 색", "features": "특징 계산",
             "rules": "판정", "clips": "클립 생성", "완료": "완료", "실패": "실패"}
    overall, per = pipeline.job_progress(job, status)
    st.progress(overall, text=f"분석 중 {overall:.0%} (시작 {status.get('started', status['created'])})")
    rows = []
    for vid, step in status["videos"].items():
        left = ""
        if step == "track" and per[vid] and status.get("step_started"):   # 지금까지 걸린 시간으로 남은 시간 어림
            spent = (datetime.now() - datetime.fromisoformat(status["step_started"])).total_seconds()
            left = f"약 {spent * (1 - per[vid]) / per[vid] / 60:.0f}분"
        rows.append({"영상": vid, "단계": names.get(step, step), "진행률": f"{per[vid]:.0%}", "남은 시간(추적)": left})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def report_section() -> None:
    st.subheader("4. 보고서")
    # 보고서는 클립을 만드는 동안에도 받을 수 있다
    jobs = [(job, s) for job, s in pipeline.list_jobs() if s["state"] not in ("대기", "진행 중") or s.get("report")]
    if not jobs:
        st.info("끝난 분석이 없습니다.")
        return
    for job, s in jobs:
        with st.container(border=True):
            c1, c2 = st.columns([3, 1], vertical_alignment="center")
            c1.markdown(f"**{s.get('finished', s['created'])} 분석 · {s['state']}** · 영상 {len(s['videos'])}개")
            c1.caption(f"메일: {s.get('mail', '-')}")
            for vid, error in s.get("errors", {}).items():
                c1.caption(f":red[{vid}: {error}]")
            for vid, error in s.get("clip_errors", {}).items():
                c1.caption(f":orange[{vid}: {error} (판정 결과와 보고서는 그대로입니다)]")
            report = Path(s.get("report") or "")
            if report.is_file():
                c2.download_button("보고서 내려받기", report.read_bytes(), f"report_{job.name}.pdf", "application/pdf",
                                   key=f"pdf-{job.name}")


def analysis_page(red_mode: str) -> None:
    """red_mode = 사이드바에서 고른 판정 모드(보통·엄격). 분석 보고서를 이 모드로 만든다."""
    everything = pipeline.load_videos()
    videos = sidebar_filters(everything)
    sidebar_status(videos)
    ui.steps(["영상 등록", "영역 설정", "분석", "보고서"])
    register_section()
    video_table(videos, pipeline.duplicates(everything))
    roi_section(videos)
    analyze_section(red_mode)
    report_section()
