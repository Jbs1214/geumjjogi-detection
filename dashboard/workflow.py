# 담당: 장범수
"""대시보드 "영상 분석" 페이지: ① 영상 등록 → ② 영역 설정 → ③ 분석(메일 선택) → ④ 리포트.
여러 교차로·영상을 등록해 두고 영역 설정을 누르면 영역이 없는 영상마다 영역 지정 창(tracking.roi_tool)을 차례로 띄운다.
영역 지정 창은 대시보드를 실행한 PC 화면에 뜬다(로컬 PC에서 쓰는 것을 전제).
화면 없는 처리(목록 저장, 영역 지정 실행, 분석 작업)는 dashboard/pipeline.py.
"""
import re
from datetime import datetime, time
from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard import mailer, pipeline
from dashboard.auth import EMAIL_PATTERN

REFRESH_SEC = 5  # 분석 진행 상황을 다시 읽는 간격(초)


def register_section() -> None:
    st.subheader("① 영상 등록")
    st.caption("교차로·영상을 여러 개 등록해 둔 뒤 아래 '영역 설정'을 누르면 한 번에 차례로 설정합니다. 영상은 PC에 있는 그대로 쓰고 복사하지 않습니다.")
    with st.form("add-video", clear_on_submit=False):
        c1, c2 = st.columns(2)
        name = c1.text_input("교차로 이름", placeholder="예: 강남역 (공백·밑줄 없이)")
        direction = c2.selectbox("우회전 방향 (들어오는 쪽 → 꺾어 나가는 쪽)", list(pipeline.DIRECTIONS))
        c3, c4 = st.columns(2)
        day = c3.date_input("촬영 날짜")
        start = c4.time_input("촬영 시작 시각", time(8, 0), step=60)
        path = st.text_input("영상 파일 경로", placeholder=r"예: D:\영상\강남역_1122_0800.mp4  (탐색기에서 Shift+우클릭 → 경로로 복사)")
        if st.form_submit_button("목록에 추가"):
            error = pipeline.add_video(name, direction, datetime.combine(day, start), path)
            if error:
                st.error(error)
            else:
                st.success("목록에 추가했습니다.")

    videos = pipeline.load_videos()
    if not videos:
        st.info("등록한 영상이 없습니다.")
        return
    busy = pipeline.running_job() is not None
    header = st.columns([2, 1, 2, 4, 1.4, 1])
    for col, label in zip(header, ["교차로", "방향", "촬영 시작", "영상", "상태", ""]):
        col.markdown(f"**{label}**")
    for v in videos:
        c = st.columns([2, 1, 2, 4, 1.4, 1], vertical_alignment="center")
        c[0].text(v["intersection"])
        c[1].text(v["direction"])
        c[2].text(v["start"])
        c[3].text(Path(v["source"]).name)
        c[4].text(pipeline.video_status(v))
        if c[5].button("빼기", key=f"remove-{v['id']}", disabled=busy, help="목록에서만 뺍니다(영상·결과는 그대로)"):
            pipeline.remove_video(v["id"])
            st.rerun()


def roi_section() -> None:
    st.subheader("② 영역 설정")
    videos = pipeline.load_videos()
    todo = [v for v in videos if pipeline.video_status(v) == "영역 미설정"]
    st.caption("누르면 이 PC 화면에 영역 지정 창이 영상마다 차례로 뜹니다. 창의 안내대로 찍고 마지막 화면에서 Enter로 저장하면 다음 영상으로 넘어갑니다(Esc = 그 영상 건너뛰기).")
    c1, c2 = st.columns([1, 2])
    again = c2.selectbox("이미 설정한 영상 다시 설정", ["(선택 안 함)", *[v["id"] for v in videos if v not in todo]])
    targets = [v for v in videos if v["id"] == again] if again != "(선택 안 함)" else todo
    label = f"{again} 다시 설정" if again != "(선택 안 함)" else f"영역 설정 시작 ({len(todo)}개)"
    if c1.button(label, type="primary", disabled=not targets or pipeline.running_job() is not None):
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


def analyze_section() -> None:
    st.subheader("③ 분석")
    ready = [v for v in pipeline.load_videos() if pipeline.video_status(v) == "영역 완료"]
    running = pipeline.running_job()
    st.caption("영역을 설정한 영상을 분석합니다(차량·보행자 검출과 추적이라 영상 길이만큼 이상 걸립니다). "
               "메일 주소를 적으면 끝난 뒤 리포트 PDF를 보냅니다. 화면을 닫아도 분석은 계속됩니다.")
    email = st.text_input("리포트 받을 메일 (선택)", placeholder="name@example.com")
    if email and not re.fullmatch(EMAIL_PATTERN, email.strip()):
        st.caption(":red[메일 주소 형식이 아닙니다.]")
    elif email and not mailer.configured():
        st.caption(":orange[보내는 메일 계정 설정(.env의 SMTP_*)이 없어 메일은 보내지 못합니다. 리포트는 아래에서 받을 수 있습니다.]")
    bad_email = bool(email) and not re.fullmatch(EMAIL_PATTERN, email.strip())
    if st.button(f"분석 시작 ({len(ready)}개 영상)", type="primary", disabled=not ready or running is not None or bad_email):
        pipeline.start_job(ready, email)
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
             "rules": "판정", "완료": "완료", "실패": "실패"}
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
    st.subheader("④ 리포트")
    jobs = [(job, s) for job, s in pipeline.list_jobs() if s["state"] not in ("대기", "진행 중")]
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
            report = Path(s.get("report") or "")
            if report.is_file():
                c2.download_button("리포트 PDF", report.read_bytes(), f"report_{job.name}.pdf", "application/pdf",
                                   key=f"pdf-{job.name}")


def analysis_page() -> None:
    register_section()
    st.divider()
    roi_section()
    st.divider()
    analyze_section()
    st.divider()
    report_section()
