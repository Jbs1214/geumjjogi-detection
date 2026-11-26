# 담당: 육주영
"""위반율 대시보드 (Streamlit, 읽기 전용). outputs/judged/*.csv를 pandas로 모아 보여 준다.
실행: streamlit run dashboard/app.py

화면(기술서 3.2, 행정 업무시스템형 모양은 dashboard/ui.py): 조회 조건(교차로·기간·위반 유형·정렬), 종합 단속 우선 검토 시간대(report_pdf.overall_slots, 보고서와 같은 계산),
유형별 총괄 현황, 유형별 단속 우선 검토 시간대 상위 3개(표본 충분, 위반 의심 1건 이상)와 클립 보기, 교차로·차종별 현황, 교차로 x 시간대 위반율 표(Wilson 95% 신뢰구간,
configs/eval.yaml의 min_n_reference 미만 "참고용")와 CSV 내려받기.
"위반 의심 통계이며 실제 단속은 별도 확인이 필요하다" 문구를 항상 표시한다.
위반 유형 "전체"는 두 유형을 나란히 보여 주고 위반율을 하나로 합치지 않는다(분모가 다르다).
사이드바의 분석 모드(보통·엄격)는 경미 정지(stopped_minor)를 정지로 볼지 위반 의심으로 볼지를 고른다(기술서 3.1, eval.report.apply_red_mode).
레포 루트 .env에 DB 접속 값이 있으면(dashboard/db.py) 로그인한 사람에게만 보여 주고 자료를 MySQL에서 읽는다. 가입은 신청 후 관리자 승인(관리자 전용 가입 승인 관리 페이지).
왼쪽 업무 메뉴에서 "영상 분석" 화면(영상 등록 → 영역 설정 → 분석 → 보고서, dashboard/workflow.py)으로 바꾼다.
위반 통계 화면의 "보고서(PDF) 생성"은 지금까지 쌓인 판정(현재 조회 조건)으로 보고서를 만든다(dashboard/report_pdf.py).
"""
import sys
from datetime import date
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # streamlit run은 레포 루트를 import 경로에 넣지 않는다

import pandas as pd
import streamlit as st
import yaml

from dashboard import db, report_pdf, ui, workflow
from dashboard.auth import PASSWORD_MIN, format_phone, signup_errors, verify_password
from eval.report import RED_MODES, apply_red_mode, rate_table, top_slots, wilson_ci

NOTICE = report_pdf.NOTICE
EXPORT_NOTE = f"자동 판정 결과(사람 확인 전). {NOTICE}"
TOP_N = report_pdf.TOP_N
KINDS = {  # 표시 이름: (판정 컬럼, 위반 아님, 위반 의심, 내려받기 파일 이름)
    "보행자 미양보": ("ped_result", "yield", "no_yield", "violation_rate.csv"),
    "적색 미정지": ("red_result", "stopped", "not_stopped", "red_rate.csv"),
}
ALL_KINDS = "전체"
BAR_COLOR = "#44546a"  # 위반율 막대
REFERENCE_STYLE = "color: #8a8f98"  # 신뢰도 "참고용" 글자
ROW_PX = 32  # 위반율 표 한 줄 높이
ACCOUNT_CHIP = """<div class="account-chip"><div class="account-name">{name}{role}</div><div class="account-agency">{agency}</div></div>"""
SORTS = {  # 표시 이름: (정렬 컬럼, 오름차순)
    "위반율 높은 순": ("violation_rate", False),
    "시간대 순": ("slot", True),
    "판정 대상 많은 순": ("n_target", False),
}
SIGNUP_INPUTS = {  # 가입 화면의 칸(위에서 아래 순서): st.text_input 인자. * 는 필수
    "email": {"label": "이메일 *", "type": "email", "validate": ""},  # 형식 검사는 signup_errors가 한다
    "password": {"label": f"비밀번호 * ({PASSWORD_MIN}자 이상)", "type": "password"},
    "confirm": {"label": "비밀번호 확인 *", "type": "password"},
    "name": {"label": "이름 *"},
    "agency": {"label": "소속 기관 *"},
    "department": {"label": "부서"},
    "job_position": {"label": "직급"},
    "phone": {"label": "휴대폰 번호 (- 없이 숫자만)", "type": "phone", "placeholder": "01011112222"},
}
APPROVAL_COLUMNS = {  # 가입 승인 관리 페이지의 칸(users 표의 칸: 표시 이름)
    "name": "이름", "email": "이메일", "agency": "소속 기관", "department": "부서", "job_position": "직급", "phone": "휴대폰 번호",
}
APPROVAL_WIDTHS = [2, 4, 3, 2, 2, 3, 1, 1]  # 위 여섯 칸 + 승인·거절 버튼
SIGNUP_DONE = "가입 신청이 접수됐습니다. 관리자 승인 후 로그인할 수 있습니다."


def load_judged(judged_dir: str) -> pd.DataFrame:
    """판정 CSV를 모두 합친다. 파일이 없으면 빈 표."""
    files = sorted(Path(judged_dir).glob("*.csv"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(p, encoding="utf-8-sig", dtype={"clip": str}) for p in files], ignore_index=True)
    df["date"] = pd.to_datetime(df["wall_time"]).dt.date
    df["clip"] = df["clip"].fillna("")
    return df


def apply_filters(df: pd.DataFrame, intersections: list[str], start: date, end: date) -> pd.DataFrame:
    return df[df["intersection"].isin(intersections) & (df["date"] >= start) & (df["date"] <= end)]


def pct(x: float) -> str:
    """정수 %. 0보다 크지만 반올림하면 0%가 되는 값은 "1% 미만"으로 적는다(위반 의심이 있는데 0%로 보이지 않게)."""
    return "1% 미만" if 0 < x < 0.005 else f"{x:.0%}"


def rate_views(table: pd.DataFrame, bad: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """위반율 표(rate_table)를 화면용과 내려받기용으로 만든다.
    화면용은 기획서 예시 화면대로 정수 %(위반율 0~100, 막대 칸), 내려받기용은 숫자로 둔다(위반율과 신뢰구간 하한·상한 0~1, 소수 넷째 자리)."""
    ci = list(zip(table["ci_low"], table["ci_high"]))
    view = pd.DataFrame({
        "교차로": table["intersection"].astype(str),
        "시간대": table["slot"],
        "판정 대상": table["n_target"],
        "위반 의심": table[f"n_{bad}"],
        "위반율": table["violation_rate"] * 100,
        "신뢰구간": [f"{lo * 100:.0f}~{hi * 100:.0f}%" if pd.notna(lo) else "-" for lo, hi in ci],
        "신뢰도": table["note"].replace("", "표본 충분"),
    })
    export = view.drop(columns="신뢰구간").assign(**{
        "교차로": "교차로 " + view["교차로"],  # 내려받는 파일의 값은 예전과 같게
        "위반율": table["violation_rate"].round(4),
        "95% 신뢰구간 하한": table["ci_low"].round(4),
        "95% 신뢰구간 상한": table["ci_high"].round(4),
    })[["교차로", "시간대", "판정 대상", "위반 의심", "위반율", "95% 신뢰구간 하한", "95% 신뢰구간 상한", "신뢰도"]]
    return view, export


def has_result(filtered: pd.DataFrame, name: str) -> bool:
    col, ok, bad, _ = KINDS[name]
    return col in filtered.columns and bool(filtered[col].isin([ok, bad]).any())


def summary(name: str, filtered: pd.DataFrame, min_n: int) -> None:
    """위반 유형 하나의 총괄 현황과 단속 우선 검토 시간대."""
    if not has_result(filtered, name):
        st.info(f"{name} 판정 결과가 없습니다.")
        return
    col, ok, bad, _ = KINDS[name]
    table = rate_table(filtered, min_n, col, ok, bad)
    n, k = int(table["n_target"].sum()), int(table[f"n_{bad}"].sum())
    low, high = wilson_ci(k, n)
    st.subheader(f"{name} 총괄 현황")
    with st.container(key=f"sum-{col}"):  # 세 칸을 이어 붙인다
        c1, c2, c3 = st.columns(3)
        c1.metric("판정 대상 우회전", f"{n:,}건", height="stretch")
        c2.metric("위반 의심", f"{k:,}건", height="stretch")
        c3.metric("위반율", pct(k / n), height="stretch")
    st.caption(f"※ 위반율 95% 신뢰구간 {low * 100:.0f}~{high * 100:.0f}%")

    priority(name, filtered, table, min_n)


def detail(name: str, filtered: pd.DataFrame, min_n: int, sort_name: str, red_mode: str) -> None:
    """위반 유형 하나의 교차로·시간대별 위반율 표와 내려받기."""
    col, ok, bad, file_name = KINDS[name]
    column, ascending = SORTS[sort_name]
    table = rate_table(filtered, min_n, col, ok, bad).sort_values(column, ascending=ascending)
    st.subheader(f"{name} 교차로·시간대별 현황 (총 {len(table)}건)")
    view, export = rate_views(table, bad)
    export = export.assign(**{
        "위반 유형": f"{name} ({red_mode} 모드)" if col == "red_result" else name,
        "자료 기간": f"{filtered['date'].min()} ~ {filtered['date'].max()}",
        "내려받은 시각": pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M"),
        "비고": EXPORT_NOTE,
    })
    st.dataframe(
        view.style.map(lambda v: "" if v == "표본 충분" else REFERENCE_STYLE, subset=["신뢰도"]),
        column_config={
            "위반율": st.column_config.ProgressColumn("위반율", min_value=0, max_value=100, format="%.0f%%", color=BAR_COLOR)
        },
        hide_index=True,
        width="stretch",
        height="content",
        row_height=ROW_PX,
    )
    st.caption(f"※ 신뢰구간은 Wilson 95%. '참고용'은 판정 대상 {min_n}건 미만으로, 단속 우선 검토 시간대에서 제외합니다.")
    with st.container(horizontal=True, horizontal_alignment="right"):
        st.download_button(
            "CSV 내려받기",
            export.to_csv(index=False).encode("utf-8-sig"),
            file_name,
            "text/csv",
            key=col,  # "전체"에서는 버튼이 둘이라 구분한다
        )


def overall(filtered: pd.DataFrame, min_n: int) -> None:
    """종합 단속 우선 검토 시간대(기술서 3.1): 어느 교차로를 몇 시에. 보고서 PDF와 같은 계산이고 위반 유형을 무엇으로 골라도 같다."""
    st.subheader("종합 단속 우선 검토 시간대")
    best = report_pdf.overall_slots(filtered, min_n)
    if best.empty:
        st.info(f"두 유형 모두 판정 대상이 {min_n}건 이상이면서 위반 의심이 있는 시간대가 없습니다.")
        return
    ui.grid(
        ["종합 순위", "교차로", "시간대", "미양보 위반율(순위)", "미양보 의심", "미정지 위반율(순위)", "미정지 의심"],
        [[f"{i}순위", str(r.intersection), r.slot, f"{pct(r.rate_ped_result)} ({r.rank_ped_result}위)", f"{r.n_ped_result:,}",
          f"{pct(r.rate_red_result)} ({r.rank_red_result}위)", f"{r.n_red_result:,}"] for i, r in enumerate(best.itertuples(), 1)],
        unit="건, %", strong_first=True,
        note=f"유형별 위반율 순위의 합이 작은 순(동점은 미양보 위반율 높은 순). 한 유형이라도 판정 대상이 {min_n}건 미만이면 제외합니다.",
    )


def vehicles(filtered: pd.DataFrame, min_n: int) -> None:
    """교차로·차종별 현황(기술서 3.1): 어느 교차로에서 어떤 차종의 위반율이 높은지. 보고서 PDF와 같은 계산이고 두 유형을 함께 보여 준다."""
    rows = report_pdf.site_vehicle_rows(filtered, min_n, pct)
    if not rows:  # 차종 칸이 없는 자료
        return
    st.subheader(f"위반율이 높은 교차로·차종 (유형별 상위 {TOP_N}개 이내)")
    top = report_pdf.site_vehicle_top(filtered, min_n, pct)
    if top:
        ui.grid(["구분", "순위", "교차로", "차종", "판정 대상", "위반 의심", "위반율"], top, unit="건, %", merge=True)
    else:
        st.info(f"판정 대상이 {min_n}건 이상이면서 위반 의심이 있는 교차로·차종이 없습니다.")
    st.subheader("교차로·차종별 현황")
    ui.grid(["교차로", "차종", "우회전", "미양보 대상", "미양보 의심", "미양보 위반율", "미정지 대상", "미정지 의심", "미정지 위반율"], rows,
            unit="건, %", note=f"* 는 참고용(판정 대상 {min_n}건 미만)입니다.", merge=True)


def priority(name: str, filtered: pd.DataFrame, table: pd.DataFrame, min_n: int) -> None:
    """위반 유형 하나의 단속 우선 검토 시간대 표와, 시간대마다 위반 의심 클립."""
    col, _, bad, _ = KINDS[name]
    st.subheader(f"{name} 단속 우선 검토 시간대 (상위 {TOP_N}개 이내)")
    top = top_slots(table, TOP_N)
    if top.empty:
        st.info(f"판정 대상이 {min_n}건 이상이면서 위반 의심이 있는 시간대가 없습니다.")
        return
    ui.grid(
        ["순위", "교차로", "시간대", "판정 대상", "위반 의심", "위반율"],
        [[str(i), str(r.intersection), r.slot, f"{r.n_target:,}", f"{getattr(r, f'n_{bad}'):,}", pct(r.violation_rate)]
         for i, r in enumerate(top.itertuples(), 1)],
        unit="건, %",
    )
    for area, (rank, r) in zip(st.columns(TOP_N), enumerate(top.itertuples(), 1)):
        clips = filtered[
            (filtered["intersection"] == r.intersection)
            & (filtered["slot"] == r.slot)
            & (filtered[col] == bad)
            & (filtered["clip"] != "")
        ].sort_values("wall_time")
        if clips.empty:
            continue
        labels = {p.clip: f"{pd.Timestamp(p.wall_time):%m-%d %H:%M:%S} {getattr(p, 'vehicle_type', '')}".strip() for p in clips.itertuples()}
        with area.expander(f"{rank}순위 영상 자료 ({len(labels)}개, 의심 {getattr(r, f'n_{bad}')}건 중)"):
            clip_player(labels, f"clip-{col}-{rank}")


def clip_player(labels: dict[str, str], key: str) -> None:
    """클립을 하나 골라 본다(한꺼번에 불러오지 않게). labels = {클립 경로: 표시 이름}"""
    clip = st.selectbox("발생 시각", list(labels), format_func=labels.get, key=key)
    if Path(clip).exists():
        st.video(clip, format="video/webm")  # eval.clips가 WebM(VP8)으로 저장한다
    else:
        st.caption(f"파일 없음: {clip}")


def all_clips(filtered: pd.DataFrame) -> None:
    """조회 조건에 맞는 클립 전부(표본, 기술서 3.4). 우선 시간대에 들지 않는 미양보 클립도 여기서 본다."""
    clips = filtered[filtered["clip"] != ""].sort_values(["intersection", "wall_time"])
    if clips.empty:
        return
    suspects = {name: filtered[col] == bad for name, (col, _, bad, _) in KINDS.items() if col in filtered}
    total = int(pd.concat(suspects, axis=1).any(axis=1).sum())
    labels = {}
    for p in clips.itertuples():
        kinds = "·".join(name for name, hit in suspects.items() if hit[p.Index])
        labels[p.clip] = f"{p.intersection} {pd.Timestamp(p.wall_time):%m-%d %H:%M:%S} {getattr(p, 'vehicle_type', '')} {kinds}"
    st.subheader(f"위반 의심 영상 자료 ({len(labels)}개, 의심 {total}건 중)")
    st.caption("※ 사람이 확인할 표본입니다. 미양보는 모두, 미정지는 우선 시간대에서 일부만 만듭니다.")
    clip_player(labels, "clip-all")


def report_button(filtered: pd.DataFrame, min_n: int, red_mode: str) -> None:
    """지금까지 쌓인 판정(현재 조회 조건)으로 분석 보고서 PDF를 만들어 내려받게 한다."""
    bar = st.container(horizontal=True, horizontal_alignment="right")
    if bar.button("보고서(PDF) 생성", type="primary", help="지금까지 쌓인 판정 중 현재 조회 조건(교차로·기간)에 맞는 것으로 만듭니다"):
        out = Path("outputs/reports") / f"report_{pd.Timestamp.now():%Y%m%d_%H%M%S}.pdf"
        st.session_state["report-pdf"] = str(report_pdf.build_report(filtered, out, min_n, "우회전 차량 위반 의심 누적 현황 보고", red_mode=red_mode))
    pdf = Path(st.session_state.get("report-pdf", ""))
    if pdf.is_file():
        bar.download_button(f"보고서 내려받기 ({pdf.name})", pdf.read_bytes(), pdf.name, "application/pdf")


def login_page() -> None:
    """로그인 전 화면: 로그인과 회원가입(신청 후 관리자가 승인해야 로그인된다)."""
    _, middle, _ = st.columns([1, 1.2, 1])
    login_tab, signup_tab = middle.container(key="login-box").tabs(["로그인", "회원가입"], key="auth-tab", on_change="rerun")
    with login_tab:
        if "auth-notice" in st.session_state:  # 가입 신청 직후 한 번 보여 준다
            st.success(st.session_state.pop("auth-notice"))
        with st.form("login"):
            email = st.text_input("이메일", key="login-email")
            password = st.text_input("비밀번호", type="password", key="login-pw")
            if st.form_submit_button("로그인", key="login-submit", type="primary", width="stretch"):
                user = db.get_user(email.strip())
                if user is None or not verify_password(password, user["password_hash"]):
                    st.error("이메일 또는 비밀번호가 맞지 않습니다.")
                elif user["status"] != "approved":
                    st.warning("관리자 승인 대기 중입니다.")
                else:
                    st.session_state["user"] = {k: user[k] for k in ("id", "name", "agency", "role")}
                    st.rerun()
    with signup_tab:
        signup_inputs()


def signup_values() -> tuple[dict, str, str]:
    """가입 칸에 지금 적힌 값: (users 표에 넣을 칸들, 비밀번호, 비밀번호 확인)."""
    state = st.session_state
    user = {name: state.get(f"signup-{name}", "").strip() for name in db.USER_FIELDS}
    return user, state.get("signup-password", ""), state.get("signup-confirm", "")


def signup_inputs() -> None:
    """가입 칸들. 타이핑을 멈추면 바로 검사해 틀린 칸 아래에 이유를 적는다(live라 st.form 밖에 둔다)."""
    user, password, confirm = signup_values()
    values = {**user, "password": password, "confirm": confirm}
    errors = signup_errors(user, password, confirm)
    if "email" not in errors and db.get_user(user["email"]):
        errors["email"] = "이미 가입된 이메일입니다."
    tried = st.session_state.get("signup-tried", False)  # 가입 신청을 누른 뒤에는 빈 필수 칸에도 이유를 적는다

    st.caption("가입을 신청하면 관리자가 확인한 뒤 승인합니다. * 는 필수입니다.")
    for name, options in SIGNUP_INPUTS.items():
        st.text_input(key=f"signup-{name}", live=True, max_chars=db.USER_FIELDS.get(name), **options)
        if name in errors and (values[name] or tried):
            st.caption(f":red[{errors[name]}]")
    st.button("가입 신청", key="signup-submit", on_click=submit_signup)


def submit_signup() -> None:
    """가입 신청 버튼: 다시 검사하고, 통과하면 신청을 넣고 칸을 비운 뒤 로그인 탭으로 보낸다."""
    state = st.session_state
    user, password, confirm = signup_values()
    if signup_errors(user, password, confirm) or not db.add_user(user, password):
        state["signup-tried"] = True
        return
    for key in [key for key in state if key.startswith("signup-")]:
        del state[key]
    state["auth-tab"] = "로그인"
    state["auth-notice"] = SIGNUP_DONE


def account_bar(pending: list[dict] | None) -> None:
    """상단 띠 오른쪽: 로그인한 사람의 이름·소속기관과 로그아웃. pending(승인 대기 목록)은 관리자일 때만 준다."""
    user = st.session_state["user"]
    name, agency = escape(user["name"]), escape(user["agency"])  # 가입자가 적은 글자라 태그로 나가지 않게
    role = '<span class="account-role">관리자</span>' if user["role"] == "admin" else ""
    with st.container(key="account", horizontal=True, horizontal_alignment="right", vertical_alignment="center"):
        st.html(ACCOUNT_CHIP.format(name=name, role=role, agency=agency), width="content")
        if pending is not None:  # 관리자: 대시보드 <-> 가입 승인 관리 페이지
            approving = st.session_state.get("view") == "approval"
            if st.button("통계 화면" if approving else f"가입 승인 관리 ({len(pending)})", key="view-toggle"):
                st.session_state["view"] = "dashboard" if approving else "approval"
                st.rerun()
        if st.button("로그아웃", key="logout"):
            st.session_state.pop("view", None)
            del st.session_state["user"]
            st.rerun()


def approval_page(pending: list[dict]) -> None:
    """관리자 전용 가입 승인 관리 페이지: 신청자가 적은 정보를 표로 보고 승인하거나 거절한다."""
    st.subheader(f"승인 대기 (총 {len(pending)}명)")
    if not pending:
        st.info("승인을 기다리는 가입 신청이 없습니다.")
        return
    rows = st.container(key="approval-list")
    for column, label in zip(rows.columns(APPROVAL_WIDTHS), APPROVAL_COLUMNS.values()):
        column.markdown(f"**{label}**")
    for p in pending:
        *cells, approve, reject = rows.columns(APPROVAL_WIDTHS, vertical_alignment="center")
        for cell, name in zip(cells, APPROVAL_COLUMNS):  # 신청자가 적은 글자라 서식 없이 그대로 보여 준다
            cell.text((format_phone(p[name]) if name == "phone" else p[name]) or "-")
        if approve.button("승인", key=f"approve-{p['id']}"):
            db.approve_user(p["id"])
            st.rerun()
        if reject.button("거절", key=f"reject-{p['id']}"):
            db.reject_user(p["id"])
            st.rerun()


def main(judged_dir: str = "outputs/judged", config: str = "configs/eval.yaml", use_db: bool = False) -> None:
    st.set_page_config(page_title=ui.SYSTEM, page_icon=":material/table_chart:", layout="wide")
    ui.frame()
    if use_db:
        if "user" not in st.session_state:
            ui.page_title("사용자 로그인")
            st.warning(NOTICE)
            login_page()
            return
        pending = db.pending_users() if st.session_state["user"]["role"] == "admin" else None
        account_bar(pending)
        if pending is not None and st.session_state.get("view") == "approval":
            ui.page_title("가입 승인 관리", "시스템 관리")
            st.warning(NOTICE)
            approval_page(pending)
            return
    page = st.sidebar.radio("업무 메뉴", ["위반 통계", "영상 분석"], key="page")
    ui.page_title(page, "우회전 위반 의심 업무")
    st.warning(NOTICE)
    red_mode = st.sidebar.segmented_control(
        "판정 모드", list(RED_MODES), default="보통", required=True, key="red-mode",
        help="경미 정지 = 정지선은 넘었지만 횡단보도 전에 선 경우. 보통: 정지로 봅니다. 엄격: 위반 의심으로 봅니다.",
    )
    if page == "영상 분석":
        workflow.analysis_page(red_mode)
        return
    if use_db:
        df = db.fetch_events()
    else:
        df = load_judged(judged_dir)
    if df.empty:
        st.info("판정 결과가 없습니다. 영상 분석을 마친 뒤 조회하십시오.")
        return
    with open(config, encoding="utf-8") as fp:
        min_n = yaml.safe_load(fp)["min_n_reference"]

    with st.container(key="search"):  # 조회 조건
        f1, f2 = st.columns([3, 2])
        f3, f4 = st.columns([3, 2])
    all_sites = sorted(df["intersection"].unique())
    sites = f1.multiselect("교차로", all_sites, default=all_sites)
    period = f2.date_input("기간", (df["date"].min(), df["date"].max()))
    kind = f3.selectbox("위반 유형", [ALL_KINDS, *KINDS])
    sort_name = f4.selectbox("정렬", list(SORTS))
    if len(period) != 2:  # 날짜 범위를 고르는 중
        return

    filtered = apply_red_mode(apply_filters(df, sites, period[0], period[1]), red_mode)
    if filtered.empty:
        st.info("조회 조건에 해당하는 자료가 없습니다.")
        return
    report_button(filtered, min_n, red_mode)
    overall(filtered, min_n)
    names = list(KINDS) if kind == ALL_KINDS else [kind]
    for area, name in zip(st.columns(len(names)), names):  # "전체"는 좌우로 나란히
        with area:
            summary(name, filtered, min_n)
    vehicles(filtered, min_n)
    all_clips(filtered)
    shown = [name for name in names if has_result(filtered, name)]
    for area, name in zip(st.tabs([f"{name} 현황" for name in shown]) if len(shown) > 1 else [st.container()], shown):  # 긴 표는 유형별 탭으로
        with area:
            detail(name, filtered, min_n, sort_name, red_mode)
    ui.footer("자동 판정 결과(사람 확인 전)입니다. 영상 자료는 얼굴·번호판을 블러한 것만 보여 줍니다.")


if __name__ == "__main__":
    main(use_db=db.configured())
