# 담당: 육주영
"""위반율 대시보드 (Streamlit, 읽기 전용). outputs/judged/*.csv를 pandas로 모아 보여 준다.
실행: streamlit run dashboard/app.py

화면(기술서 3.2, 배치는 기획서 3-4절 예시 화면): 화면 위 필터(교차로·기간·위반 유형·정렬), 요약 카드, 교차로 x 시간대 위반율 표(Wilson 95% 신뢰구간, configs/eval.yaml의 min_n_reference 미만 "참고용"),
단속 우선 검토 시간대 TOP 3(표본 충분, 위반 의심 1건 이상)와 클립 보기, 현재 필터의 표를 CSV로 내려받기.
"위반 의심 통계이며 실제 단속은 별도 확인이 필요하다" 문구를 항상 표시한다.
위반 유형 "전체"는 두 유형을 나란히 보여 주고 위반율을 하나로 합치지 않는다(분모가 다르다).
레포 루트 .env에 DB 접속 값이 있으면(dashboard/db.py) 로그인한 사람에게만 보여 주고 자료를 MySQL에서 읽는다. 가입은 신청 후 관리자 승인(관리자 전용 가입 승인 관리 페이지).
왼쪽 사이드바에서 "영상 분석" 화면(영상 등록 → 영역 설정 → 분석 → 리포트, dashboard/workflow.py)으로 바꾼다.
위반 통계 화면의 "리포트 PDF 만들기"는 지금까지 쌓인 판정(현재 필터)으로 리포트를 만든다(dashboard/report_pdf.py).
"""
import sys
from datetime import date
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # streamlit run은 레포 루트를 import 경로에 넣지 않는다

import pandas as pd
import streamlit as st
import yaml

from dashboard import db, report_pdf, workflow
from dashboard.auth import PASSWORD_MIN, format_phone, signup_errors, verify_password
from eval.report import rate_table, wilson_ci

NOTICE = report_pdf.NOTICE
EXPORT_NOTE = f"자동 판정 결과(사람 확인 전). {NOTICE}"
TOP_N = 3
KINDS = {  # 표시 이름: (판정 컬럼, 위반 아님, 위반 의심, 내려받기 파일 이름)
    "보행자 미양보": ("ped_result", "yield", "no_yield", "violation_rate.csv"),
    "적색 미정지": ("red_result", "stopped", "not_stopped", "red_rate.csv"),
}
ALL_KINDS = "전체"
BAR_COLOR = "#e05252"  # 위반율 막대
BADGE_ENOUGH = "background-color: #d7f0df; color: #1a7f43"  # 신뢰도 배지
BADGE_REFERENCE = "background-color: #fdecc8; color: #a05a00"
# 기획서 3-4절 예시 화면의 색. Streamlit이 key를 st-key-<이름> 클래스로 내보내는 것에 기댄다(버전을 올려 이름이 바뀌면 색만 빠진다)
STYLE = """<style>
h1 { color: #1f3a68 !important; }
[class*="st-key-bad-"] [data-testid="stMetricValue"] { color: #c62828; }
[class*="st-key-top-"] { background: #fdf1f0; border: 0; border-left: 5px solid #d32f2f; border-radius: 6px; }
.account-chip { display: flex; align-items: center; gap: 10px; }
.account-avatar { flex: none; width: 38px; height: 38px; border-radius: 50%; background: #1f3a68; color: #fff;
  display: flex; align-items: center; justify-content: center; font-weight: 700; }
.account-name { font-weight: 700; line-height: 1.3; }
.account-agency { font-size: 0.8rem; line-height: 1.3; color: #6b7280; }
.account-role { margin-left: 6px; padding: 1px 8px; border-radius: 999px; background: #e8eef7; color: #1f3a68;
  font-size: 0.7rem; font-weight: 600; vertical-align: middle; }
</style>"""
ACCOUNT_CHIP = """<div class="account-chip"><div class="account-avatar">{initial}</div>
<div><div class="account-name">{name}{role}</div><div class="account-agency">{agency}</div></div></div>"""
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


def top_slots(table: pd.DataFrame, n: int = TOP_N) -> pd.DataFrame:
    """표본이 충분하고(참고용이 아니고) 위반 의심이 한 건이라도 있는 칸 중 위반율 높은 순 n개."""
    return table[(table["note"] == "") & (table["violation_rate"] > 0)].sort_values("violation_rate", ascending=False).head(n)


def pct(x: float) -> str:
    """정수 %. 0보다 크지만 반올림하면 0%가 되는 값은 "1% 미만"으로 적는다(위반 의심이 있는데 0%로 보이지 않게)."""
    return "1% 미만" if 0 < x < 0.005 else f"{x:.0%}"


def rate_views(table: pd.DataFrame, bad: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """위반율 표(rate_table)를 화면용과 내려받기용으로 만든다.
    화면용은 기획서 예시 화면대로 정수 %(위반율 0~100, 막대 칸), 내려받기용은 숫자로 둔다(위반율과 신뢰구간 하한·상한 0~1, 소수 넷째 자리)."""
    ci = list(zip(table["ci_low"], table["ci_high"]))
    view = pd.DataFrame({
        "교차로": "교차로 " + table["intersection"].astype(str),
        "시간대": table["slot"],
        "판정 대상": table["n_target"],
        "위반 의심": table[f"n_{bad}"],
        "위반율": table["violation_rate"] * 100,
        "신뢰구간": [f"{lo * 100:.0f}~{hi * 100:.0f}%" if pd.notna(lo) else "-" for lo, hi in ci],
        "신뢰도": table["note"].replace("", "표본 충분"),
    })
    export = view.drop(columns="신뢰구간").assign(**{
        "위반율": table["violation_rate"].round(4),
        "95% 신뢰구간 하한": table["ci_low"].round(4),
        "95% 신뢰구간 상한": table["ci_high"].round(4),
    })[["교차로", "시간대", "판정 대상", "위반 의심", "위반율", "95% 신뢰구간 하한", "95% 신뢰구간 상한", "신뢰도"]]
    return view, export


def render(name: str, filtered: pd.DataFrame, min_n: int, sort_name: str) -> None:
    """위반 유형 하나의 요약 카드, 위반율 표, 내려받기, 단속 우선 검토 시간대. 배치는 기획서 3-4절 예시 화면을 따른다."""
    col, ok, bad, file_name = KINDS[name]
    if col not in filtered.columns or not filtered[col].isin([ok, bad]).any():
        st.info(f"{name} 판정 결과가 아직 없다.")
        return
    table = rate_table(filtered, min_n, col, ok, bad)
    column, ascending = SORTS[sort_name]
    table = table.sort_values(column, ascending=ascending)

    n, k = int(table["n_target"].sum()), int(table[f"n_{bad}"].sum())
    low, high = wilson_ci(k, n)
    c1, c2, c3 = st.columns(3)
    c1.metric("판정 대상 우회전", f"{n}건", border=True, height="stretch")  # 세 카드 높이를 맞춘다
    c2.container(key=f"bad-n-{col}", height="stretch").metric("위반 의심", f"{k}건", border=True, height="stretch")
    c3.container(key=f"bad-rate-{col}").metric(
        "위반율 (신뢰구간)", pct(k / n), f"{low * 100:.0f}~{high * 100:.0f}%", delta_color="off", delta_arrow="off", border=True
    )

    view, export = rate_views(table, bad)
    export = export.assign(**{
        "위반 유형": name,
        "자료 기간": f"{filtered['date'].min()} ~ {filtered['date'].max()}",
        "내려받은 시각": pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M"),
        "비고": EXPORT_NOTE,
    })
    st.dataframe(
        view.style.map(lambda v: BADGE_ENOUGH if v == "표본 충분" else BADGE_REFERENCE, subset=["신뢰도"]),
        column_config={
            "위반율": st.column_config.ProgressColumn("위반율", min_value=0, max_value=100, format="%.0f%%", color=BAR_COLOR)
        },
        hide_index=True,
        width="stretch",
    )
    st.caption(f"신뢰구간은 Wilson 95%. 신뢰도 '참고용' = 판정 대상 {min_n}건 미만. 단속 우선 검토 시간대 계산에서 뺀다.")
    st.download_button(
        "현재 표를 CSV로 내려받기",
        export.to_csv(index=False).encode("utf-8-sig"),
        file_name,
        "text/csv",
        key=col,  # "전체"에서는 버튼이 둘이라 구분한다
    )

    st.subheader(f"단속 우선 검토 시간대 TOP {TOP_N} (표본 충분한 시간대 중)")
    top = top_slots(table)
    if top.empty:
        st.info(f"판정 대상이 {min_n}건 이상이면서 위반 의심이 있는 시간대가 아직 없다.")
        return
    for area, (rank, r) in zip(st.columns(TOP_N), enumerate(top.itertuples(), 1)):
        clips = filtered[
            (filtered["intersection"] == r.intersection)
            & (filtered["slot"] == r.slot)
            & (filtered[col] == bad)
            & (filtered["clip"] != "")
        ]["clip"]
        with area.container(border=True, key=f"top-{col}-{rank}"):
            st.markdown(f"**:red[{rank}. 교차로 {r.intersection} {r.slot}]**")
            st.markdown(f"위반율 {pct(r.violation_rate)} · 위반 의심 {getattr(r, f'n_{bad}')}건")
            with st.expander(f"클립 보기 ▶ ({len(clips)}개)"):
                for clip in clips:
                    if Path(clip).exists():
                        st.video(clip, format="video/webm")  # eval.clips가 WebM(VP8)으로 저장한다
                    else:
                        st.caption(f"파일 없음: {clip}")


def report_button(filtered: pd.DataFrame, min_n: int) -> None:
    """지금까지 쌓인 판정(현재 필터)으로 분석 리포트 PDF를 만들어 내려받게 한다."""
    c1, c2 = st.columns([1, 3], vertical_alignment="center")
    if c1.button("리포트 PDF 만들기", help="지금까지 쌓인 판정 중 현재 필터(교차로·기간)에 맞는 것으로 만듭니다"):
        out = Path("outputs/reports") / f"report_{pd.Timestamp.now():%Y%m%d_%H%M%S}.pdf"
        st.session_state["report-pdf"] = str(report_pdf.build_report(filtered, out, min_n, "우회전 위반 의심 누적 리포트"))
    pdf = Path(st.session_state.get("report-pdf", ""))
    if pdf.is_file():
        c2.download_button(f"내려받기 ({pdf.name})", pdf.read_bytes(), pdf.name, "application/pdf")


def login_page() -> None:
    """로그인 전 화면: 로그인과 회원가입(신청 후 관리자가 승인해야 로그인된다)."""
    _, middle, _ = st.columns(3)
    login_tab, signup_tab = middle.tabs(["로그인", "회원가입"], key="auth-tab", on_change="rerun")
    with login_tab:
        if "auth-notice" in st.session_state:  # 가입 신청 직후 한 번 보여 준다
            st.success(st.session_state.pop("auth-notice"))
        with st.form("login"):
            email = st.text_input("이메일", key="login-email")
            password = st.text_input("비밀번호", type="password", key="login-pw")
            if st.form_submit_button("로그인", key="login-submit"):
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


def account_bar(area, pending: list[dict] | None) -> None:
    """화면 오른쪽 위: 로그인한 사람의 이름·소속기관(프로필 칩)과 로그아웃. pending(승인 대기 목록)은 관리자일 때만 준다."""
    user = st.session_state["user"]
    name, agency = escape(user["name"]), escape(user["agency"])  # 가입자가 적은 글자라 태그로 나가지 않게
    role = '<span class="account-role">관리자</span>' if user["role"] == "admin" else ""
    with area.container(horizontal=True, horizontal_alignment="right", vertical_alignment="center"):
        st.html(ACCOUNT_CHIP.format(initial=escape(user["name"][:1]), name=name, role=role, agency=agency), width="content")
        if pending is not None:  # 관리자: 대시보드 <-> 가입 승인 관리 페이지
            approving = st.session_state.get("view") == "approval"
            if st.button("대시보드" if approving else f"가입 승인 관리 ({len(pending)})", key="view-toggle"):
                st.session_state["view"] = "dashboard" if approving else "approval"
                st.rerun()
        if st.button("로그아웃", key="logout"):
            st.session_state.pop("view", None)
            del st.session_state["user"]
            st.rerun()


def approval_page(pending: list[dict]) -> None:
    """관리자 전용 가입 승인 관리 페이지: 신청자가 적은 정보를 표로 보고 승인하거나 거절한다."""
    st.subheader(f"가입 승인 관리 (대기 {len(pending)}명)")
    if not pending:
        st.info("승인을 기다리는 가입 신청이 없습니다.")
        return
    for column, label in zip(st.columns(APPROVAL_WIDTHS), APPROVAL_COLUMNS.values()):
        column.markdown(f"**{label}**")
    for p in pending:
        *cells, approve, reject = st.columns(APPROVAL_WIDTHS, vertical_alignment="center")
        for cell, name in zip(cells, APPROVAL_COLUMNS):  # 신청자가 적은 글자라 서식 없이 그대로 보여 준다
            cell.text((format_phone(p[name]) if name == "phone" else p[name]) or "-")
        if approve.button("승인", key=f"approve-{p['id']}"):
            db.approve_user(p["id"])
            st.rerun()
        if reject.button("거절", key=f"reject-{p['id']}"):
            db.reject_user(p["id"])
            st.rerun()


def main(judged_dir: str = "outputs/judged", config: str = "configs/eval.yaml", use_db: bool = False) -> None:
    st.set_page_config(page_title="우회전 위반 의심 통계", layout="wide")
    st.markdown(STYLE, unsafe_allow_html=True)
    logged_in = use_db and "user" in st.session_state
    # 로그인한 상태에서만 제목 줄 오른쪽 끝에 계정 표시를 둔다
    title_area, account_area = st.columns([3, 2], vertical_alignment="center") if logged_in else (st.container(), None)
    title_area.title("우회전 위반율: 교차로·시간대별")
    st.warning(NOTICE)

    if use_db:
        if not logged_in:
            login_page()
            return
        pending = db.pending_users() if st.session_state["user"]["role"] == "admin" else None
        account_bar(account_area, pending)
        if pending is not None and st.session_state.get("view") == "approval":
            approval_page(pending)
            return
    if st.sidebar.radio("화면", ["위반 통계", "영상 분석"], key="page") == "영상 분석":
        workflow.analysis_page()
        return
    if use_db:
        df = db.fetch_events()
    else:
        df = load_judged(judged_dir)
    if df.empty:
        where = "`python -m dashboard.db import <judged 폴더>`로 넣은" if use_db else f"`{judged_dir}`에 judged CSV를 만든"
        st.info(f"판정 결과가 없다. {where} 뒤 다시 연다.")
        return
    with open(config, encoding="utf-8") as fp:
        min_n = yaml.safe_load(fp)["min_n_reference"]

    f1, f2, f3, f4 = st.columns(4)  # 필터는 화면 위 한 줄
    all_sites = sorted(df["intersection"].unique())
    sites = f1.multiselect("교차로", all_sites, default=all_sites)
    period = f2.date_input("기간", (df["date"].min(), df["date"].max()))
    kind = f3.selectbox("위반 유형", [*KINDS, ALL_KINDS])
    sort_name = f4.selectbox("정렬", list(SORTS))
    if len(period) != 2:  # 날짜 범위를 고르는 중
        return

    filtered = apply_filters(df, sites, period[0], period[1])
    if filtered.empty:
        st.info("이 필터에 해당하는 이벤트가 없다.")
        return
    report_button(filtered, min_n)
    names = list(KINDS) if kind == ALL_KINDS else [kind]
    for area, name in zip(st.columns(len(names)), names):  # "전체"는 좌우로 나란히
        with area:
            if len(names) > 1:
                st.header(name)
            render(name, filtered, min_n, sort_name)
    st.caption("클립은 얼굴·번호판을 블러한 영상만 보여 줍니다.")


if __name__ == "__main__":
    main(use_db=db.configured())
