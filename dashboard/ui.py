# 담당: 장범수
"""대시보드 화면의 공통 모양(행정 업무시스템형): 남색 상단 띠, 왼쪽 업무 메뉴, 화면 경로, 라벨 칸이 있는 조회 조건, 괘선 표.
색·글꼴·모서리의 기본값은 .streamlit/config.toml이고, 여기서는 배치만 CSS로 맞춘다.
Streamlit의 data-testid와 key(st-key-<이름>)에 기댄다(버전을 올려 이름이 바뀌면 모양만 빠지고 동작은 그대로다).
"""
from html import escape

import streamlit as st

SYSTEM = "우회전 위반 의심 통계 시스템"
STYLE = """<style>
/* 상단 띠 */
.gov-band { position: fixed; top: 0; left: 0; right: 0; height: 46px; z-index: 999980; background: #1f3864; color: #fff;
  display: flex; align-items: center; padding: 0 20px; font-size: 1.15rem; font-weight: 700; border-bottom: 3px solid #8a9bb8; }
[data-testid="stHeader"] { background: transparent; height: 46px; min-height: 46px; z-index: 999990; pointer-events: none; }
[data-testid="stHeader"] button { pointer-events: auto; color: #fff; }
[data-testid="stSidebar"] { top: 46px; height: calc(100vh - 46px); min-width: 260px; max-width: 260px; border-right: 1px solid #c3cad4; }
[data-testid="stSidebarHeader"] { display: none; }
[data-testid="stSidebarUserContent"] { padding: 14px 14px 24px; }
[data-testid="stSidebarContent"] { padding: 0; }
[data-testid="stMainBlockContainer"] { padding: 60px 24px 24px; max-width: 1500px; }
[data-testid="stMain"] [data-testid="stVerticalBlock"] { gap: 0.6rem; }
/* 왼쪽 업무 메뉴 */
[data-testid="stSidebar"] [data-testid="stRadio"] > label { width: 100%; min-height: 34px; background: #1f3864; padding: 7px 12px; margin: 0; }
[data-testid="stSidebar"] [data-testid="stRadio"] > label p { color: #fff; font-weight: 700; }
[data-testid="stSidebar"] :is([data-testid="stElementContainer"], [data-testid="stRadio"], [data-testid="stRadioGroup"]) { width: 100% !important; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.5rem; }
[data-testid="stRadioGroup"] { gap: 0; border: 1px solid #c3cad4; border-top: 0; background: #fff; }
[data-testid="stRadioGroup"] > div { width: 100%; }
[data-testid="stRadioOption"] { width: 100%; padding: 8px 12px; margin: 0; border-bottom: 1px solid #e2e6ec; }
[data-testid="stRadioOption"] > div > div:not([data-testid]) { display: none; }
[data-testid="stRadioOption"]:has(input:checked) { background: #dfe6f1; box-shadow: inset 4px 0 0 #1f3864; }
[data-testid="stRadioOption"]:has(input:checked) p { font-weight: 700; color: #1f3864; }
[data-testid="stSidebar"] h3 { border-left: 0 !important; border-bottom: 1px solid #8a9bb8; padding: 0 0 4px 2px !important; font-size: 0.95rem !important; }
/* 화면 제목과 경로 */
.gov-title { display: flex; justify-content: space-between; align-items: flex-end; border-bottom: 2px solid #1f3864; padding-bottom: 6px; margin-bottom: 1rem; }
.gov-title b { font-size: 1.3rem; color: #1b1b1b; }
.gov-title span { font-size: 0.82rem; color: #5b6470; }
[data-testid="stHeading"] [data-testid="stMarkdownContainer"] { margin-bottom: 0 !important; }
[data-testid="stMain"] [data-testid="stHeading"] { padding-top: 12px; }
[data-testid="stMain"] h3 { font-size: 1.05rem !important; font-weight: 700; padding: 0 0 0 9px !important; margin: 0 !important;
  border-left: 4px solid #1f3864; line-height: 1.35; }
/* 안내 문구 */
[data-testid="stAlertContainer"] { background: #f7f8fa; border: 1px solid #c3cad4; color: #333; padding: 7px 12px; }
[data-testid="stAlertContentWarning"] p::before { content: "※ "; }
[data-testid="stAlertContentError"] { color: #b3261e; }
/* 조회 조건·입력 칸: 왼쪽 회색 라벨 칸 + 오른쪽 입력 */
.st-key-search, .st-key-register { border-top: 2px solid #1f3864; border-bottom: 1px solid #8a9bb8; gap: 0 !important; }
.st-key-search [data-testid="stHorizontalBlock"], .st-key-register [data-testid="stHorizontalBlock"] { gap: 0 !important; }
.st-key-search [data-testid="stVerticalBlock"], .st-key-register [data-testid="stVerticalBlock"] { gap: 0 !important; }
:is(.st-key-search, .st-key-register) :is([data-testid="stMultiSelect"], [data-testid="stSelectbox"], [data-testid="stDateInput"],
  [data-testid="stTimeInput"], [data-testid="stTextInput"], [data-testid="stButtonGroup"]) {
  display: flex; flex-direction: row; align-items: center; min-height: 44px; border-top: 1px solid #dde2e9; padding-right: 10px; }
:is(.st-key-search, .st-key-register) [data-testid="stWidgetLabel"] { flex: 0 0 110px; align-self: stretch; min-height: 44px; margin: 0 10px 0 0;
  padding: 4px 10px; background: #eef1f6; display: flex; align-items: center; }
.st-key-register [data-testid="stWidgetLabel"] { flex-basis: 170px; }
:is(.st-key-search, .st-key-register) [data-testid="stWidgetLabel"] p { font-weight: 600; font-size: 0.9rem; }
:is(.st-key-search, .st-key-register) [data-testid="stWidgetLabel"] ~ div { flex: 1 1 auto; min-width: 0; }
:is(.st-key-search, .st-key-register) [data-testid="stElementContainer"], :is(.st-key-search, .st-key-register) [data-testid="stButtonGroup"] { width: 100% !important; }
:is(.st-key-search, .st-key-register) [data-testid="stWidgetLabel"] ~ div[data-testid="stRadioGroup"] { flex: 1; }
[data-testid="stMultiSelectTagsContainer"] > span { background: #e6ebf2 !important; color: #222 !important; border: 1px solid #b8c2d0; }
[data-testid="stMultiSelectTagsContainer"] > span * { color: #222 !important; background: transparent !important; }
/* 괘선 표 */
.gov-unit { text-align: right; font-size: 0.8rem; color: #5b6470; margin-bottom: 3px; }
.gov-grid { width: 100%; border-collapse: collapse; border-top: 2px solid #1f3864; border-bottom: 1px solid #8a9bb8; font-size: 0.9rem; }
.gov-grid th { background: #e6ebf2; font-weight: 700; padding: 7px 8px; border: 1px solid #c3cad4; text-align: center; white-space: nowrap; }
.gov-grid td { padding: 7px 8px; border: 1px solid #d5dbe4; text-align: center; }
.gov-grid th:first-child, .gov-grid td:first-child { border-left: 0; }
.gov-grid th:last-child, .gov-grid td:last-child { border-right: 0; }
.gov-grid tr.first td { font-weight: 700; background: #f3f6fb; }
.gov-note { font-size: 0.82rem; color: #5b6470; margin-top: 4px; }
/* 총괄 현황: 이어 붙인 칸 */
[class*="st-key-sum-"] [data-testid="stHorizontalBlock"] { gap: 0 !important; border-top: 2px solid #1f3864; border-bottom: 1px solid #8a9bb8; }
[class*="st-key-sum-"] [data-testid="stMetric"] { padding: 0 0 8px; border: 0; border-right: 1px solid #d5dbe4; text-align: center; }
[class*="st-key-sum-"] [data-testid="stColumn"]:last-child [data-testid="stMetric"] { border-right: 0; }
[class*="st-key-sum-"] [data-testid="stMetricLabel"] { width: 100%; background: #e6ebf2; border-bottom: 1px solid #c3cad4; padding: 6px 0; margin-bottom: 6px; }
[class*="st-key-sum-"] [data-testid="stMetricLabel"] { display: flex; justify-content: center; }
[class*="st-key-sum-"] [data-testid="stMetricLabel"] p { font-weight: 700; }
[class*="st-key-sum-"] [data-testid="stMetricValue"], [class*="st-key-sum-"] [data-testid="stMetricValue"] * { width: 100%; text-align: center; font-weight: 700; }
[class*="st-key-sum-"] [data-testid="stMetricDelta"] { width: 100%; justify-content: center; background: transparent; }
/* 등록 영상·가입 신청 목록 */
:is(.st-key-video-list, .st-key-approval-list) { border-top: 2px solid #1f3864; border-bottom: 1px solid #8a9bb8; gap: 0 !important; }
:is(.st-key-video-list, .st-key-approval-list) [data-testid="stHorizontalBlock"] { border-bottom: 1px solid #dde2e9; padding: 5px 10px; min-height: 40px; align-items: center; }
:is(.st-key-video-list, .st-key-approval-list) > div:first-child [data-testid="stHorizontalBlock"] { background: #e6ebf2; }
/* 계정·로그인·버튼 */
.st-key-account { position: fixed; top: 0; right: 20px; height: 43px; width: auto !important; z-index: 999995; align-items: center; }
.st-key-account button { min-height: 26px; padding: 0 10px; background: transparent !important; border: 1px solid #9fb0cc !important; }
.st-key-account button, .st-key-account button p { color: #fff; font-size: 0.82rem; }
.account-chip { display: flex; gap: 8px; align-items: center; color: #fff; font-size: 0.88rem; white-space: nowrap; }
.account-name { font-weight: 700; }
.account-agency { color: #cfd8e6; }
.account-role { margin-left: 6px; padding: 0 6px; border: 1px solid #9fb0cc; font-size: 0.75rem; font-weight: 400; }
.st-key-login-box { border: 1px solid #c3cad4; border-top: 3px solid #1f3864; padding: 22px 26px; margin-top: 24px; background: #fff; }
.st-key-login-box [data-testid="stForm"] { border: 0; padding: 0; }
[data-testid="stMain"] button[kind="secondary"], [data-testid="stMain"] button[kind="secondaryFormSubmit"] { background: #f5f6f8; border-color: #9aa5b5; }
.gov-steps { display: flex; margin-bottom: 1rem; border: 1px solid #c3cad4; background: #f7f8fa; font-size: 0.9rem; }
.gov-steps div { flex: 1; padding: 8px 0; text-align: center; border-right: 1px solid #c3cad4; }
.gov-steps div:last-child { border-right: 0; }
.gov-foot { border-top: 1px solid #c3cad4; margin-top: 18px; padding-top: 8px; font-size: 0.8rem; color: #5b6470; }
</style>"""


def frame() -> None:
    """모든 화면에 공통인 상단 띠와 모양."""
    st.markdown(STYLE + f'<div class="gov-band">{SYSTEM}</div>', unsafe_allow_html=True)


def page_title(title: str, *path: str) -> None:
    """화면 제목(왼쪽)과 화면 경로(오른쪽)."""
    st.markdown(f'<div class="gov-title"><b>{title}</b><span>{" &gt; ".join(["홈", *path, title])}</span></div>', unsafe_allow_html=True)


def grid(columns: list[str], rows: list[list[str]], unit: str = "", note: str = "", strong_first: bool = False) -> None:
    """괘선 표. 칸 값은 사용자가 적은 글자(교차로 이름 등)일 수 있어 태그로 나가지 않게 한다. strong_first = 첫 줄 강조."""
    head = "".join(f"<th>{escape(c)}</th>" for c in columns)
    body = "".join(
        f'<tr{" class=first" if strong_first and i == 0 else ""}>' + "".join(f"<td>{escape(v)}</td>" for v in row) + "</tr>"
        for i, row in enumerate(rows)
    )
    above = f'<div class="gov-unit">(단위: {escape(unit)})</div>' if unit else ""
    below = f'<div class="gov-note">※ {escape(note)}</div>' if note else ""
    st.html(f'{above}<table class="gov-grid"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>{below}')


def steps(names: list[str]) -> None:
    """업무 절차 줄."""
    st.markdown('<div class="gov-steps">' + "".join(f"<div>{i}. {n}</div>" for i, n in enumerate(names, 1)) + "</div>", unsafe_allow_html=True)


def footer(text: str) -> None:
    st.markdown(f'<div class="gov-foot">{text}</div>', unsafe_allow_html=True)
