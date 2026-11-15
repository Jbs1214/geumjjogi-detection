# 담당: 육주영
"""위반율 대시보드 (Streamlit, 읽기 전용). outputs/judged/*.csv를 pandas로 모아 보여 준다.
실행: streamlit run dashboard/app.py

화면(기술서 3.2, 배치는 기획서 3-4절 예시 화면): 화면 위 필터(교차로·기간·위반 유형·정렬), 요약 카드, 교차로 x 시간대 위반율 표(Wilson 95% 신뢰구간, configs/eval.yaml의 min_n_reference 미만 "참고용"),
단속 우선 시간대 TOP 3와 클립 보기, 현재 필터의 표를 CSV로 내려받기.
"위반 의심 통계이며 실제 단속은 별도 확인이 필요하다" 문구를 항상 표시한다.
위반 유형 "전체"는 두 유형을 나란히 보여 주고 위반율을 하나로 합치지 않는다(분모가 다르다).
"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # streamlit run은 레포 루트를 import 경로에 넣지 않는다

import pandas as pd
import streamlit as st
import yaml

from eval.report import rate_table, wilson_ci

NOTICE = "위반 의심 통계이며 실제 단속은 별도 확인이 필요하다"
TOP_N = 3
KINDS = {  # 표시 이름: (판정 컬럼, 위반 아님, 위반 의심, 내려받기 파일 이름)
    "보행자 미양보": ("ped_result", "yield", "no_yield", "violation_rate.csv"),
    "적색 미정지": ("red_result", "stopped", "not_stopped", "red_rate.csv"),
}
ALL_KINDS = "전체"
BAR_COLOR = "#e05252"  # 위반율 막대
BADGE_ENOUGH = "background-color: #d7f0df; color: #1a7f43"  # 신뢰도 배지
BADGE_REFERENCE = "background-color: #fdecc8; color: #a05a00"
SORTS = {  # 표시 이름: (정렬 컬럼, 오름차순)
    "위반율 높은 순": ("violation_rate", False),
    "시간대 순": ("slot", True),
    "판정 대상 많은 순": ("n_target", False),
}


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
    c1.metric("판정 대상 우회전", f"{n}건", border=True)
    c2.metric("위반 의심", f"{k}건", border=True)
    c3.metric("위반율 (신뢰구간)", f"{k / n:.1%}", f"{low:.1%}~{high:.1%}", delta_color="off", delta_arrow="off", border=True)

    view = pd.DataFrame({
        "교차로": "교차로 " + table["intersection"].astype(str),
        "시간대": table["slot"],
        "판정 대상": table["n_target"],
        "위반 의심": table[f"n_{bad}"],
        "위반율": table["violation_rate"],
        "신뢰구간": [
            f"{lo:.1%}~{hi:.1%}" if pd.notna(lo) else "-" for lo, hi in zip(table["ci_low"], table["ci_high"])
        ],
        "신뢰도": table["note"].replace("", "표본 충분"),
    })
    st.dataframe(
        view.style.map(lambda v: BADGE_ENOUGH if v == "표본 충분" else BADGE_REFERENCE, subset=["신뢰도"]),
        column_config={
            "위반율": st.column_config.ProgressColumn("위반율", min_value=0, max_value=1, format="percent", color=BAR_COLOR)
        },
        hide_index=True,
        width="stretch",
    )
    st.caption(f"신뢰도 '참고용' = 판정 대상 {min_n}건 미만. 단속 우선 검토 시간대 계산에서 뺀다.")
    st.download_button(
        "현재 표를 CSV로 내려받기",
        view.to_csv(index=False).encode("utf-8-sig"),
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
        with area.container(border=True):
            st.markdown(f"**:red[{rank}. 교차로 {r.intersection} {r.slot}]**")
            st.markdown(f"위반율 {r.violation_rate:.1%} · 위반 의심 {getattr(r, f'n_{bad}')}건")
            with st.expander(f"클립 보기 ▶ ({len(clips)}개)"):
                for clip in clips:
                    if Path(clip).exists():
                        st.video(clip, format="video/webm")  # eval.clips가 WebM(VP8)으로 저장한다
                    else:
                        st.caption(f"파일 없음: {clip}")


def main(judged_dir: str = "outputs/judged", config: str = "configs/eval.yaml") -> None:
    st.set_page_config(page_title="우회전 위반 의심 통계", layout="wide")
    st.title("우회전 위반율: 교차로·시간대별")
    st.warning(NOTICE)

    df = load_judged(judged_dir)
    if df.empty:
        st.info(f"판정 결과가 없다. `{judged_dir}`에 judged CSV를 만든 뒤 다시 연다.")
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
    names = list(KINDS) if kind == ALL_KINDS else [kind]
    for area, name in zip(st.columns(len(names)), names):  # "전체"는 좌우로 나란히
        with area:
            if len(names) > 1:
                st.header(name)
            render(name, filtered, min_n, sort_name)
    st.caption("클립은 얼굴·번호판을 블러한 영상만 보여 줍니다.")


if __name__ == "__main__":
    main()
