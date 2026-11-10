# 담당: 육주영
"""위반율 대시보드 (Streamlit, 읽기 전용). outputs/judged/*.csv를 pandas로 모아 보여 준다.
실행: streamlit run dashboard/app.py

화면(기술서 3.2): 필터(교차로·기간·위반 유형·정렬), 요약 카드, 교차로 x 시간대 위반율 표(Wilson 95% 신뢰구간, configs/eval.yaml의 min_n_reference 미만 "참고용"),
단속 우선 시간대 TOP 3와 클립 보기, 현재 필터의 표를 CSV로 내려받기.
"위반 의심 통계이며 실제 단속은 별도 확인이 필요하다" 문구를 항상 표시한다.

지금은 1단계(보행자 미양보)만 보여 준다. 위반 유형 필터와 적색 미정지는 2단계에서 추가한다.
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
    """표본이 충분한(참고용이 아닌) 칸 중 위반율 높은 순 n개."""
    return table[table["note"] == ""].sort_values("violation_rate", ascending=False).head(n)


def main(judged_dir: str = "outputs/judged", config: str = "configs/eval.yaml") -> None:
    st.set_page_config(page_title="우회전 위반 의심 통계", layout="wide")
    st.title("우회전 보행자 미양보 — 교차로·시간대별 위반율")
    st.warning(NOTICE)

    df = load_judged(judged_dir)
    if df.empty:
        st.info(f"판정 결과가 없다. `{judged_dir}`에 judged CSV를 만든 뒤 다시 연다.")
        return
    with open(config, encoding="utf-8") as fp:
        min_n = yaml.safe_load(fp)["min_n_reference"]

    st.sidebar.header("필터")
    all_sites = sorted(df["intersection"].unique())
    sites = st.sidebar.multiselect("교차로", all_sites, default=all_sites)
    period = st.sidebar.date_input("기간", (df["date"].min(), df["date"].max()))
    sort_name = st.sidebar.selectbox("정렬 기준", list(SORTS))
    if len(period) != 2:  # 날짜 범위를 고르는 중
        return

    filtered = apply_filters(df, sites, period[0], period[1])
    table = rate_table(filtered, min_n)
    if table.empty:
        st.info("이 필터에 해당하는 이벤트가 없다.")
        return
    column, ascending = SORTS[sort_name]
    table = table.sort_values(column, ascending=ascending)

    n, k = int(table["n_target"].sum()), int(table["n_no_yield"].sum())
    low, high = wilson_ci(k, n)
    c1, c2, c3 = st.columns(3)
    c1.metric("판정 대상", f"{n}건")
    c2.metric("위반 의심 (미양보)", f"{k}건")
    c3.metric("위반율", f"{k / n:.1%}" if n else "-", f"95% 신뢰구간 {low:.1%} ~ {high:.1%}" if n else None, delta_color="off")

    st.subheader("교차로 × 시간대 위반율")
    view = pd.DataFrame({
        "교차로": table["intersection"],
        "시간대": table["slot"],
        "판정 대상": table["n_target"],
        "위반 의심": table["n_no_yield"],
        "위반율": table["violation_rate"],
        "위반율(%)": table["violation_rate"] * 100,
        "95% 신뢰구간": [
            f"{lo:.1%} ~ {hi:.1%}" if pd.notna(lo) else "-" for lo, hi in zip(table["ci_low"], table["ci_high"])
        ],
        "신뢰도": table["note"].replace("", "충분"),
    })
    st.dataframe(
        view.style.background_gradient(subset=["위반율(%)"], cmap="Reds", vmin=0, vmax=100).format(
            {"위반율(%)": "{:.1f}"}, na_rep="-"
        ),
        column_config={"위반율": st.column_config.ProgressColumn("위반율", min_value=0, max_value=1, format="percent")},
        hide_index=True,
        width="stretch",
    )
    st.caption(f"신뢰도 '참고용' = 판정 대상 {min_n}건 미만. 단속 우선 시간대 계산에서 뺀다.")
    st.download_button(
        "현재 표를 CSV로 내려받기",
        view.drop(columns="위반율(%)").to_csv(index=False).encode("utf-8-sig"),
        "violation_rate.csv",
        "text/csv",
    )

    st.subheader(f"단속 우선 시간대 TOP {TOP_N}")
    top = top_slots(table)
    if top.empty:
        st.info(f"판정 대상이 {min_n}건 이상인 시간대가 아직 없다.")
    for rank, r in enumerate(top.itertuples(), 1):
        st.markdown(
            f"**{rank}. 교차로 {r.intersection} · {r.slot}** — 위반율 {r.violation_rate:.1%} "
            f"({r.n_no_yield}/{r.n_target}건, 95% 신뢰구간 {r.ci_low:.1%} ~ {r.ci_high:.1%})"
        )
        clips = filtered[
            (filtered["intersection"] == r.intersection)
            & (filtered["slot"] == r.slot)
            & (filtered["ped_result"] == "no_yield")
            & (filtered["clip"] != "")
        ]["clip"]
        with st.expander(f"클립 보기 ({len(clips)}개)"):
            for clip in clips:
                if Path(clip).exists():
                    st.video(clip)
                else:
                    st.caption(f"파일 없음: {clip}")


if __name__ == "__main__":
    main()
