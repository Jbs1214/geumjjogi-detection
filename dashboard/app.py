# 담당: 육주영
"""위반율 대시보드 (Streamlit, 읽기 전용). outputs/judged/*.csv를 pandas로 모아 보여 준다.
실행: streamlit run dashboard/app.py

화면(기술서 3.2): 필터(교차로·기간·위반 유형·정렬), 요약 카드, 교차로 x 시간대 위반율 표(Wilson 95% 신뢰구간, configs/eval.yaml의 min_n_reference 미만 "참고용"),
단속 우선 시간대 TOP 3와 클립 보기, 현재 필터의 표를 CSV로 내려받기.
"위반 의심 통계이며 실제 단속은 별도 확인이 필요하다" 문구를 항상 표시한다.
"""


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
