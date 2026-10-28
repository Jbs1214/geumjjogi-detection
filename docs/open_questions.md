# 팀 결정 필요 목록

커밋 전 리뷰(261028)에서 나온 항목 중 팀이 정해야 하는 것. 정하면 해당 문서·설정에 반영하고 이 표에서 지운다.

| # | 질문 | 관련 파일 | 담당 | 기한 |
| --- | --- | --- | --- | --- |
| 1 | site 설정 파일 단위: 교차로마다 1개(`site_A.yaml`)인가, 촬영일마다 1개(`site_A_20261031.yaml`)인가? 삼각대라 화각이 매번 바뀔 수 있다 | `configs/site_example.yaml`, `tracking/roi_tool.py` | 김민영 | 10.31 1주차 촬영 전 |
| 2 | 정답 구간 이벤트와 파이프라인 이벤트를 같은 것으로 보는 규칙 (예: `t_enter_s` 차이 ±N초 이내) | `eval/metrics.py` `extraction_recall` | 육주영 | 11.09 |
| 3 | 이중 라벨 구간에서 어느 라벨을 정답으로 쓰는가, 두 라벨이 다를 때 어떻게 정하는가 | `docs/labeling_guide.md`, `yield_classifier/train.py`, `eval/report.py` | 육주영·장범수 | 11.09 |
| 4 | 결과물 1번 "분석 영상(mp4)"을 만드는 스크립트와 담당 | `CLAUDE.md` 1절 | 팀 | 11.16 |
| 5 | 모델 비교(YOLO11n vs 26s)용 박스 라벨을 누가 몇 장 만들고 `data.yaml`을 어디에 두는가 | `detection/compare.py`, `docs/labeling_guide.md` | 장범수 | 11.06 |
| 6 | RF·XGB 판정 결과도 `outputs/judged/`로 내보내 양보율 집계에 쓰는가, 규칙 판정만 쓰는가 | `yield_classifier/train.py`, `eval/report.py` | 육주영 | 11.21 |
| 7 | 오토바이 운전자를 뺄 때 "크게 겹침"의 기준값(IoU 등)과 그 값을 둘 설정 위치 | `tracking/track.py`, `configs/` | 김민영 | 11.15 |
| 8 | 미세조정 스텁(`detection/train.py`)을 지금 둘지, 필요해질 때 만들지 | `detection/train.py` | 장범수 | 11.06 |
