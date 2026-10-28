# data/

이 폴더는 git에서 제외된다(이 README만 올라간다).

- 원본 영상은 여기에 두지 않는다. T5 SSD `D:\geumjjogi-detection\raw\`에 두고 스크립트에 `--video`로 경로를 넘긴다.
- 여기에는 작은 파일만 둔다.
  - `labels/` — 이벤트 라벨 CSV (`docs/labeling_guide.md`)
  - `boxes/` — 미세조정용 박스 라벨 (labelImg, YOLO 형식)
