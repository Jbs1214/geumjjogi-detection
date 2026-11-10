# 차량 우회전 양보 감지 (Right-Turn Yield Detection)

교차로 영상에서 우회전 차량·오토바이가 출구 횡단보도의 보행자에게 **양보했는지**(1단계), 적색 신호에서 정지선에 **일시정지했는지**(2단계) 자동으로 판정하고, 교차로·시간대별 위반율을 95% 신뢰구간과 함께 보여 주는 영상 분석 프로젝트입니다. 현장 적발용이 아니라 단속 위치·시간의 우선순위를 정하는 통계 도구입니다.

- 팀: 금쪽같은 우리 사이 (장범수 · 김민영 · 육주영)
- 기간: 2026.10.26 ~ 2026.11.27 · MVP 11.18
- 방식: YOLO 검출 → ByteTrack 추적 → homography(미터 좌표) → 우회전 이벤트 추출 → 규칙 판정(양보 / 미양보, 일시정지 / 미정지) → 위반율 집계 → Streamlit 대시보드

판정 기준은 [`docs/judgment_criteria.md`](docs/judgment_criteria.md)에 있습니다. 이 기준은 영상 분석용 정의이며 법적 위반 판정이 아닙니다.

## 설치

```bash
git clone https://github.com/Jbs1214/geumjjogi-detection.git
cd geumjjogi-detection

conda env create -f environment.yml     # Python 3.11 + torch(cu118) + requirements.txt
conda activate geumjjogi-detection
python check_env.py                     # 모두 [OK]면 성공
```

- 다른 사람이 `environment.yml`·`requirements.txt`를 바꾼 뒤에는 `conda env update -f environment.yml --prune`으로 맞춥니다.
- RTX 5090(학원 서버)에서는 `environment.yml`의 torch 세 줄을 cu128 빌드로 바꿉니다(서버 제공 시 확정).

**`conda env create`가 `This program is blocked by group policy`로 실패할 때 (학원 PC)**

학원 PC는 `.bat` 실행이 막혀 있어서 conda가 pip 단계를 돌리지 못합니다. PowerShell에서 두 단계로 나눠 설치합니다(설치되는 버전은 같습니다).

```powershell
conda env remove -n geumjjogi-detection          # 실패하며 반쯤 만들어진 환경이 있으면 지운다
conda create -n geumjjogi-detection python=3.11 -y
conda activate geumjjogi-detection
pip install --extra-index-url https://download.pytorch.org/whl/cu118 torch==2.7.1+cu118 torchvision==0.22.1+cu118 -r requirements.txt
python check_env.py
```

## 실행 순서

원본 영상은 레포 밖(T5 SSD 등)에 두고 `--video`로 경로를 넘깁니다. 단계마다 결과 CSV가 `outputs/`에 남고 다음 단계가 읽습니다.

```bash
python -m tracking.roi_tool     --video <영상> --out configs/site_<영상>.yaml --width-m <실측> --length-m <실측>  # 횡단보도·ROI 지정
python -m tracking.track        --video <영상> --site configs/site_<영상>.yaml         # → outputs/tracks/
python -m tracking.events       --tracks outputs/tracks/<영상>.csv --site configs/site_<영상>.yaml   # → outputs/events/
python -m traffic_light.color          --video <영상> --site configs/site_<영상>.yaml         # 2단계 → outputs/signal/ (빨간불 시간표)
python -m features.build        --events outputs/events/<영상>.csv --tracks outputs/tracks/<영상>.csv --site configs/site_<영상>.yaml --signal outputs/signal/<영상>.csv  # → outputs/features/
python -m rules.rule            --features outputs/features/<영상>.csv --site configs/site_<영상>.yaml  # → outputs/judged/
python -m eval.report           --judged outputs/judged --labels data/labels      # 위반율 표·그래프
python -m eval.clips            --judged outputs/judged/<영상>.csv --tracks outputs/tracks/<영상>.csv --video <영상>  # → outputs/clips/ (블러한 위반 의심 클립)
python -m eval.overlay          --video <영상> --tracks outputs/tracks/<영상>.csv --judged outputs/judged/<영상>.csv --site configs/site_<영상>.yaml  # → outputs/overlay/ (블러한 분석 영상)
streamlit run dashboard/app.py                                                          # 위반율 대시보드(읽기 전용)
```

단계 사이 CSV의 컬럼은 [`docs/csv_schema.md`](docs/csv_schema.md)에 있습니다. MVP(1단계)에서는 `traffic_light.color`를 건너뛰고 `features.build`의 `--signal`을 생략합니다.

## 폴더 구조

| 폴더 | 내용 | 담당 |
| --- | --- | --- |
| `configs/` | 검출·추적(`detect.yaml`), 영상별 ROI·판정 기준값(`site_<영상>.yaml`), 평가·블러(`eval.yaml`) | 공통 |
| `detection/` | 사전학습 시험, 모델 비교(YOLO11n vs YOLO26s), 미세조정 | 장범수 |
| `tracking/` | ROI 지정 도구, homography, 추적, 우회전 이벤트 추출 | 김민영 |
| `features/` | 속도·거리·보행자 특징 계산 | 김민영 |
| `traffic_light/` | 차량 신호등 색 판정(빨간불 시간표). 영역은 `tracking.roi_tool`에서 지정 | 김민영 |
| `rules/` | 규칙 판정(보행자·정지선) | 육주영 |
| `eval/` | 평가 지표, 시간대별 위반율, 위반 의심 클립, 분석 영상 | 육주영 |
| `dashboard/` | Streamlit 위반율 화면(판정 CSV를 읽는 읽기 전용) | 육주영 |
| `docs/` | 판정 기준표, 라벨링 가이드, CSV 스키마, 팀 결정 필요 목록, 가중치·실험 기록 | 공통 |
| `notebooks/` | 정리된 분석 노트북 | 공통 |
| `data/`, `outputs/`, `weights/` | 라벨·결과·가중치 (git 제외) | — |

## 용어: 미양보 = 위반 의심

코드·CSV는 `yield` / `no_yield`를 쓰고, 문서·발표·대시보드는 "위반 의심"을 씁니다. **미양보 = 위반 의심**, **위반율 = 미양보 ÷ (양보 + 미양보) = 1 − 양보율**. 적색 위반율은 미정지 ÷ (일시정지 + 미정지)이고 따로 집계합니다.

## 개인정보

- 원본 영상은 Google Drive·카카오톡 등 외부 서비스에 올리지 않습니다. 최종 발표 후 폐기합니다.
- 위반 의심(미양보·미정지) 클립과 분석 영상은 사람 박스 전체(운전자로 뺀 사람 포함)와 차량·오토바이 박스 아래 40%(번호판 위치)를 블러합니다. 번호 인식은 하지 않습니다.
- 원본 보관: T5 SSD `D:\geumjjogi-detection\raw\`(육주영) + 촬영자(장범수) PC. 팀원에게는 SSD·USB로 직접 전달합니다.

## 환경 메모

torch는 GPU마다 빌드가 달라 `environment.yml`에만 적습니다.

| GPU | `environment.yml`의 torch 줄 |
| --- | --- |
| GTX 1060 (현재 PC) | `--extra-index-url .../whl/cu118`, `torch==2.7.1+cu118`, `torchvision==0.22.1+cu118` |
| RTX 5090 (학원 서버) | cu128 빌드 — 서버 제공 시 버전 확정 |

## 쓰지 않기로 한 라이브러리

| 라이브러리 | 이유 |
| --- | --- |
| W&B, MLflow | ultralytics `runs/`로 충분하고, W&B 연동은 원본 프레임을 외부에 올린다. 실험 기록은 Drive 실험표 + `docs/experiments.md` |
| CVAT | 클라우드에 원본을 올리게 된다. CSV 라벨표·labelImg로 대체 |
| supervision, 별도 ByteTrack, DeepSORT | ultralytics 내장 추적과 겹친다 |
| Gradio | 대시보드는 Streamlit 하나로 통일 |
| statsmodels | Wilson 신뢰구간은 공식 몇 줄로 계산 |
| 얼굴·번호판 인식 모델, OCR | 개인정보. 블러는 검출 박스로 처리 |
| seaborn | matplotlib으로 충분 |

목록에 없는 라이브러리를 추가하려면 먼저 팀에 공유하고 `requirements.txt`에 버전을 고정해 넣습니다.

## 팀 규칙

- 공통 규칙: [`.claude/rules/team.md`](.claude/rules/team.md)
- 이 레포 규칙: [`CLAUDE.md`](CLAUDE.md)
- 브랜치 이름에 파트를 넣습니다: `feat/det-…`, `feat/track-…`, `feat/rules-…`. 문서만 고칠 때는 `docs/…`.
- MVP가 완성되면 `v0.1-mvp` 태그를 남깁니다.
