# 차량 우회전 양보 감지 (Right-Turn Yield Detection)

교차로 영상에서 우회전 차량·오토바이가 출구 횡단보도의 보행자에게 **양보했는지** 자동으로 판정하고, 시간대별 양보율을 95% 신뢰구간과 함께 보여 주는 영상 분석 프로젝트입니다.

- 팀: 금쪽같은 우리 사이 (장범수 · 김민영 · 육주영)
- 기간: 2026.10.26 ~ 2026.11.27 · MVP 11.18
- 방식: YOLO 검출 → ByteTrack 추적 → homography(미터 좌표) → 우회전 이벤트 추출 → 양보 / 미양보 판정 → 양보율 집계

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
python -m tracking.roi_tool     --video <영상> --out configs/site_A.yaml --width-m <실측> --length-m <실측>  # 횡단보도·ROI 지정
python -m tracking.track        --video <영상> --site configs/site_A.yaml         # → outputs/tracks/
python -m tracking.events       --tracks outputs/tracks/<영상>.csv --site configs/site_A.yaml   # → outputs/events/
python -m features.build        --events outputs/events/<영상>.csv --tracks outputs/tracks/<영상>.csv --site configs/site_A.yaml  # → outputs/features/
python -m yield_classifier.rule --features outputs/features/<영상>.csv --site configs/site_A.yaml  # → outputs/judged/
python -m eval.report           --judged outputs/judged --labels data/labels      # 양보율 표·그래프
```

단계 사이 CSV의 컬럼은 [`docs/csv_schema.md`](docs/csv_schema.md)에 있습니다.

## 폴더 구조

| 폴더 | 내용 | 담당 |
| --- | --- | --- |
| `configs/` | 검출·추적 설정, 교차로별 ROI·횡단보도 설정 | 공통 |
| `detection/` | 사전학습 시험, 모델 비교(YOLO11n vs YOLO26s), 미세조정 | 장범수 |
| `tracking/` | ROI 지정 도구, homography, 추적, 우회전 이벤트 추출 | 김민영 |
| `features/` | 속도·거리·보행자 특징 계산 | 김민영 |
| `yield_classifier/` | 규칙 기반 판정, RF·XGBoost 분류기 | 육주영 |
| `eval/` | 평가 지표, 시간대별 양보율, 미양보 클립 | 육주영 |
| `docs/` | 판정 기준표, 라벨링 가이드, CSV 스키마, 팀 결정 필요 목록, 가중치·실험 기록 | 공통 |
| `notebooks/` | 정리된 분석 노트북 | 공통 |
| `data/`, `outputs/`, `weights/` | 라벨·결과·가중치 (git 제외) | — |

## 개인정보

- 원본 영상은 Google Drive·카카오톡 등 외부 서비스에 올리지 않습니다. 최종 발표 후 폐기합니다.
- 미양보 클립은 보행자 박스 전체와 차량·오토바이 박스 아래 40%(번호판 위치)를 블러합니다. 번호 인식은 하지 않습니다.

## 팀 규칙

- 공통 규칙: [`.claude/rules/team.md`](.claude/rules/team.md)
- 이 레포 규칙: [`CLAUDE.md`](CLAUDE.md)
