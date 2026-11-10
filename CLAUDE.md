이 레포에만 적용하는 지역지침이다. 팀 공통 규칙(브랜치·커밋·작업 원칙)은
`.claude/rules/team.md`(전역지침)를 따른다.
전역지침과 내용이 다르면 이 파일이 우선한다.

- **레포:** https://github.com/Jbs1214/geumjjogi-detection
- **주제:**  차량 우회전 위반(양보) 감지
- **기준 문서:** 기획서·기술서(Drive, 2026-11-04). 레포 문서와 다르면 기획서·기술서가 우선한다.

---

## 1. 방식

교차로 고정 촬영 영상을 오프라인으로 분석해, 우회전 차량·오토바이가 (1단계) 출구 횡단보도 위 보행자에게 **양보 / 미양보**했는지, (2단계) 적색 신호에서 정지선에 **일시정지 / 미정지**했는지 판정한다. 판정 기준은 `docs/judgment_criteria.md`를 따른다. 결과는 위반 의심이며 법적 판정이 아니다.

| 단계 | 방법 |
| --- | --- |
| 검출 (M1) | 사전학습 **YOLO11n**으로 시작(GTX 1060 선 테스트). YOLO26s와 비교하고, RTX 5090 서버가 제공되면 모델을 다시 고른다 |
| 검출 고도화 | 멀리 있는 사람·오토바이 YOLO 미세조정(약 400프레임, 영상 단위로 학습·검증 분리), 전후 재현율·mAP50·미양보 F1 비교. 확정 단계, 장범수, 11.19~25 병행 |
| 추적 | ultralytics 내장 ByteTrack (`tracker="bytetrack.yaml"`) |
| 좌표 | 횡단보도 네 모서리 → homography → 미터 단위 평면 |
| 신호 (2단계) | 차량 신호등(직진 신호) 영역을 영상마다 지정, HSV 빨간 픽셀 비율로 판정(5프레임 연속 같을 때만 상태 변경) → 영상 전체 빨간불 시간표 |
| 우회전 판별 | ROI 규칙 (우회전 진입로 → 출구 통과) |
| 판정 (M2) | **규칙 판정만** 사용(학습 모델은 검출 미세조정 하나뿐). 보행자: 양보 / 미양보 / 제외, 정지선: 일시정지 / 미정지 / 제외 |

**클래스** (매핑은 `configs/detect.yaml`)

| COCO 클래스 | 프로젝트 클래스 | 미세조정 ID |
| --- | --- | --- |
| car, bus, truck | vehicle | 0 |
| motorcycle | motorcycle | 1 |
| person | pedestrian | 2 |

- `bicycle` 등 나머지 COCO 클래스는 버린다.
- 고도화 예정: 전동킥보드 클래스 (COCO에 없으므로 직접 라벨링·미세조정 필요).

**화면:** 판정 결과 CSV를 읽는 읽기 전용 **Streamlit 대시보드**(`dashboard/`)를 만든다. 중간 발표(11.18) 시연은 분석 영상으로 한다. 결과물은 아래 5가지이다.

1. 박스·궤적·횡단보도·판정을 그린 분석 영상(mp4)
2. 이벤트 CSV
3. 교차로·시간대별 위반율 표·그래프 (Wilson 95% 신뢰구간, 판정 대상 30건 미만은 "참고용")
4. 위반 의심 클립 (보행자 박스 전체·차량·오토바이 박스 아래 40%(번호판 위치) 블러)
5. Streamlit 대시보드: 위반율 표, 단속 우선 시간대 TOP 3, 클립 보기, CSV 내보내기. 화면에 "위반 의심 통계이며 실제 단속은 별도 확인이 필요하다" 문구를 항상 표시

**파트별 담당**

| 담당 | 폴더 | 브랜치 예 |
| --- | --- | --- |
| 장범수 | `detection/` | `feat/det-compare` |
| 김민영 | `tracking/`, `features/`, `traffic_light/` | `feat/track-homography` |
| 육주영 | `rules/`, `eval/`, `dashboard/` | `feat/rules-ped` |
| 공통 | `configs/`, `docs/`, `notebooks/` | — |

## 2. 데이터

- **출처:** 팀 직접 촬영 (교차로 1곳, 육교·건물 위 삼각대). 공공 CCTV는 쓰지 않는다.
- **파일 이름:** `{교차로}_{YYYYMMDD}_{HHMM}.mp4` (예: `A_20261031_0800.mp4`). 교차로 코드는 1순위 A, 2순위 B. 촬영 직후 장범수가 바꾼다.
- **원본 보관:** T5 SSD `D:\geumjjogi-detection\raw\`(육주영) + 촬영자(장범수) PC. 팀원에게는 SSD·USB로 직접 전달한다.
- **원본은 Google Drive·카카오톡 등 외부 서비스에 올리지 않는다.** 전역지침 5절(대용량 파일은 Drive)보다 이 규칙이 우선한다.
- **Drive에 올리는 것:** 가중치, 라벨표, 결과표, 블러 처리한 클립.
- **폐기:** 최종 발표(11.27) 후 원본을 모두 지운다.
- **레포 `data/`:** 라벨 CSV·특징 CSV 같은 작은 파일만 둔다. git에서 제외한다. 원본 영상은 스크립트에 `--video`로 경로를 넘긴다.
- **라벨링:** 이벤트 라벨(양보/미양보/제외)은 CSV 라벨표, 박스 라벨(검출 고도화용)은 로컬 labelImg(YOLO 형식). 규칙은 `docs/labeling_guide.md`.

## 3. 환경

- conda 환경: `geumjjogi-detection` (Python 3.11)
- 공통 환경(Python·GPU·CUDA 등)은 전역지침 `team.md`를 따른다.
- 환경은 `conda env create -f environment.yml`로 만든다. 바뀐 뒤에는 `conda env update -f environment.yml --prune`.
- torch는 GPU마다 빌드가 달라서 `environment.yml`에만 적고, 나머지 라이브러리 버전은 `requirements.txt` 한 곳에서 고정한다. 라이브러리를 추가할 때는 `requirements.txt`에 버전을 고정해서 넣는다.
- `conda env export` 결과는 커밋하지 않는다(PyTorch 주소가 빠지고 PC 경로·Windows 빌드가 들어가 다른 PC에서 실패한다).
- `environment.yml`·`requirements.txt`에는 **한글을 쓰지 않는다.** 한글 Windows의 conda가 cp949로 읽다가 `UnicodeDecodeError`로 멈춘다.

| GPU | `environment.yml`의 torch 줄 |
| --- | --- |
| GTX 1060 (현재 PC) | `--extra-index-url .../whl/cu118`, `torch==2.7.1+cu118`, `torchvision==0.22.1+cu118` |
| RTX 5090 (학원 서버) | cu128 빌드 — 서버 제공 시 버전 확정 |

## 4. 실행 방법

```bash
conda activate geumjjogi-detection
python check_env.py                                    # GPU·라이브러리 확인

python -m tracking.roi_tool     --video <영상> --out configs/site_A.yaml --width-m <실측> --length-m <실측>
python -m tracking.track        --video <영상> --site configs/site_A.yaml
python -m tracking.events       --tracks outputs/tracks/<영상>.csv --site configs/site_A.yaml
python -m traffic_light.color         --video <영상> --site configs/site_A.yaml          # 2단계, → outputs/signal/ (빨간불 시간표)
python -m features.build        --events outputs/events/<영상>.csv --tracks outputs/tracks/<영상>.csv --site configs/site_A.yaml --signal outputs/signal/<영상>.csv
python -m rules.rule            --features outputs/features/<영상>.csv --site configs/site_A.yaml
python -m eval.report           --judged outputs/judged --labels data/labels
streamlit run dashboard/app.py                          # 위반율 대시보드(읽기 전용)
```

- 단계마다 결과를 `outputs/{tracks,events,features,judged}/<영상>.csv`로 남기고 다음 단계가 읽는다. 컬럼은 `docs/csv_schema.md`가 기준이다.
- 그 밖의 결과: `outputs/detection/`(검출 시험), `outputs/report/`(위반율 표·그래프), `outputs/clips/`(위반 의심 클립). `outputs/`는 git에서 제외한다.
- 노트북은 **출력 셀을 모두 지우고** 커밋한다(원본 프레임이 이미지로 박히는 것을 막기 위해).
- 한 번에 돌리는 스크립트는 MVP 이후 필요하면 만든다.
- 팀이 아직 정하지 않은 항목은 `docs/open_questions.md`에 있다.

## 5. 가중치 버전 관리

- **사전학습 가중치**(`yolo11n.pt`, `yolo26s.pt`)는 ultralytics가 자동으로 내려받는다. 따로 관리하지 않는다.
- **직접 학습한 가중치** 이름:
  - 검출: `det_{모델}_v{번호}_{YYMMDD}.pt` (예: `det_yolo11n_v1_261112.pt`)
- 파일은 Google Drive `weights/`에 올리고, 레포의 `weights/`(git 제외)에 내려받아 쓴다.
- 버전마다 `docs/weights.md`에 날짜·학습 데이터(영상 목록)·주요 수치·GPU·Drive 링크를 적는다. 이 표는 git에 올린다.

## 6. 라이브러리

- **쓰기로 한 것:** torch, ultralytics(YOLO·ByteTrack), opencv-python, numpy, pandas, scikit-learn(F1·kappa), matplotlib, pyyaml, streamlit(대시보드, 기획서 3-4절)
- **쓰지 않기로 한 것:**

| 라이브러리 | 이유 |
| --- | --- |
| W&B, MLflow | ultralytics `runs/`의 결과로 충분하고, W&B 연동은 원본 프레임을 외부에 올린다. 실험 기록은 Drive 실험 결과표 + `docs/experiments.md` |
| CVAT | 클라우드에 원본을 올리게 된다. CSV 라벨표·labelImg로 대체 |
| supervision, 별도 ByteTrack, DeepSORT | ultralytics 내장 추적과 겹친다 |
| Gradio | 대시보드는 Streamlit 하나로 통일 |
| statsmodels | Wilson 신뢰구간은 공식 몇 줄로 계산 |
| 얼굴·번호판 인식 모델, OCR | 개인정보. 블러는 검출 박스로 처리 |
| seaborn | matplotlib으로 충분 |

목록에 없는 라이브러리를 추가하려면 먼저 팀에 공유한다(전역지침 2절).

## 7. 브랜치·커밋 (지역 추가 규칙)

- 브랜치·커밋 형식은 전역지침 4절을 따른다.
- 브랜치 이름에 파트를 넣는다: `feat/det-…`, `feat/track-…`, `feat/rules-…`
- MVP가 완성되면 `v0.1-mvp` 태그를 남긴다.
