# 차량 우회전 양보 감지 (Right-Turn Yield Detection)

교차로 영상에서 우회전 차량·오토바이가 우회전 경로가 지나는 횡단보도의 보행자에게 **양보했는지**(1단계), 적색 신호에서 정지선에 **일시정지했는지**(2단계) 자동으로 판정하고, 교차로·시간대별 위반율을 95% 신뢰구간과 함께 보여 주는 영상 분석 프로젝트입니다. 현장 적발용이 아니라 단속 위치·시간의 우선순위를 정하는 통계 도구입니다.

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
- 학원 서버(L40S 46GB)도 cu118 빌드를 그대로 씁니다. 서버에서 `import cv2`가 `libGL.so.1` 오류를 내면 `opencv-python`을 지우고 같은 버전의 `opencv-python-headless`를 설치합니다.

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
python -m tracking.roi_tool     --video <영상> --out configs/site_<영상>.yaml --width-m <실측> --length-m <실측>  # 횡단보도·ROI 지정 (실측은 선택. 없으면 미터 좌표·진입 속도만 빔)
python -m tracking.track        --video <영상> --site configs/site_<영상>.yaml         # → outputs/tracks/
python -m tracking.events       --tracks outputs/tracks/<영상>.csv --site configs/site_<영상>.yaml   # → outputs/events/
python -m traffic_light.color          --video <영상> --site configs/site_<영상>.yaml         # 2단계 → outputs/signal/ (빨간불 시간표)
python -m features.build        --events outputs/events/<영상>.csv --tracks outputs/tracks/<영상>.csv --site configs/site_<영상>.yaml --signal outputs/signal/<영상>.csv  # → outputs/features/
python -m rules.rule            --features outputs/features/<영상>.csv --site configs/site_<영상>.yaml  # → outputs/judged/
python -m eval.report           --judged outputs/judged --labels data/labels      # 위반율 표·그래프
python -m eval.clips            --judged outputs/judged/<영상>.csv --tracks outputs/tracks/<영상>.csv --video <영상>  # → outputs/clips/ (블러한 위반 의심 클립)
python -m eval.overlay          --video <영상> --tracks outputs/tracks/<영상>.csv --judged outputs/judged/<영상>.csv --site configs/site_<영상>.yaml  # → outputs/overlay/ (블러한 분석 영상)
python -m eval.blur_video       --video <영상>                                           # → outputs/blur/ (영상 전체 블러본, 판정 결과 없이 공유·발표용)
streamlit run dashboard/app.py                                                          # 위반율 대시보드 + 영상 분석 화면(사이드바)
```

단계 사이 CSV의 컬럼은 [`docs/csv_schema.md`](docs/csv_schema.md)에 있습니다. MVP(1단계)에서는 `traffic_light.color`를 건너뛰고 `features.build`의 `--signal`을 생략합니다.

### 대시보드 로그인·MySQL (선택)

레포 루트에 `.env`가 없으면 대시보드는 위처럼 `outputs/judged`의 CSV를 바로 읽습니다. `.env`에 아래 값이 있으면 로그인한 사람에게만 보여 주고 자료를 MySQL에서 읽습니다. 아이디는 이메일이고, 가입은 신청 후 관리자가 승인해야 로그인됩니다. `.env`와 DB 백업 파일(이름·이메일·연락처 포함)은 git·Drive·카카오톡에 올리지 않습니다.

```
DB_HOST=localhost
DB_PORT=3306
DB_USER=<MySQL 계정>
DB_PASSWORD=<비밀번호>
DB_NAME=<DB 이름>
```

```bash
python -m dashboard.db init                    # users·events 표 만들기
python -m dashboard.db admin                   # 첫 관리자 계정(비밀번호는 화면에 보이지 않게 입력)
python -m dashboard.db import outputs/judged   # 판정 CSV를 MySQL에 넣기(같은 영상은 다시 넣음)
streamlit run dashboard/app.py                 # 로그인 화면부터 뜬다
```

### 대시보드 영상 분석 화면

`streamlit run dashboard/app.py` → 왼쪽 사이드바 **영상 분석**. 이 PC에서 쓴다(영역 지정 창이 대시보드를 띄운 PC 화면에 뜬다).

1. **영상 등록**: 교차로 이름(등록한 이름 목록에서 고르거나 "+ 새 교차로", 공백·밑줄 없이), 우회전 방향(남→동·동→북·북→서·서→남), 촬영 날짜·시작 시각, 날씨(맑음·우천), 신호 조건(직진 신호만 / 우회전 전용 신호 있음 = 적색 정지 판정에서 제외), 영상 파일 경로(탐색기에서 Shift+우클릭 → 경로로 복사). 낮·야간은 시작 시각으로 정해진다(`configs/eval.yaml`의 `day_hours`). 여러 교차로·영상을 쌓아 둔다. 영상은 복사하지 않는다. 같은 파일을 두 번 등록하면 목록에 "중복"이 붙는다.
   - 사이드바 **공통 필터**(교차로·촬영 기간·낮/야간·날씨)는 등록된 영상 목록과 처리 현황에만 적용된다. **분석 모드**(보통·엄격)는 위반 통계 화면과 분석 리포트의 적색 판정 모드다.
2. **영역 설정**: 사이드바 처리 현황의 **영역 설정 시작**을 누르면 영역이 없는 영상마다 `tracking.roi_tool` 창이 차례로 뜬다. 마지막 화면에서 Enter로 저장하면 다음 영상으로 넘어간다(Esc = 건너뛰기). site 파일은 `configs/site_<영상 이름>.yaml`.
3. **분석**: 영역을 설정한 영상을 백그라운드로 분석한다(track → events → 신호등 색(신호등 영역이 있을 때) → features → rules, DB 설정 시 events 표에 넣기). 화면을 닫아도 계속되고 한 번에 하나만 돈다. 진행 막대와 영상별 단계·진행률(검출·추적 %)·남은 시간 어림이 5초마다 갱신된다. 메일 주소를 적으면 끝난 뒤 리포트 PDF를 보낸다.
4. **리포트**: 분석마다 리포트 PDF를 내려받는다. 위반 통계 화면의 **리포트 PDF 만들기**는 지금까지 쌓인 판정(현재 필터)으로 만든다.

영상 이름은 `{교차로}-{방향 코드}_{YYYYMMDD}_{HHMM}`(S·E·N·W)으로 정해지고, 영상 목록·분석 작업(진행 상태·로그·리포트)은 `data/dashboard/`에 남는다(git에 올리지 않음). 리포트 PDF와 메일에는 통계만 담고 영상·클립은 넣지 않는다.

메일을 보내려면 `.env`에 보내는 계정을 넣는다(없으면 메일만 빼고 나머지는 그대로 된다).

```
SMTP_HOST=smtp.gmail.com
SMTP_PORT=465
SMTP_USER=<보내는 계정>
SMTP_PASSWORD=<앱 비밀번호>
SMTP_FROM=<보내는 사람 주소, 생략하면 SMTP_USER>
```

포트 465는 SSL, 587 등 그 밖의 포트는 STARTTLS로 접속한다. `.env`는 줄 끝 주석을 지원하지 않으므로 값 뒤에 `#` 설명이나 따옴표를 붙이지 않는다(붙이면 그 글자까지 값으로 읽혀 발송이 실패한다). Gmail은 계정 비밀번호 대신 앱 비밀번호(2단계 인증을 켠 뒤 발급)를 넣는다.

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

코드·CSV는 `yield` / `no_yield`를 쓰고, 문서·발표·대시보드는 "위반 의심"을 씁니다. **미양보 = 위반 의심**, **위반율 = 미양보 ÷ (양보 + 미양보) = 1 − 양보율**. 양보는 "진입 순간 횡단보도에 보행자가 없음(위반 아님)"이라 보행자가 없던 우회전도 포함합니다. 적색 판정은 `stopped`(준수) / `stopped_minor`(경미: 정지선을 넘었지만 횡단보도 전에 섬) / `not_stopped`이고, 적색 위반율은 따로 집계합니다. 보통 모드(기본)는 미정지 ÷ (준수 + 경미 + 미정지), 엄격 모드는 (미정지 + 경미) ÷ 같은 분모입니다(기술서 2.3).

## 개인정보

- 원본 영상은 Google Drive·카카오톡 등 외부 서비스에 올리지 않습니다. 최종 발표 후 폐기합니다.
- 위반 의심(미양보·미정지) 클립과 분석 영상은 사람 박스 전체(운전자로 뺀 사람 포함)와 차량·오토바이 박스 아래 40%(번호판 위치)를 블러합니다. 번호 인식은 하지 않습니다.
- 원본 보관: T5 SSD `D:\geumjjogi-detection\raw\`(육주영) + 촬영자(장범수) PC. 팀원에게는 SSD·USB로 직접 전달합니다.

## 환경 메모

torch는 GPU마다 빌드가 달라 `environment.yml`에만 적습니다.

| GPU | `environment.yml`의 torch 줄 |
| --- | --- |
| GTX 1060 (현재 PC), L40S 46GB (학원 서버) | `--extra-index-url .../whl/cu118`, `torch==2.7.1+cu118`, `torchvision==0.22.1+cu118` |

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
