# 단계별 CSV 스키마

> **초안 — 10.30 팀 합의로 확정.** 파트 사이의 경계이므로 컬럼을 바꾸면 이 문서를 먼저 고치고 팀에 공유한다.

## 공통 규칙

- 인코딩 **UTF-8 with BOM** (`encoding="utf-8-sig"`, 엑셀에서 한글이 깨지지 않게).
- `video` = 영상 파일 이름에서 확장자를 뺀 것 (예: `A_20261031_0800`). 출력 파일 이름도 `<video>.csv`.
- `cls` 값은 `vehicle` / `motorcycle` / `pedestrian` 셋뿐이다 (`configs/detect.yaml`의 매핑).
- 시간은 영상 시작부터 초(`_s`), 거리는 m(`_m`), 속도는 m/s(`_mps`).
- 미터 좌표: 원점 = 횡단보도 **좌상** 모서리, x축 = 좌상→우상(`width_m` 변), y축 = 좌상→좌하(`length_m` 변). 모서리 순서는 `configs/site_example.yaml`.

## 1. tracks — `outputs/tracks/<video>.csv` (김민영, `tracking.track`)

한 행 = 한 프레임의 객체 1개. 운전자로 판단한 pedestrian은 이미 빠져 있다.

| 컬럼 | 뜻 |
| --- | --- |
| `frame` | 프레임 번호 |
| `t_s` | 시각 |
| `track_id` | 추적 ID |
| `cls` | 클래스 |
| `conf` | 검출 신뢰도 |
| `x1, y1, x2, y2` | 박스 (픽셀) |
| `bx_m, by_m` | 박스 하단 중심점의 미터 좌표 |

## 2. events — `outputs/events/<video>.csv` (김민영, `tracking.events`)

한 행 = 우회전 이벤트 1건 (vehicle·motorcycle만).

| 컬럼 | 뜻 |
| --- | --- |
| `video`, `event_id`, `track_id`, `cls` | 식별 |
| `t_approach_s` | 접근 구간에 들어온 시각 |
| `t_enter_s` | 횡단보도 영역에 진입한 시각 (박스 하단 중심점 기준) |
| `t_exit_s` | 출구 ROI를 통과한 시각 |

## 3. features — `outputs/features/<video>.csv` (김민영, `features.build`)

events의 컬럼 전부 + 아래 컬럼. MVP에서는 시계열을 요약값으로만 넣는다.

| 컬럼 | 단위 | 뜻 |
| --- | --- | --- |
| `approach_speed_mps` | m/s | 접근 구간 평균 속도 |
| `enter_speed_mps` | m/s | 진입 시점 속도 |
| `min_dist_m` | m | 접근 중 횡단보도까지 최소 거리 |
| `stopped` | 0/1 | 정지(0.5 m/s 이하 1초 이상) 여부 |
| `stop_duration_s` | s | 가장 긴 정지 시간 |
| `n_ped_in_zone_max` | 명 | 접근~진입 사이 충돌 구간 안 보행자 수 최댓값 (0이면 판정 제외 대상) |
| `ped_in_zone_at_enter` | 0/1 | 진입 시점(`t_enter_s`)에 충돌 구간 안에 보행자가 있었는지 (1이면 미양보 근거) |
| `ped_min_dist_m` | m | 차량과 충돌 구간 안 보행자 사이 최소 거리 |

## 4. judged — `outputs/judged/<video>.csv` (육주영, `yield_classifier.rule`)

features의 컬럼 전부 + `pred` (`yield` / `no_yield` / `exclude`).

## 5. labels — `data/labels/<video>_<라벨러>.csv`

`docs/labeling_guide.md` 참고.
