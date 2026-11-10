# 단계별 CSV 스키마

> **초안 — 10.30 팀 합의로 확정, 11.10 기획서·기술서(11.04) 기준으로 개정.** 파트 사이의 경계이므로 컬럼을 바꾸면 이 문서를 먼저 고치고 팀에 공유한다.

## 공통 규칙

- 인코딩 **UTF-8 with BOM** (`encoding="utf-8-sig"`, 엑셀에서 한글이 깨지지 않게).
- `video` = 영상 파일 이름에서 확장자를 뺀 것 (예: `A_20261031_0800`). 출력 파일 이름도 `<video>.csv`.
- `cls` 값은 `vehicle` / `motorcycle` / `pedestrian` 셋뿐이다 (`configs/detect.yaml`의 매핑).
- 시간은 영상 시작부터 초(`_s`), 거리는 m(`_m`), 속도는 m/s(`_mps`).
- 미터 좌표: 원점 = 횡단보도 **좌상** 모서리, x축 = 좌상→우상(`width_m` 변), y축 = 좌상→좌하(`length_m` 변). 모서리 순서는 `configs/site_example.yaml`.
- 판정 값은 코드에서 `yield` / `no_yield`를 쓴다. 문서에서는 미양보 = 위반 의심으로 부른다(README 용어 대응표).

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
| `t_approach_s` | 접근 구간(진입선 앞 약 15 m)에 들어온 시각 |
| `t_enter_s` | 횡단보도 영역에 진입한 시각 (박스 하단 중심점 기준) |
| `t_exit_s` | 출구 ROI를 통과한 시각 |
| `t_stopline_s` | 정지선을 통과한 시각 (2단계, 정지선이 없으면 비움) |

## 3. signal — `outputs/signal/<video>.csv` (김민영, `traffic_light.color`)

한 행 = 신호 상태가 바뀐 구간 1개 (영상 전체의 빨간불 시간표).

| 컬럼 | 뜻 |
| --- | --- |
| `video` | 영상 |
| `t_start_s`, `t_end_s` | 구간 시작·끝 |
| `state` | `red` / `not_red` / `unknown` |

## 4. features — `outputs/features/<video>.csv` (김민영, `features.build`)

events의 컬럼 전부 + 아래 컬럼. MVP에서는 시계열을 요약값으로만 넣는다.

| 컬럼 | 단위 | 뜻 |
| --- | --- | --- |
| `approach_speed_mps` | m/s | 접근 구간 평균 속도 |
| `entry_speed` | m/s | 진입 시점 속도 |
| `min_dist_m` | m | 접근 중 횡단보도까지 최소 거리 |
| `n_ped_max` | 명 | 접근~진입 사이 횡단보도 위 보행자 수 최댓값 (0이면 판정 제외 대상) |
| `peds_at_entry` | 명 | 진입 시점(`t_enter_s`)에 횡단보도 위에 있던 보행자 수 (1 이상이면 미양보) |
| `stop_sec` | s | 정지 기준(0.5 m/s 이하)을 채운 가장 긴 시간. **참고값**, 판정 조건 아님 |
| `signal_at_stopline` | - | 정지선 통과 시각의 신호 상태 (`red` / `not_red` / `unknown`) |
| `stop_sec_before_line` | s | 정지선 직전 정지 시간 (2단계) |

## 5. judged — `outputs/judged/<video>.csv` (육주영, `rules.rule`)

features의 컬럼 전부 + 이벤트 로그 필드(기술서 3.3).

| 컬럼 | 뜻 |
| --- | --- |
| `intersection` | 교차로 이름 (`configs/site_*.yaml`의 `site`, 대시보드 필터 기준) |
| `wall_time`, `slot` | 실제 시각, 시간대(1시간, 예: 08–09시). 영상 파일 이름의 촬영 시작 시각 기준 |
| `vehicle_type` | 승용·버스·트럭·오토바이 |
| `ped_result` | `yield` / `no_yield` / `exclude` (+ 제외 사유 `ped_exclude_reason`) |
| `red_result` | `stopped` / `not_stopped` / `exclude` / `na` (적색 아님) |
| `clip` | 위반 의심 클립 경로 (`eval.clips`가 채움, 없으면 비움) |

## 6. labels — `data/labels/<video>_<라벨러>.csv`

`docs/labeling_guide.md` 참고.
