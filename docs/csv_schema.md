# 단계별 CSV 스키마

> **10.30 팀 합의로 확정, 11.10 기획서·기술서(11.04) 기준으로 개정, 같은 날 기술서 11.1(11.10) 반영, 1단계를 진입 순간 판정으로 개정(11.10).** 파트 사이의 경계이므로 컬럼을 바꾸면 이 문서를 먼저 고치고 팀에 공유한다.

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
| `coco_cls` | COCO 원래 클래스 (`car` / `bus` / `truck` / `motorcycle` / `person`). events에 트랙의 최빈값으로 옮기고, judged의 `vehicle_type`(승용·버스·트럭·오토바이)으로 바꾼다 |
| `conf` | 검출 신뢰도 |
| `x1, y1, x2, y2` | 박스 (픽셀) |
| `bx_m, by_m` | 박스 하단 중심점의 미터 좌표 |

## 2. events — `outputs/events/<video>.csv` (김민영, `tracking.events`)

한 행 = 우회전 이벤트 1건 (vehicle·motorcycle만).

| 컬럼 | 뜻 |
| --- | --- |
| `video`, `event_id`, `track_id`, `cls` | 식별 |
| `coco_cls` | 트랙의 COCO 클래스 최빈값 (`car` / `bus` / `truck` / `motorcycle`) |
| `t_enter_s` | 출구 횡단보도 영역에 진입한 시각 (박스 하단 중심점 기준). 이벤트의 기준 시각 |
| `t_stopline_s` | 정지선(없으면 진입 쪽 횡단보도 앞 기준선)을 통과한 시각. 궤적 전체에서 찾는다 (2단계) |
| `link_track_id` | 끊긴 트랙을 이어 본 이벤트(`link_gap_s`·`link_dist_box`)의 앞 트랙 id. 진입로를 지난 쪽이 이 트랙이고 `track_id`는 출구에 들어간 쪽. features는 두 트랙 경로를 합쳐 횡단보도 진입을 본다. 이어 보지 않은 이벤트는 비움 |

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
| `entry_speed` | m/s | 진입 시점 속도 (참고값) |
| `peds_lost` | 명 | 경로상 각 횡단보도 진입 직전 `merge_gap_s` 안에, 그 횡단보도 경계에서 `occluded_edge_m`보다 안쪽에서 끊긴 보행자 트랙 수의 합 (1 이상이고 `peds_at_entry`가 0이면 가림 의심 → 제외) |
| `peds_at_entry` | 명 | 경로상 각 횡단보도(`crosswalk`=`exit`, `other_crosswalks_px`)에 차량이 진입한 순간 그 위에 있던 보행자 수의 합 (1 이상이면 미양보, 0이면 양보) |
| `no_yield_at` | - | 진입 순간 보행자가 있던 횡단보도 이름 (`exit` 또는 `other_crosswalks_px`의 이름, 여러 개면 `;`로 구분, 없으면 비움) |
| `signal_at_stopline` | - | 정지선 통과 시각의 신호 상태 (`red` / `not_red` / `unknown`) |
| `stop_sec_before_line` | s | 정지선 앞 `stopline_zone_m` 안에서 정지 기준을 채운 가장 긴 시간 (2단계, `stop_min_duration_s` 이상이면 일시정지) |

## 5. judged — `outputs/judged/<video>.csv` (육주영, `rules.rule`)

features의 컬럼 전부 + 이벤트 로그 필드(기술서 3.3).

| 컬럼 | 뜻 |
| --- | --- |
| `intersection` | 교차로 이름 (`configs/site_*.yaml`의 `site`, 대시보드 필터 기준) |
| `wall_time`, `slot` | 실제 시각, 시간대(1시간, 예: 08–09시). 영상 파일 이름의 촬영 시작 시각 기준 |
| `vehicle_type` | 승용·버스·트럭·오토바이 |
| `ped_result` | `yield` / `no_yield` / `exclude` (+ 제외 사유 `ped_exclude_reason`: `occluded` 가림 의심, `not_right_turn` 우회전 아님(사람 확인 때만), `camera_moved` 카메라가 움직인 구간) |
| `red_result` | `stopped` / `not_stopped` / `exclude` / `na` (적색 아님 = 녹색·황색) (+ 제외 사유 `red_exclude_reason`: `signal_unknown` 신호 불명, `stopline_hidden` 정지선 가려짐, `rturn_signal` 우회전 전용 신호, `camera_moved` 카메라가 움직인 구간) |
| `status` | `auto` (규칙 판정 그대로). 사람이 확인한 이벤트는 labels에 행이 생기고 `eval.report`가 합쳐 `reviewed`로 센다 |
| `clip` | 위반 의심 클립 경로 (`eval.clips`가 채움, 없으면 비움) |

## 6. labels — `data/labels/` (사람이 단 정답)

자세한 규칙은 `docs/labeling_guide.md`.

| 파일 | 컬럼 |
| --- | --- |
| `<video>_<라벨러>.csv` (이벤트 라벨) | `video`, `event_id`, `cls`, `t_enter_s`, `ped_label`, `red_label`, `labeler`, `note` |
| `<video>_red.csv` (빨간불 시각) | `video`, `t_start_s`, `t_end_s`, `labeler` |

- `ped_label`: `yield` / `no_yield` / `exclude`. `red_label`: `stopped` / `not_stopped` / `exclude` / `na`.
