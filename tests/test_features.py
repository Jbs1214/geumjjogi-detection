# 담당: 김민영
"""features/build.py 테스트 (가짜 트랙, 실제 영상 없이). 실행: python -m unittest discover -s tests -t . -v
화면 10 px = 1 m. 출구(0~100 px), near(200~300 px), diag(400~500 px). 차량은 y=50에서 프레임마다 10 px(10 m/s) 왼쪽으로 간다:
near 진입 frame 5, 출구 진입 frame 25 (t_enter_s 2.5). diag에는 들어가지 않는다."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from features.build import NEW_COLUMNS, build_features, front_points, signal_at
from tracking.homography import compute_homography, to_meters

FPS = 10.0
EXIT = [[0, 0], [100, 0], [100, 100], [0, 100]]
SITE = {"crosswalk": {"corners_px": EXIT, "area_px": [], "width_m": 10.0, "length_m": 10.0},
        "other_crosswalks_px": {"near": [[200, 0], [300, 0], [300, 100], [200, 100]],
                                "diag": [[400, 0], [500, 0], [500, 100], [400, 100]]},
        "occluded_edge_m": 1.0,
        "stop_line_px": [[325, 0], [325, 100]],   # 세로선. 차는 오른쪽(x > 325)에서 와서 넘는다
        "stop_zone_px": [[300, 0], [355, 0], [355, 100], [300, 100]],   # 정지선 뒤 3 m(x 355) ~ near 시작선(x 300)
        "stop_min_duration_s": 1.0, "stop_move_min_px": 5, "stop_move_ratio": 0.1, "stop_strict_margin_ratio": 0.2}
SIGNAL = pd.DataFrame({"video": "A_20261031_0800", "t_start_s": [0.0, 1.0, 5.0], "t_end_s": [1.0, 5.0, 6.0],
                       "state": ["not_red", "red", "unknown"]})
CONFIG = {"merge_gap_s": 2.0, "speed_window_s": 0.5}
H = compute_homography(EXIT, 10.0, 10.0)


def track(tid, cls, frames, points):
    """frames마다 하단 중심 (x, y) 픽셀. points가 (x, y) 하나면 그 자리에 서 있다."""
    frames = list(frames)
    pts = np.array(points if isinstance(points, list) else [points] * len(frames), dtype=float)
    m = to_meters(H, pts)
    return pd.DataFrame({
        "frame": frames, "t_s": np.array(frames) / FPS, "track_id": tid, "cls": cls,
        "coco_cls": "car" if cls == "vehicle" else "person", "conf": 0.9,
        "x1": pts[:, 0] - 5, "y1": pts[:, 1] - 30, "x2": pts[:, 0] + 5, "y2": pts[:, 1], "bx_m": m[:, 0], "by_m": m[:, 1],
    })


CAR = track(1, "vehicle", range(41), [(350 - 10 * f, 50) for f in range(41)])
EVENTS = pd.DataFrame({"video": ["A_20261031_0800"], "event_id": [1], "track_id": [1], "cls": ["vehicle"],
                       "coco_cls": ["car"], "t_enter_s": [2.5], "t_stopline_s": [np.nan]})


def run(*peds, signal=None, car=CAR, events=EVENTS, site=SITE):
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        pd.concat([car, *peds]).to_csv(d / "tracks.csv", index=False, encoding="utf-8-sig")
        events.to_csv(d / "A_20261031_0800.csv", index=False, encoding="utf-8-sig")
        (d / "site.yaml").write_text(yaml.safe_dump(site), encoding="utf-8")
        (d / "detect.yaml").write_text(yaml.safe_dump(CONFIG), encoding="utf-8")
        if signal is not None:
            signal.to_csv(d / "signal.csv", index=False, encoding="utf-8-sig")
            signal = str(d / "signal.csv")
        out = build_features(str(d / "A_20261031_0800.csv"), str(d / "tracks.csv"), str(d / "site.yaml"),
                             signal, str(d / "detect.yaml"), out_dir=str(d / "out"))
        return pd.read_csv(out, encoding="utf-8-sig", keep_default_na=False).iloc[0]


class PedsAtEntryTest(unittest.TestCase):
    def test_ped_on_exit_at_entry(self):
        r = run(track(101, "pedestrian", range(61), (50, 80)))
        self.assertEqual((r["peds_at_entry"], r["no_yield_at"]), (1, "exit"))

    def test_ped_left_before_entry(self):
        walk = [(50, 80 + 4 * f) if f <= 20 else (50, 160) for f in range(61)]   # frame 20에 횡단보도 밖
        self.assertEqual(run(track(102, "pedestrian", range(61), walk))["peds_at_entry"], 0)

    def test_ped_on_near_at_near_entry(self):
        r = run(track(103, "pedestrian", range(11), (250, 80)))               # near 진입(frame 5) 때만 있음
        self.assertEqual((r["peds_at_entry"], r["no_yield_at"]), (1, "near"))

    def test_crosswalk_not_entered_is_ignored(self):
        self.assertEqual(run(track(104, "pedestrian", range(61), (450, 50)))["peds_at_entry"], 0)


GAP = {"exit": [[0, 100], [100, 100], [100, 130], [0, 130]],      # 출구 줄무늬 아래 ~ 연석 선(y 130)
       "diag": [[400, 100], [500, 100], [500, 130], [400, 130]]}


class CurbGapTest(unittest.TestCase):
    """ped_extra_px: 줄무늬~연석 사이 차도의 보행자도 그 횡단보도로 센다(기술서 2.2, 11.19)."""

    def test_ped_in_gap_at_exit_entry_is_counted(self):
        r = run(track(111, "pedestrian", range(61), (50, 115)), site={**SITE, "ped_extra_px": GAP})
        self.assertEqual((r["peds_at_entry"], r["no_yield_at"]), (1, "exit"))

    def test_without_gap_setting_ped_in_gap_is_not_counted(self):
        self.assertEqual(run(track(111, "pedestrian", range(61), (50, 115)))["peds_at_entry"], 0)

    def test_ped_on_sidewalk_beyond_curb_is_not_counted(self):
        r = run(track(112, "pedestrian", range(61), (50, 140)), site={**SITE, "ped_extra_px": GAP})
        self.assertEqual(r["peds_at_entry"], 0)

    def test_gap_of_crosswalk_not_entered_is_ignored(self):
        r = run(track(113, "pedestrian", range(61), (450, 115)), site={**SITE, "ped_extra_px": GAP})
        self.assertEqual(r["peds_at_entry"], 0)

    def test_ped_where_gap_overlaps_stripes_counted_once(self):
        overlap = {"exit": [[0, 90], [100, 90], [100, 130], [0, 130]]}
        r = run(track(114, "pedestrian", range(61), (50, 95)), site={**SITE, "ped_extra_px": overlap})
        self.assertEqual(r["peds_at_entry"], 1)

    def test_one_missing_frame_is_interpolated(self):
        frames = [f for f in range(61) if f != 25]                           # 출구 진입 순간만 검출 빠짐
        self.assertEqual(run(track(105, "pedestrian", frames, (50, 20)))["peds_at_entry"], 1)

    def test_long_gap_not_interpolated(self):
        frames = [*range(3), *range(48, 61)]                                  # 4.5초 빔
        self.assertEqual(run(track(106, "pedestrian", frames, (50, 20)))["peds_at_entry"], 0)

    def test_two_crosswalks_summed_in_time_order(self):
        r = run(track(101, "pedestrian", range(61), (50, 80)), track(103, "pedestrian", range(11), (250, 80)))
        self.assertEqual((r["peds_at_entry"], r["no_yield_at"]), (2, "near;exit"))


class LinkedEventTest(unittest.TestCase):
    """같은 차가 near를 지난 뒤 끊겨(트랙 1: frame 0~12) 다른 트랙(2: frame 16~40)으로 출구에 들어간 이벤트."""
    BROKEN = pd.concat([CAR[CAR.frame <= 12], CAR[CAR.frame >= 16].assign(track_id=2)])
    PED = track(103, "pedestrian", range(11), (250, 80))                    # near 진입(frame 5) 때만 있음

    def test_near_seen_through_linked_track(self):
        ev = EVENTS.assign(track_id=2, link_track_id=1)
        r = run(self.PED, car=self.BROKEN, events=ev)
        self.assertEqual((r["peds_at_entry"], r["no_yield_at"]), (1, "near"))

    def test_without_link_near_is_missed(self):
        r = run(self.PED, car=self.BROKEN, events=EVENTS.assign(track_id=2, link_track_id=np.nan))
        self.assertEqual(r["peds_at_entry"], 0)


class PedsLostTest(unittest.TestCase):
    def test_ended_inside_center_is_lost(self):
        r = run(track(107, "pedestrian", range(16), (50, 50)))                # t 1.5에 가운데(경계에서 5 m)에서 끝남
        self.assertEqual((r["peds_lost"], r["peds_at_entry"]), (1, 0))

    def test_ended_near_edge_is_not_lost(self):
        self.assertEqual(run(track(108, "pedestrian", range(16), (95, 50)))["peds_lost"], 0)   # 경계에서 0.5 m

    def test_ended_too_early_is_not_lost(self):
        self.assertEqual(run(track(109, "pedestrian", range(4), (50, 50)))["peds_lost"], 0)    # 진입 2초 전보다 먼저


class OutputTest(unittest.TestCase):
    def test_entry_speed(self):
        self.assertAlmostEqual(run()["entry_speed"], 10.0, places=1)

    def test_columns_and_stage2_empty(self):
        r = run()
        self.assertEqual(list(r.index), list(EVENTS.columns) + NEW_COLUMNS)
        self.assertEqual((r["signal_at_stopline"], r["stop_level"], r["stop_note"], r["no_yield_at"]), ("", "", "", ""))



def drive(stop_x, stop_frames, start_x=400):
    """start_x에서 프레임마다 10 px(10 m/s) 왼쪽으로 오다가 stop_x에서 stop_frames 프레임 서 있고 다시 간다(x 100까지).
    돌려주는 값: (트랙, 정지선 x 325를 넘은 시각, 출구 진입 시각)."""
    return move(list(range(start_x, stop_x, -10)) + [stop_x] * (stop_frames + 1) + list(range(stop_x - 10, 90, -10)))


def move(xs, height=30):
    """프레임마다 하단 중심 x가 xs인 차(박스 폭 10, 높이 height). 돌려주는 값은 drive와 같다."""
    car = track(1, "vehicle", range(len(xs)), [(x, 50) for x in xs]).assign(y1=50 - height)
    t_line = next(i for i, x in enumerate(xs) if x < 325) / FPS
    return car, t_line, next(i for i, x in enumerate(xs) if x <= 100) / FPS


def stage2(car, t_stopline, t_enter, site=SITE):
    r = run(signal=SIGNAL, car=car, events=EVENTS.assign(t_enter_s=t_enter, t_stopline_s=t_stopline), site=site)
    return r["stop_level"], r["stop_note"]


class SignalAtTest(unittest.TestCase):
    def test_state_of_segment(self):
        self.assertEqual([signal_at(SIGNAL, t) for t in (0.3, 2.0, 5.5)], ["not_red", "red", "unknown"])

    def test_boundary_belongs_to_next_segment(self):
        self.assertEqual(signal_at(SIGNAL, 1.0), "red")

    def test_no_time_or_outside_table_is_empty(self):
        self.assertEqual([signal_at(SIGNAL, np.nan), signal_at(SIGNAL, 99.0)], ["", ""])


class FrontPointsTest(unittest.TestCase):
    """정지선 x 325 세로선. 박스 x 310~320, 아래변 y 50."""
    BOX = pd.DataFrame({"x1": [310.0], "y1": [20.0], "x2": [320.0], "y2": [50.0]})

    def test_car_going_left_uses_left_corner(self):
        front, over = front_points(self.BOX, SITE["stop_line_px"], np.array([50.0, 50.0]))   # 출구가 왼쪽
        self.assertEqual((tuple(front[0]), over[0]), ((310.0, 50.0), 15.0))

    def test_car_going_right_uses_right_corner(self):
        front, over = front_points(self.BOX, SITE["stop_line_px"], np.array([500.0, 50.0]))  # 출구가 오른쪽
        self.assertEqual((tuple(front[0]), over[0]), ((320.0, 50.0), -5.0))


class StopLevelTest(unittest.TestCase):
    """정지 구간 x 300~355, 차 앞 끝 = 하단 중심 x − 5, 엄격 허용 폭 = 박스 높이 30 × 0.2 = 6 px(앞 끝 x 319까지)."""

    def test_stop_before_line_is_strict(self):
        self.assertEqual(stage2(*drive(stop_x=330, stop_frames=20)), ("strict", ""))   # 앞 끝이 정지선에 맞춰 2초

    def test_stop_past_line_before_crosswalk_is_minor(self):
        self.assertEqual(stage2(*drive(stop_x=315, stop_frames=20)), ("minor", ""))    # 앞 끝이 정지선을 15 px 넘음

    def test_stop_before_line_then_creep_past_is_strict(self):
        xs = list(range(400, 330, -10)) + [330] * 21 + [325, 320, 315] + [315] * 20 + list(range(305, 90, -10))
        self.assertEqual(stage2(*move(xs)), ("strict", ""))   # 앞 끝 325에서 2초 → 슬금 310까지 → 2초 (기술서 2.3, 11.19)

    def test_short_stop_is_none(self):
        self.assertEqual(stage2(*drive(stop_x=330, stop_frames=5)), ("none", ""))      # 0.5초

    def test_stop_outside_zone_is_none(self):
        self.assertEqual(stage2(*drive(stop_x=390, stop_frames=20, start_x=450)), ("none", ""))   # 정지선 6.5 m 앞

    def test_no_stop_is_none(self):
        self.assertEqual(stage2(*drive(stop_x=330, stop_frames=0)), ("none", ""))

    def test_jitter_within_5px_is_stop(self):
        xs = list(range(400, 330, -10)) + [330, 332, 328] * 7 + list(range(320, 90, -10))
        self.assertEqual(stage2(*move(xs)), ("strict", ""))

    def test_allowed_move_grows_with_box_height(self):
        xs = list(range(400, 340, -10)) + [340 - 0.8 * k for k in range(11)] + list(range(322, 90, -10))  # 1초에 8 px
        self.assertEqual(stage2(*move(xs, height=100)), ("strict", ""))   # 허용 max(5, 10) px
        self.assertEqual(stage2(*move(xs, height=30)), ("none", ""))      # 허용 max(5, 3) px

    def test_no_stopline_time_is_hidden(self):
        self.assertEqual(stage2(CAR, np.nan, 2.5), ("", "stopline_hidden"))

    def test_track_starting_inside_zone_is_unseen(self):
        self.assertEqual(stage2(CAR, 0.3, 2.5), ("", "zone_unseen"))     # 첫 앞 끝 x 345가 이미 구간 안

    def test_long_gap_in_zone_is_hidden(self):
        car, t_line, t_enter = drive(stop_x=330, stop_frames=0)
        car = car[(car.frame <= 6) | (car.frame >= 25)]                  # 정지선 근처 1.9초 동안 검출 없음
        self.assertEqual(stage2(car, t_line, t_enter), ("", "stopline_hidden"))

    def test_site_without_stop_zone_is_empty(self):
        site = {**SITE, "stop_zone_px": []}
        self.assertEqual(stage2(*drive(stop_x=330, stop_frames=20), site=site), ("", ""))


if __name__ == "__main__":
    unittest.main()
