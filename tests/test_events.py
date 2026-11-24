# 담당: 김민영
"""tracking/events.py 테스트 (가짜 트랙, 실제 영상 없이). 실행: python -m unittest discover -s tests -t . -v"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tracking.events import COLUMNS, drop_contained, exit_area, extract_events, find_events, stopline_time

FPS = 10.0
APPROACH = [[100, 0], [200, 0], [200, 100], [100, 100]]  # 오른쪽 위
EXIT = [[0, 200], [100, 200], [100, 300], [0, 300]]      # 왼쪽 아래 (homography 사각형)
AREA = [[0, 150], [100, 150], [100, 300], [0, 300]]      # 사각형 위로 블록이 더 나온 판정 영역


def path(track_id, points, cls="vehicle", coco="car", t0=0.0, w=10, h=20, conf=0.9):
    """points = 하단 중심 픽셀 [(x, y), ...]를 0.1초 간격 프레임으로. 박스 폭 w(기본 10), 높이 h."""
    f0 = round(t0 * FPS)
    return pd.DataFrame([{
        "frame": f0 + i, "t_s": (f0 + i) / FPS, "track_id": track_id, "cls": cls, "coco_cls": coco, "conf": conf,
        "x1": x - w / 2, "y1": y - h, "x2": x + w / 2, "y2": y, "bx_m": 0.0, "by_m": 0.0,
    } for i, (x, y) in enumerate(points)])


RIGHT_TURN = [(150, 50), (150, 150), (50, 250)]  # 진입로 → 중간 → 출구


class FindEventsTest(unittest.TestCase):
    def test_approach_then_exit_is_event(self):
        ev = find_events(path(1, RIGHT_TURN), APPROACH, EXIT)
        self.assertEqual(list(ev["track_id"]), [1])
        self.assertAlmostEqual(ev["t_enter_s"].iloc[0], 0.2)

    def test_exit_without_approach_is_not_event(self):
        self.assertTrue(find_events(path(1, [(150, 150), (50, 250)]), APPROACH, EXIT).empty)

    def test_approach_without_exit_is_not_event(self):
        self.assertTrue(find_events(path(1, [(150, 50), (150, 150), (300, 300)]), APPROACH, EXIT).empty)

    def test_exit_before_approach_is_not_event(self):
        self.assertTrue(find_events(path(1, RIGHT_TURN[::-1]), APPROACH, EXIT).empty)

    def test_reentry_uses_first_entry(self):
        pts = RIGHT_TURN + [(150, 250), (50, 250)]  # 출구 → 밖 → 다시 출구
        ev = find_events(path(1, pts), APPROACH, EXIT)
        self.assertEqual(len(ev), 1)
        self.assertAlmostEqual(ev["t_enter_s"].iloc[0], 0.2)

    def test_pedestrian_ignored_motorcycle_kept(self):
        df = pd.concat([path(1, RIGHT_TURN, cls="pedestrian", coco="person"),
                        path(2, RIGHT_TURN, cls="motorcycle", coco="motorcycle")])
        self.assertEqual(list(find_events(df, APPROACH, EXIT)["track_id"]), [2])

    def test_area_catches_blocks_outside_rectangle(self):
        pts = [(150, 50), (50, 170)]  # 사각형 밖, 판정 영역 안
        self.assertTrue(find_events(path(1, pts), APPROACH, EXIT).empty)
        self.assertEqual(len(find_events(path(1, pts), APPROACH, AREA)), 1)

    def test_coco_cls_is_track_mode(self):
        df = path(1, RIGHT_TURN + [(50, 260)], coco="truck")
        df.loc[0, "coco_cls"] = "car"
        self.assertEqual(find_events(df, APPROACH, EXIT)["coco_cls"].iloc[0], "truck")

    def test_sorted_by_enter_time(self):
        df = pd.concat([path(1, RIGHT_TURN, t0=5.0), path(2, RIGHT_TURN, t0=1.0)])
        self.assertEqual(list(find_events(df, APPROACH, EXIT)["track_id"]), [2, 1])


DETECT = {"link_gap_s": 3.0, "link_dist_box": 2.0, "merge_size_min": 0.5, "dedup_gap_s": 2.0, "dedup_contain_min": 0.8, "dedup_frames_min": 0.8}


def dedup(df, frames_min=0.8):
    return drop_contained(df, find_events(df, APPROACH, EXIT), 2.0, 0.8, frames_min)


def big(track_id, **kw):
    """RIGHT_TURN을 도는 큰 박스(폭 40, 높이 40). 기본 박스(폭 10, 높이 20)가 그 안에 다 든다."""
    return path(track_id, RIGHT_TURN, w=40, h=40, **kw)


class DropContainedTest(unittest.TestCase):
    def test_inner_box_event_dropped(self):
        self.assertEqual(list(dedup(pd.concat([path(1, RIGHT_TURN), big(2)]))["track_id"]), [2])

    def test_side_by_side_kept(self):
        other = path(2, [(x + 30, y) for x, y in RIGHT_TURN])       # 옆 차: 박스가 안 겹침
        self.assertEqual(len(dedup(pd.concat([path(1, RIGHT_TURN), other]))), 2)

    def test_far_in_time_kept(self):
        self.assertEqual(len(dedup(pd.concat([path(1, RIGHT_TURN), big(2, t0=3.0)]))), 2)

    def test_motorcycle_not_compared(self):
        df = pd.concat([path(1, RIGHT_TURN, cls="motorcycle", coco="motorcycle"), big(2)])
        self.assertEqual(len(dedup(df)), 2)

    def test_contained_in_few_frames_kept(self):
        inner = path(1, [(190, 50), (190, 150), (50, 250)])         # 세 프레임 중 마지막에만 큰 박스 안
        self.assertEqual(len(dedup(pd.concat([inner, big(2)]))), 2)

    def test_coco_cls_from_higher_conf_track(self):
        ev = dedup(pd.concat([path(1, RIGHT_TURN, coco="bus", conf=0.9), big(2, coco="truck", conf=0.5)]))
        self.assertEqual((list(ev["track_id"]), list(ev["coco_cls"])), ([2], ["bus"]))

    def test_off_when_zero(self):
        self.assertEqual(len(dedup(pd.concat([path(1, RIGHT_TURN), big(2)]), frames_min=0)), 2)

    def test_extract_reads_config(self):
        self.assertEqual(list(extract(pd.concat([path(1, RIGHT_TURN), big(2)]))["track_id"]), [2])


A_PART = [(150, 50), (150, 90)]       # 진입로 안에서 끝남 (박스 폭 10 → 이어 보기 거리 20 px)
B_PART = [(150, 105), (50, 250)]      # 진입로 밖에서 시작해 출구로


def broken(gap_s=1.0, b_start=(150, 105), cls_b="vehicle"):
    """한 차가 진입로 끝(t 0.1)에서 끊기고 gap_s 뒤 b_start에서 다른 트랙으로 다시 잡힌 것."""
    return pd.concat([path(1, A_PART), path(2, [b_start, (50, 250)], cls=cls_b, t0=0.1 + gap_s)])


def linked(df):
    return find_events(df, APPROACH, EXIT, link_gap_s=3.0, link_dist_box=2.0)


class LinkBrokenTest(unittest.TestCase):
    def test_broken_track_is_linked(self):
        ev = linked(broken())
        self.assertEqual(list(ev["track_id"]), [2])
        self.assertAlmostEqual(ev["t_enter_s"].iloc[0], 1.2)
        self.assertEqual(ev["link_track_id"].iloc[0], 1)                  # 앞 트랙 id를 남김

    def test_normal_event_has_no_link(self):
        df = pd.concat([path(1, RIGHT_TURN), broken().assign(track_id=lambda d: d.track_id + 10)])
        ev = linked(df).set_index("track_id")["link_track_id"]
        self.assertTrue(pd.isna(ev[1]))
        self.assertEqual(ev[12], 11)

    def test_off_by_default(self):
        self.assertTrue(find_events(broken(), APPROACH, EXIT).empty)

    def test_gap_too_long(self):
        self.assertTrue(linked(broken(gap_s=3.5)).empty)

    def test_too_far(self):
        self.assertTrue(linked(broken(b_start=(150, 125))).empty)   # 35 px > 20 px

    def test_very_different_size_not_linked(self):
        # 진입로에서 끝난 큰 오검출 박스(폭 100·높이 60 → 반경 200 px)와 폭 10·높이 20 차: 크기 조건이 있으면 이어 보지 않는다
        df = pd.concat([path(1, A_PART, w=100, h=60), path(2, B_PART, t0=1.1)])
        self.assertEqual(list(linked(df)["track_id"]), [2])
        self.assertTrue(find_events(df, APPROACH, EXIT, link_gap_s=3.0, link_dist_box=2.0, size_min=0.5).empty)

    def test_similar_size_linked_with_size_check(self):
        ev = find_events(broken(), APPROACH, EXIT, link_gap_s=3.0, link_dist_box=2.0, size_min=0.5)
        self.assertEqual(list(ev["track_id"]), [2])

    def test_other_class_not_linked(self):
        self.assertTrue(linked(broken(cls_b="motorcycle")).empty)

    def test_approach_track_already_event_not_reused(self):
        df = pd.concat([path(1, RIGHT_TURN), path(2, B_PART, t0=0.5)])   # 1은 이미 우회전
        self.assertEqual(list(linked(df)["track_id"]), [1])

    def test_one_to_one_nearest(self):
        df = pd.concat([path(1, A_PART), path(2, B_PART, t0=1.0), path(3, [(150, 101), (50, 250)], t0=1.0)])
        self.assertEqual(list(linked(df)["track_id"]), [3])          # 3이 더 가까움(11 px < 15 px), 2는 짝 없음


class ExitAreaTest(unittest.TestCase):
    def test_empty_area_falls_back_to_corners(self):
        self.assertEqual(exit_area({"crosswalk": {"corners_px": EXIT, "area_px": []}}), EXIT)
        self.assertEqual(exit_area({"crosswalk": {"corners_px": EXIT}}), EXIT)

    def test_area_used_when_given(self):
        self.assertEqual(exit_area({"crosswalk": {"corners_px": EXIT, "area_px": AREA}}), AREA)


STOP_LINE = [[100, 120], [200, 120]]  # 진입로 아래 가로선. 출구(아래쪽)가 지난 쪽
AFTER = (50, 250)                     # 출구 영역 안의 점


class StoplineTimeTest(unittest.TestCase):
    def test_first_frame_past_line(self):
        self.assertAlmostEqual(stopline_time(path(1, RIGHT_TURN), STOP_LINE, AFTER), 0.1)

    def test_jitter_on_line_uses_last_crossing(self):
        pts = [(150, 50), (150, 125), (150, 115), (150, 125), (150, 118), (150, 130), (50, 250)]
        self.assertAlmostEqual(stopline_time(path(1, pts), STOP_LINE, AFTER), 0.5)

    def test_track_starting_past_line_has_no_time(self):
        self.assertTrue(np.isnan(stopline_time(path(1, [(150, 150), (50, 250)]), STOP_LINE, AFTER)))

    def test_line_is_extended_beyond_its_ends(self):
        line = [[300, 120], [400, 120]]   # 차가 끝점 바깥을 지남
        self.assertAlmostEqual(stopline_time(path(1, RIGHT_TURN), line, AFTER), 0.1)


def extract(df, **site):
    """가짜 트랙으로 extract_events를 돌린 결과 표."""
    with tempfile.TemporaryDirectory() as d:
        tracks = Path(d) / "A_20261031_0800.csv"
        df.to_csv(tracks, index=False, encoding="utf-8-sig")
        site_path = Path(d) / "site.yaml"
        site_path.write_text(yaml.safe_dump({"approach_roi_px": APPROACH,
                                             "crosswalk": {"corners_px": EXIT, "area_px": []}, **site}), encoding="utf-8")
        config = Path(d) / "detect.yaml"
        config.write_text(yaml.safe_dump(DETECT), encoding="utf-8")
        return pd.read_csv(extract_events(str(tracks), str(site_path), str(config), out_dir=d), encoding="utf-8-sig")


class ExtractStoplineTest(unittest.TestCase):
    def test_stopline_time_written(self):
        out = extract(path(1, RIGHT_TURN), stop_line_px=STOP_LINE)
        self.assertAlmostEqual(out["t_stopline_s"].iloc[0], 0.1)

    def test_linked_event_uses_both_tracks(self):
        out = extract(broken(), stop_line_px=[[100, 100], [200, 100]])   # 앞 트랙은 선 앞(y 90)에서 끝, 뒤 트랙은 선 뒤(y 105)에서 시작
        self.assertEqual(list(out["track_id"]), [2])
        self.assertAlmostEqual(out["t_stopline_s"].iloc[0], 1.1)

    def test_crossing_after_exit_entry_is_ignored(self):
        pts = RIGHT_TURN + [(150, 10), (150, 150)]   # 선(y 20) 뒤에서 시작, 출구 진입 뒤에야 선을 넘음
        self.assertTrue(extract(path(1, pts), stop_line_px=[[100, 20], [200, 20]])["t_stopline_s"].isna().all())

    def test_unset_stop_line_leaves_empty(self):
        self.assertTrue(extract(path(1, RIGHT_TURN), stop_line_px=[[0, 0], [0, 0]])["t_stopline_s"].isna().all())


class ExtractEventsTest(unittest.TestCase):
    def test_csv_schema_and_ids(self):
        with tempfile.TemporaryDirectory() as d:
            tracks = Path(d) / "A_20261031_0800.csv"
            pd.concat([path(7, RIGHT_TURN, t0=3.0), path(4, RIGHT_TURN)]).to_csv(tracks, index=False, encoding="utf-8-sig")
            site = Path(d) / "site.yaml"
            site.write_text(yaml.safe_dump({"approach_roi_px": APPROACH,
                                            "crosswalk": {"corners_px": EXIT, "area_px": []}}), encoding="utf-8")
            config = Path(d) / "detect.yaml"
            config.write_text(yaml.safe_dump(DETECT), encoding="utf-8")
            out = pd.read_csv(extract_events(str(tracks), str(site), str(config), out_dir=d), encoding="utf-8-sig")
        self.assertEqual(list(out.columns), COLUMNS)
        self.assertEqual(list(out["event_id"]), [1, 2])
        self.assertEqual(list(out["track_id"]), [4, 7])
        self.assertEqual(set(out["video"]), {"A_20261031_0800"})
        self.assertTrue(out["t_stopline_s"].isna().all())   # 정지선을 지정하지 않은 설정


if __name__ == "__main__":
    unittest.main()
