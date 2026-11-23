# 담당: 김민영
"""tracking/roi_tool.py 화면·단계 진행 테스트 (합성 그림, 실제 영상·창 없이). 실행: python -m unittest discover -s tests -t . -v"""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from tracking.roi_tool import (BAR_H, COLORS, EX_H, EX_W, STEPS, build_site, collect, curb_fills, draw, example, intro,
                                merge, next_hint, split_corner, update_site)

BLANK = np.full((900, 1600, 3), 90, np.uint8)


class FakeVideo:
    duration = 10.0

    def frame_at(self, sec):
        return np.zeros((900, 1600, 3), np.uint8), sec   # 화면 최대 크기와 같아 배율 1 (클릭 좌표 = 원본 좌표)


SQUARE = [(10, 10), (20, 10), (20, 20), (10, 20)]
CLICKS = {"click1": SQUARE[:1], "click3": SQUARE[:3], "click4": SQUARE,
          "curb_right": [(26, 0), (26, 30)]}   # SQUARE(진입 횡단보도) 오른쪽 6 px 떨어진 연석 선


def run(keys: list, steps: list = STEPS, existing: dict | None = None) -> dict | None:
    """창 없이 collect를 돌린다. keys = waitKey가 차례로 돌려줄 키, CLICKS 이름 = 그 점들 클릭."""
    callback, keys = {}, list(keys)

    def wait(_):
        key = keys.pop(0) if keys else 27
        if key in CLICKS:
            for x, y in CLICKS[key]:
                callback["f"](cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
            return 255
        return key

    with patch("cv2.namedWindow"), patch("cv2.imshow"), patch("cv2.destroyWindow"), \
            patch("cv2.setMouseCallback", lambda _w, f: callback.setdefault("f", f)), \
            patch("cv2.waitKey", wait), patch("builtins.input", side_effect=EOFError), patch("builtins.print"):
        return collect(FakeVideo(), 0.0, steps, existing)


class ScreenTest(unittest.TestCase):
    def test_every_step_draws_bar_card_and_keys(self):
        for i in range(len(STEPS)):
            img = draw(BLANK, 1.0, 0, 0, {}, [], STEPS, i)
            self.assertFalse(np.array_equal(img[:BAR_H], BLANK[:BAR_H]))          # 진행 막대
            self.assertFalse(np.array_equal(img[200:300, 20:400], BLANK[200:300, 20:400]))  # 미션 카드
            self.assertFalse(np.array_equal(img[-20:], BLANK[-20:]))              # 키 안내
            self.assertTrue(np.array_equal(img[500:800, 600:1500], BLANK[500:800, 600:1500]))  # 장면은 가리지 않는다

    def test_help_hidden_keeps_scene_clear(self):
        img = draw(BLANK, 1.0, 0, 0, {}, [], STEPS, 0, show_help=False)
        self.assertTrue(np.array_equal(img[200:800, :], BLANK[200:800, :]))

    def test_card_moves_away_from_cursor(self):
        left = draw(BLANK, 1.0, 0, 0, {}, [], STEPS, 0, cursor=(1500, 800))
        right = draw(BLANK, 1.0, 0, 0, {}, [], STEPS, 0, cursor=(100, 200))
        self.assertFalse(np.array_equal(left[150:250, 20:300], BLANK[150:250, 20:300]))
        self.assertFalse(np.array_equal(right[150:250, 1300:1580], BLANK[150:250, 1300:1580]))
        self.assertTrue(np.array_equal(right[300:400, 20:90], BLANK[300:400, 20:90]))

    def test_intro_and_final_screens(self):
        self.assertFalse(np.array_equal(intro(BLANK, STEPS), BLANK))
        final = draw(BLANK, 1.0, 0, 0, {"corners_px": [[1, 1], [9, 1], [9, 9], [1, 9]]}, [], STEPS, len(STEPS))
        self.assertFalse(np.array_equal(final[100:200, 20:400], BLANK[100:200, 20:400]))

    def test_narrow_vertical_video(self):
        img = draw(np.full((900, 506, 3), 90, np.uint8), 1.0, 0, 0, {}, [], STEPS, 4, cursor=(480, 880))
        self.assertEqual(img.shape, (900, 506, 3))

    def test_example_picture_for_every_step(self):
        for s in STEPS:
            pic, _ = example(s.key, 0, True, COLORS[s.key])
            self.assertEqual(pic.shape, (EX_H, EX_W, 3))
            self.assertGreater(len(np.unique(pic.reshape(-1, 3), axis=0)), 3)


def fill(poly, shape=(400, 500)):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.array(poly, np.int32)], 1)
    return m


class CornerTest(unittest.TestCase):
    NEAR = [[100, 100], [200, 100], [200, 200], [100, 200]]
    EXIT = [[150, 150], [300, 150], [300, 250], [150, 250]]   # NEAR 오른쪽 아래 모서리와 50x50 겹침

    def test_overlap_becomes_corner_and_nothing_overlaps(self):
        out, corners = split_corner({"exit": self.EXIT, "near": self.NEAR})
        self.assertEqual(len(corners), 1)
        c, e, n = fill(corners[0]), fill(out["exit"]), fill(out["near"])
        self.assertTrue(c[175, 175] and not e[175, 175] and not n[175, 175])   # 겹친 자리는 corner만
        self.assertTrue(e[200, 250] and n[120, 120])                            # 나머지는 그대로
        self.assertLess(int((c & e).sum() + (c & n).sum() + (e & n).sum()), 200)  # 경계 근처 오차만
        before = fill(self.EXIT) | fill(self.NEAR)
        self.assertLess(int((before & ~(c | e | n)).sum()), 600)               # 빈 곳은 경계 1 px 이음새 정도

    def test_touching_makes_no_corner(self):
        out, corners = split_corner({"exit": [[200, 100], [300, 100], [300, 200], [201, 200]], "near": self.NEAR})
        self.assertEqual(corners, [])
        self.assertEqual(out["near"], self.NEAR)

    def test_corner_preview_on_curb_step(self):
        i = next(n for n, s in enumerate(STEPS) if s.key == "curb_px")
        shown = {"corners_px": [[230, 100], [330, 100], [330, 200], [230, 200]], "near_px": self.NEAR}
        apart = draw(BLANK, 1.0, 0, 0, shown, [[215, 80], [215, 220]], STEPS, i, show_help=False)   # near 끝 → 연석, 출구와 안 겹침 (카드가 가리지 않게)
        meet = draw(BLANK, 1.0, 0, 0, shown, [[240, 80], [240, 220]], STEPS, i, show_help=False)    # near를 늘인 곳이 출구와 겹침 → 빨강
        red = lambda img: int((img[:, :, 2].astype(int) - img[:, :, 0] > 40).sum())   # 반투명 빨강
        self.assertGreater(red(meet[90:210, 225:245]), red(apart[90:210, 225:245]))


class CurbFillTest(unittest.TestCase):
    BAND = {"near_px": [[0, 0], [100, 0], [100, 50], [0, 50]]}   # 가로로 긴 횡단보도(오른쪽 끝 = 줄무늬 끝)

    def area_mask(self, poly):
        m = np.zeros((200, 300), np.uint8)
        cv2.fillPoly(m, [np.array(poly, np.int32) + 50], 1)
        return m

    def test_straight_curb_fills_band_to_line(self):
        fills = curb_fills(self.BAND, [[[130, -20], [130, 70]]])
        m = self.area_mask(fills["near"][0])
        self.assertTrue(m[25 + 50, 115 + 50] and m[2 + 50, 128 + 50] and m[48 + 50, 128 + 50])
        self.assertFalse(m[25 + 50, 135 + 50] or m[60 + 50, 115 + 50] or m[25 + 50, 50 + 50])   # 연석 밖·띠 밖·줄무늬 쪽 아님

    def test_curved_curb_followed(self):
        fills = curb_fills(self.BAND, [[[130, -20], [130, 20], [140, 40], [160, 70]]])
        poly = fills["near"][0]
        self.assertIn([130, 20], poly)
        self.assertIn([140, 40], poly)   # 두 교점 사이의 연석 점을 따라간다

    def test_far_or_missing_curb_not_filled(self):
        self.assertEqual(curb_fills(self.BAND, [[[200, -20], [200, 70]]]), {})   # 테두리 100 × 0.5 = 50 px보다 멂
        self.assertEqual(curb_fills(self.BAND, [[[130, 60], [130, 90]]]), {})    # 띠 옆을 지나 안 만남

    def test_short_curb_line_still_fills(self):
        # 연석 선이 띠 위쪽 테두리(y 0)까지 안 닿게 짧게 그려져도(y 10~70) 양 끝을 늘여 채운다
        fills = curb_fills(self.BAND, [[[130, 10], [130, 70]]])
        m = self.area_mask(fills["near"][0])
        self.assertTrue(m[2 + 50, 128 + 50] and m[48 + 50, 128 + 50])

    def test_both_ends(self):
        fills = curb_fills(self.BAND, [[[130, -20], [130, 70]], [[-20, -20], [-20, 70]]])
        self.assertEqual(len(fills["near"]), 2)


class MergeTest(unittest.TestCase):
    STRIPES = [[0, 0], [100, 0], [100, 50], [0, 50]]

    def test_gap_beside_crosswalk_becomes_one_polygon(self):
        poly = merge(self.STRIPES, [[100, 0], [130, 0], [130, 50], [100, 50]])
        m = np.zeros((80, 160), np.uint8)
        cv2.fillPoly(m, [np.array(poly, np.int32)], 1)
        self.assertTrue(m[25, 5] and m[25, 115] and m[25, 128])   # 줄무늬 + 빈 차도
        self.assertFalse(m[25, 140] or m[60, 50])
        self.assertLessEqual(len(poly), 6)                         # 계단 모양이 아니라 꼭짓점 몇 개로

    def test_small_gap_is_joined_far_one_refused(self):
        self.assertIsNotNone(merge(self.STRIPES, [[102, 0], [130, 0], [130, 50], [102, 50]]))   # 2 px 틈
        self.assertIsNone(merge(self.STRIPES, [[120, 0], [150, 0], [150, 50], [120, 50]]))


class HintTest(unittest.TestCase):
    def test_corners_one_by_one_then_enter(self):
        corners = STEPS[0]
        self.assertEqual([next_hint(corners, i)[0][0] for i in range(4)], list("①②③④"))
        self.assertIn("왼쪽 위", next_hint(corners, 0)[0])
        self.assertEqual(next_hint(corners, 4)[1], True)

    def test_polygon_ready_after_four_points(self):
        zone = next(s for s in STEPS if s.key == "stop_zone_px")
        self.assertFalse(next_hint(zone, 3)[1])
        self.assertTrue(next_hint(zone, 4)[1])
        self.assertTrue(next_hint(zone, 7)[1])   # 점을 더 찍어도 된다

    def test_every_counted_step_names_each_point(self):
        for s in STEPS:
            if isinstance(s.n, int):
                self.assertEqual(len(s.points), s.n, s.key)


class CollectTest(unittest.TestCase):
    def test_intro_esc_cancels(self):
        self.assertIsNone(run([27]))

    def test_final_screen_enter_saves_and_esc_cancels(self):
        steps = [s for s in STEPS if s.key == "stop_zone_px"]
        self.assertEqual(run([ord(" "), "click4", 13, 13], steps), {"stop_zone_px": [[10, 10], [20, 10], [20, 20], [10, 20]]})
        self.assertIsNone(run([ord(" "), "click4", 13, 27], steps))

    def test_polygon_needs_four_points(self):
        steps = [s for s in STEPS if s.key == "stop_zone_px"]
        self.assertIsNone(run([ord(" "), "click3", 13, 13], steps))   # 3점 Enter는 넘어가지 않는다(키가 다 떨어져 Esc)

    def test_backspace_goes_back_and_keeps_points(self):
        # 다음 단계에서 Backspace → 앞 단계 점이 찍힌 채로 열려 Enter로 그대로 넘기고, 찍다 만 점도 돌아오면 남아 있다
        steps = [s for s in STEPS if s.key in ("near_px", "diag_px")]
        done = run([ord(" "), "click4", 13, "click1", 8, 13, "click3", 13, 13], steps)  # 8 = Backspace
        self.assertEqual(done["near_px"], [list(p) for p in SQUARE])
        self.assertEqual(len(done["diag_px"]), 4)   # 찍다 만 1점 + 새로 3점

    def test_curb_step_lines_then_done(self):
        # 진입 횡단보도 4점 → 연석 선 1개(2점 Enter) → 점 없이 Enter = 단계 끝. 1점에서 Enter는 무시
        steps = [s for s in STEPS if s.key in ("near_px", "curb_px")]
        done = run([ord(" "), "click4", 13, "click1", 13, ord("r"), "curb_right", 13, 13, 13], steps)
        self.assertEqual(done["curb_px"], [[list(p) for p in CLICKS["curb_right"]]])

    def test_curb_backspace_keeps_lines(self):
        steps = [s for s in STEPS if s.key in ("curb_px", "stop_line_px")]
        done = run([ord(" "), "curb_right", 13, 13, 8, 13, ord("s"), 13], steps)   # 정지선 단계에서 돌아와 그대로 넘김
        self.assertEqual(done["curb_px"], [[list(p) for p in CLICKS["curb_right"]]])

    def test_curb_skip(self):
        steps = [s for s in STEPS if s.key == "curb_px"]
        self.assertEqual(run([ord(" "), ord("s"), 13], steps), {"curb_px": []})

    def test_backspace_from_final_screen(self):
        steps = [s for s in STEPS if s.key == "stop_line_px"]
        self.assertEqual(run([ord(" "), ord("s"), 8, "click1", "click1", 13, 13], steps), {"stop_line_px": [[10, 10], [10, 10]]})

    def test_skip_marks_step_empty(self):
        steps = [s for s in STEPS if s.key in ("stop_line_px", "signal_roi_px")]
        self.assertEqual(run([ord(" "), ord("s"), ord("s"), 13], steps), {"stop_line_px": [], "signal_roi_px": []})

    def test_build_site_puts_near_diag_under_other_crosswalks(self):
        near = [[0, 0], [5, 0], [5, 5]]
        points = {"corners_px": [[0, 0], [10, 0], [10, 10], [0, 10]], "area_px": [], "near_px": near, "diag_px": [],
                  "approach_roi_px": [[0, 0], [1, 0], [1, 1], [0, 1]], "stop_line_px": [], "stop_zone_px": [],
                  "signal_roi_px": []}
        with patch("builtins.print"):
            site = build_site("A_20261123_0800.mp4", points, 0.0, 0.0)
        self.assertEqual((site["site"], site["crosswalk"]["corners_px"]), ("A", points["corners_px"]))
        self.assertEqual(site["other_crosswalks_px"], {"near": near})   # 건너뛴 diag는 넣지 않는다
        self.assertNotIn("near_px", site)
        self.assertEqual(site["ped_extra_px"], {})                       # 새로 그리지 않음(11.22)

    def test_build_site_merges_gaps_into_crosswalks(self):
        corners = [[0, 0], [100, 0], [100, 50], [0, 50]]
        near = [[0, 100], [100, 100], [100, 150], [0, 150]]
        points = {"corners_px": corners, "area_px": [], "near_px": near, "diag_px": [],
                  "curb_px": [[[130, -30], [130, 160]], [[-20, -30], [-20, 160]]],   # 양쪽 끝 연석
                  "approach_roi_px": [[0, 0], [1, 0], [1, 1], [0, 1]], "stop_line_px": [], "stop_zone_px": [], "signal_roi_px": []}
        with patch("builtins.print"):
            site = build_site("A_20261123_0800.mp4", points, 0.0, 0.0)
        self.assertEqual(site["crosswalk"]["corners_px"], corners)                 # homography 사각형은 그대로
        self.assertEqual({x for x, _ in site["crosswalk"]["area_px"]} >= {-20, 130}, True)   # 판정 영역 = 사각형 + 양쪽 빈 차도
        self.assertEqual({x for x, _ in site["other_crosswalks_px"]["near"]} >= {-20, 130}, True)
        self.assertNotIn("curb_px", site)
        self.assertNotIn("corner", site["other_crosswalks_px"])   # 두 횡단보도가 떨어져 있어 겹친 곳 없음

    def test_steps_order(self):
        keys = [s.key for s in STEPS]
        self.assertEqual(keys[:4], ["corners_px", "near_px", "diag_px", "curb_px"])   # 횡단보도 다 찍은 뒤 연석 선
        self.assertNotIn("area_px", keys)   # 판정 영역은 연석까지 채워 만든다(직접 찍는 단계 없음)

    def test_update_site_only_near_keeps_other_names(self):
        site = {"crosswalk": {"corners_px": [[0, 0], [10, 0], [10, 10], [0, 10]], "area_px": [], "width_m": 0, "length_m": 0},
                "other_crosswalks_px": {"near": [[1, 1]], "diag": [[2, 2]]}}
        with patch("builtins.print"):
            update_site(site, {"near_px": [[3, 3], [4, 3], [4, 4]], "diag_px": []}, None, None)
        self.assertEqual(site["other_crosswalks_px"], {"near": [[3, 3], [4, 3], [4, 4]]})

if __name__ == "__main__":
    unittest.main()
