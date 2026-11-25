# 담당: 육주영
"""dashboard/app.py 테스트. 실행: python -m unittest discover -s tests -t . -v"""
import io
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

import pandas as pd
import yaml
from streamlit.testing.v1 import AppTest

from dashboard.app import EXPORT_NOTE, NOTICE, SIGNUP_DONE, apply_filters, load_judged, pct, rate_views, top_slots
from dashboard.auth import hash_password


def make_judged() -> pd.DataFrame:
    """A 08–09시 40건(미양보 10), A 09–10시 40건(미양보 20), B 08–09시 3건(미양보 3, 참고용)."""
    rows = []
    for site, day, slot, hour, n, k in [
        ("A", "2026-10-31", "08–09시", 8, 40, 10),
        ("A", "2026-10-31", "09–10시", 9, 40, 20),
        ("B", "2026-11-07", "08–09시", 8, 3, 3),
    ]:
        for i in range(n):
            rows.append({
                "video": f"{site}_{day.replace('-', '')}_0800",
                "event_id": len(rows) + 1,
                "intersection": site,
                "wall_time": f"{day} {hour:02d}:{i:02d}:00",
                "slot": slot,
                "ped_result": "no_yield" if i < k else "yield",
                "clip": "",
            })
    return pd.DataFrame(rows)


def run_app(judged_dir, config):
    from dashboard.app import main

    main(judged_dir, config)


def run_app_with_login(config):
    from dashboard.app import main

    main(config=config, use_db=True)


class DataTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        make_judged().to_csv(Path(self.tmp.name) / "all.csv", index=False, encoding="utf-8-sig")
        self.df = load_judged(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_filters(self):
        self.assertEqual(len(apply_filters(self.df, ["A"], date(2026, 10, 1), date(2026, 11, 30))), 80)
        self.assertEqual(len(apply_filters(self.df, ["A", "B"], date(2026, 11, 1), date(2026, 11, 30))), 3)

    def test_top_slots_skip_reference_only(self):
        from eval.report import rate_table

        top = top_slots(rate_table(self.df, 30))
        # B는 위반율 100%지만 3건이라 참고용 → 빠진다
        self.assertEqual(list(zip(top["intersection"], top["slot"])), [("A", "09–10시"), ("A", "08–09시")])

    def test_top_slots_skip_zero_violation(self):
        # 위반 의심이 0건인 칸은 단속 우선 시간대가 아니다
        table = pd.DataFrame({"note": ["", ""], "violation_rate": [0.0, 0.2]})
        self.assertEqual(list(top_slots(table)["violation_rate"]), [0.2])

    def test_pct_keeps_small_rates_visible(self):
        self.assertEqual([pct(0), pct(1 / 264), pct(0.398)], ["0%", "1% 미만", "40%"])

    def test_export_keeps_precision(self):
        from eval.report import rate_table

        view, export = rate_views(rate_table(self.df, 30), "no_yield")
        # 화면용은 막대 칸 때문에 0~100, 내려받기용은 0~1이고 신뢰구간은 숫자 칸 둘(하한·상한)이다
        self.assertEqual(view["위반율"].max(), 100)
        self.assertEqual(export["위반율"].max(), 1)
        self.assertTrue(pd.api.types.is_float_dtype(export["95% 신뢰구간 하한"]))
        self.assertTrue(pd.api.types.is_float_dtype(export["95% 신뢰구간 상한"]))
        self.assertNotIn("95% 신뢰구간", export.columns)

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as empty:
            self.assertTrue(load_judged(empty).empty)


class AppSmokeTest(unittest.TestCase):
    def test_renders_with_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_judged().to_csv(Path(tmp) / "all.csv", index=False, encoding="utf-8-sig")
            config = Path(tmp) / "eval.yaml"
            config.write_text(yaml.safe_dump({"min_n_reference": 30}), encoding="utf-8")
            at = AppTest.from_function(run_app, args=(tmp, str(config)), default_timeout=30).run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.warning[0].value, NOTICE)
        self.assertEqual([m.value for m in at.metric], ["83건", "33건", "40%"])
        self.assertEqual(len(at.dataframe), 1)

    def test_download_has_context_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_judged().to_csv(Path(tmp) / "all.csv", index=False, encoding="utf-8-sig")
            config = Path(tmp) / "eval.yaml"
            config.write_text(yaml.safe_dump({"min_n_reference": 30}), encoding="utf-8")
            with mock.patch("streamlit.download_button") as button:  # 버튼에 넘어가는 CSV를 잡는다
                at = AppTest.from_function(run_app, args=(tmp, str(config)), default_timeout=30).run()
        self.assertEqual(len(at.exception), 0)
        csv = pd.read_csv(io.BytesIO(button.call_args.args[1]), encoding="utf-8-sig")
        self.assertEqual(set(csv["위반 유형"]), {"보행자 미양보"})
        self.assertEqual(set(csv["자료 기간"]), {"2026-10-31 ~ 2026-11-07"})
        self.assertEqual(set(csv["비고"]), {EXPORT_NOTE})
        self.assertIn("내려받은 시각", csv.columns)

    def run_with_red(self, kind):
        """make_judged에 적색 판정을 붙인다: 83건 중 미정지 20, 일시정지 30, 나머지 na."""
        judged = make_judged().assign(red_result=["not_stopped"] * 20 + ["stopped"] * 30 + ["na"] * 33)
        with tempfile.TemporaryDirectory() as tmp:
            judged.to_csv(Path(tmp) / "all.csv", index=False, encoding="utf-8-sig")
            config = Path(tmp) / "eval.yaml"
            config.write_text(yaml.safe_dump({"min_n_reference": 30}), encoding="utf-8")
            at = AppTest.from_function(run_app, args=(tmp, str(config)), default_timeout=30).run()
            at.selectbox[0].select(kind).run()
        self.assertEqual(len(at.exception), 0)
        return at

    def test_red_kind_shows_red_rate(self):
        at = self.run_with_red("적색 미정지")
        # na는 분모에서 빠진다: 판정 대상 50, 미정지 20
        self.assertEqual([m.value for m in at.metric], ["50건", "20건", "40%"])
        self.assertEqual(len(at.dataframe), 1)

    def test_all_kinds_side_by_side_not_merged(self):
        at = self.run_with_red("전체")
        self.assertEqual([m.value for m in at.metric], ["83건", "33건", "40%", "50건", "20건", "40%"])
        self.assertEqual(len(at.dataframe), 2)

    def test_strict_mode_counts_minor_stop_as_violation(self):
        """경미 정지 10건: 판정 대상은 두 모드 모두 60, 위반 의심은 보통 20 → 엄격 30 (기술서 3.1)."""
        judged = make_judged().assign(
            red_result=["not_stopped"] * 20 + ["stopped"] * 30 + ["stopped_minor"] * 10 + ["na"] * 23
        )
        with tempfile.TemporaryDirectory() as tmp:
            judged.to_csv(Path(tmp) / "all.csv", index=False, encoding="utf-8-sig")
            config = Path(tmp) / "eval.yaml"
            config.write_text(yaml.safe_dump({"min_n_reference": 30}), encoding="utf-8")
            at = AppTest.from_function(run_app, args=(tmp, str(config)), default_timeout=30).run()
            at.selectbox[0].select("적색 미정지").run()
            normal = [m.value for m in at.metric]
            at.sidebar.button_group[0].select("엄격").run()   # 사이드바의 분석 모드
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(normal, ["60건", "20건", "33%"])
        self.assertEqual([m.value for m in at.metric], ["60건", "30건", "50%"])

    def test_overall_priority_matches_report(self):
        """종합 단속 우선 검토 시간대 = 보고서 PDF와 같은 계산(report_pdf.overall_slots). 태그는 글자로만 나간다."""
        rows = [("<b>A</b>", "08–09시", 20, 5), ("B", "18–19시", 10, 30), ("C", "12–13시", 15, 20)]  # 40건 중 미양보, 미정지
        judged = pd.concat([pd.DataFrame({
            "video": "v", "event_id": range(40), "intersection": site, "wall_time": "2026-11-16 08:00:00", "slot": slot, "clip": "",
            "ped_result": ["no_yield"] * ped + ["yield"] * (40 - ped),
            "red_result": ["not_stopped"] * red + ["stopped"] * (40 - red)}) for site, slot, ped, red in rows])
        with tempfile.TemporaryDirectory() as tmp:
            judged.to_csv(Path(tmp) / "all.csv", index=False, encoding="utf-8-sig")
            config = Path(tmp) / "eval.yaml"
            config.write_text(yaml.safe_dump({"min_n_reference": 30}), encoding="utf-8")
            at = AppTest.from_function(run_app, args=(tmp, str(config)), default_timeout=30).run()
        self.assertEqual(len(at.exception), 0)
        body = at.get("html")[0].proto.body
        order = [body.index(text) for text in ("&lt;b&gt;A&lt;/b&gt;", "<td>C</td>", "<td>B</td>")]
        self.assertEqual(order, sorted(order))  # 순위 합은 모두 4 → 미양보 위반율이 높은 순 A, C, B
        self.assertNotIn("<b>A</b>", body)

    def test_both_kinds_by_default_and_vehicle_tables(self):
        """처음부터 두 유형이 나란히 나오고, 교차로·차종 표가 위반 유형 선택과 상관없이 나온다."""
        judged = make_judged().assign(red_result=["not_stopped"] * 20 + ["stopped"] * 30 + ["na"] * 33,
                                      vehicle_type=["트럭"] * 40 + ["차"] * 43)
        with tempfile.TemporaryDirectory() as tmp:
            judged.to_csv(Path(tmp) / "all.csv", index=False, encoding="utf-8-sig")
            config = Path(tmp) / "eval.yaml"
            config.write_text(yaml.safe_dump({"min_n_reference": 30}), encoding="utf-8")
            at = AppTest.from_function(run_app, args=(tmp, str(config)), default_timeout=30).run()
            first = [m.value for m in at.metric]
            at.selectbox[0].select("적색 미정지").run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(first, ["83건", "33건", "40%", "50건", "20건", "40%"])
        titles = [s.value for s in at.subheader]
        self.assertIn("교차로·차종별 현황", titles)
        self.assertNotIn("보행자 미양보 총괄 현황", titles)
        table = at.get("html")[-1].proto.body
        self.assertIn('<td rowspan="2">A</td>', table)
        self.assertIn("<td>승용·승합차</td>", table)
        self.assertIn("<td>100%*</td>", table)  # B 승용·승합차 3건 중 3건은 참고용

    def test_clip_chosen_one_at_a_time(self):
        """우선 시간대의 클립은 하나씩 골라 본다(수십 개를 한꺼번에 불러오지 않게)."""
        judged = make_judged()
        judged["clip"] = [f"outputs/clips/v_{i}.webm" if r == "no_yield" else "" for i, r in enumerate(judged["ped_result"])]
        with tempfile.TemporaryDirectory() as tmp:
            judged.to_csv(Path(tmp) / "all.csv", index=False, encoding="utf-8-sig")
            config = Path(tmp) / "eval.yaml"
            config.write_text(yaml.safe_dump({"min_n_reference": 30}), encoding="utf-8")
            at = AppTest.from_function(run_app, args=(tmp, str(config)), default_timeout=30).run()
        self.assertEqual(len(at.exception), 0)
        first = at.selectbox(key="clip-ped_result-1")  # 1순위 A 09–10시, 미양보 20건
        self.assertEqual(len(first.options), 20)
        self.assertEqual(first.options[0], "10-31 09:00:00")
        self.assertEqual(len([c for c in at.caption if c.value.startswith("파일 없음")]), 3)  # 시간대마다 하나 + 전체에서 하나
        self.assertIn("위반 의심 영상 자료 (33개, 의심 33건 중)", [s.value for s in at.subheader])

    def test_red_kind_without_red_judgment(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_judged().to_csv(Path(tmp) / "all.csv", index=False, encoding="utf-8-sig")
            config = Path(tmp) / "eval.yaml"
            config.write_text(yaml.safe_dump({"min_n_reference": 30}), encoding="utf-8")
            at = AppTest.from_function(run_app, args=(tmp, str(config)), default_timeout=30).run()
            at.selectbox[0].select("적색 미정지").run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.metric), 0)
        self.assertEqual(at.info[-1].value, "적색 미정지 판정 결과가 없습니다.")

    def test_renders_without_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            at = AppTest.from_function(run_app, args=(tmp, "unused.yaml"), default_timeout=30).run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.warning[0].value, NOTICE)
        self.assertEqual(len(at.info), 1)


class LoginTest(unittest.TestCase):
    """MySQL 모드(use_db). dashboard.db 함수는 mock으로 바꿔 MySQL 없이 돈다."""

    def setUp(self):
        tmp = self.enterContext(tempfile.TemporaryDirectory())
        make_judged().to_csv(Path(tmp) / "all.csv", index=False, encoding="utf-8-sig")
        config = Path(tmp) / "eval.yaml"
        config.write_text(yaml.safe_dump({"min_n_reference": 30}), encoding="utf-8")
        names = ["get_user", "add_user", "pending_users", "approve_user", "fetch_events"]
        self.db = self.enterContext(mock.patch.multiple("dashboard.db", **dict.fromkeys(names, mock.DEFAULT)))
        self.db["fetch_events"].return_value = load_judged(tmp)
        self.db["pending_users"].return_value = []
        self.db["get_user"].return_value = None  # 가입된 이메일이 없다
        self.at =AppTest.from_function(run_app_with_login, args=(str(config),), default_timeout=30).run()

    def login(self, password="password-1", **changes):
        self.db["get_user"].return_value = {
            "id": 1, "email": "officer@example.go.kr", "password_hash": hash_password("password-1"),
            "name": "홍길동", "agency": "예시경찰서", "role": "viewer", "status": "approved", **changes,
        }
        self.at.text_input(key="login-email").set_value("officer@example.go.kr")
        self.at.text_input(key="login-pw").set_value(password)
        self.at.button(key="login-submit").click().run()
        self.assertEqual(len(self.at.exception), 0)

    def fill_signup(self, **changes):
        """가입 칸을 채우고(버튼은 누르지 않음) 화면의 작은 글자들을 돌려준다."""
        values = {"email": "new@example.go.kr", "password": "password-1", "confirm": "password-1",
                  "name": "김신청", "agency": "예시구청", **changes}
        for key, value in values.items():
            self.at.text_input(key=f"signup-{key}").set_value(value)
        self.at.run()
        self.assertEqual(len(self.at.exception), 0)
        return [c.value for c in self.at.caption]

    def test_dashboard_hidden_before_login(self):
        self.assertEqual(len(self.at.exception), 0)
        self.assertEqual(self.at.warning[0].value, NOTICE)  # 안내 문구는 로그인 전에도 보인다
        self.assertEqual(len(self.at.metric) + len(self.at.dataframe), 0)
        self.db["fetch_events"].assert_not_called()

    def test_approved_user_sees_dashboard(self):
        self.login()
        self.assertEqual([m.value for m in self.at.metric], ["83건", "33건", "40%"])
        chip = self.at.get("html")[0].proto.body  # 오른쪽 위 프로필 칩
        self.assertIn('<div class="account-name">홍길동</div><div class="account-agency">예시경찰서</div>', chip)
        self.assertNotIn("관리자", chip)
        self.assertEqual(self.at.button(key="logout").label, "로그아웃")
        self.assertFalse(any(b.key == "view-toggle" for b in self.at.button))  # 가입 승인 관리는 관리자 전용
        self.db["pending_users"].assert_not_called()

    def test_wrong_password_or_pending_user_blocked(self):
        self.login(password="password-2")
        self.assertEqual(self.at.error[0].value, "이메일 또는 비밀번호가 맞지 않습니다.")
        self.login(status="pending")
        self.assertIn("관리자 승인 대기 중입니다.", [w.value for w in self.at.warning])
        self.assertEqual(len(self.at.metric), 0)

    def test_admin_approves_signup(self):
        self.db["pending_users"].return_value = [{
            "id": 7, "email": "new@example.go.kr", "name": "김신청", "agency": "예시구청", "department": "교통과",
            "job_position": "주무관", "phone": "01011112222",
        }]
        self.login(role="admin", name="<b>관리</b>")
        chip = self.at.get("html")[0].proto.body
        self.assertIn('&lt;b&gt;관리&lt;/b&gt;<span class="account-role">관리자</span>', chip)  # 이름은 태그로 나가지 않는다
        self.assertEqual(len(self.at.metric), 3)  # 관리자도 로그인하면 대시보드부터 본다
        self.assertEqual(self.at.button(key="view-toggle").label, "가입 승인 관리 (1)")
        self.at.button(key="view-toggle").click().run()
        # 별도 페이지: 대시보드 대신 신청자 정보 표. 휴대폰 번호는 - 를 넣어 보여 준다
        self.assertEqual(len(self.at.metric), 0)
        self.assertEqual([t.value for t in self.at.text],
                         ["김신청", "new@example.go.kr", "예시구청", "교통과", "주무관", "010-1111-2222"])
        self.at.button(key="approve-7").click().run()
        self.db["approve_user"].assert_called_once_with(7)
        self.at.button(key="view-toggle").click().run()
        self.assertEqual(len(self.at.metric), 3)  # 대시보드로 돌아온다

    def test_signup_checks_while_typing(self):
        # 버튼을 누르기 전에 틀린 칸 아래에 이유가 뜬다
        captions = self.fill_signup(email="abc", password="short", confirm="other", phone="010-1111-2222")
        for message in ["이메일 형식이 아닙니다.", "비밀번호는 8자 이상이어야 합니다.", "비밀번호 확인이 다릅니다.",
                        "휴대폰 번호는 - 없이 숫자만 적어 주세요(예: 01011112222)."]:
            self.assertIn(f":red[{message}]", captions)
        self.assertFalse(any("red" in c for c in self.fill_signup(email="new@example.go.kr", password="password-1",
                                                                 confirm="password-1", phone="01011112222")))

    def test_empty_required_field_flagged_only_after_submit(self):
        self.assertNotIn(":red[필수 입력입니다.]", self.fill_signup(name=""))  # 아직 손대지 않은 칸
        self.at.button(key="signup-submit").click().run()
        self.assertIn(":red[필수 입력입니다.]", [c.value for c in self.at.caption])
        self.db["add_user"].assert_not_called()

    def test_signup_rejects_registered_email(self):
        self.db["get_user"].return_value = {"id": 1}
        self.assertIn(":red[이미 가입된 이메일입니다.]", self.fill_signup())
        self.db["add_user"].return_value = False  # DB의 중복 불가 조건이 마지막으로 막는다
        self.at.button(key="signup-submit").click().run()
        self.assertEqual(len(self.at.success), 0)

    def test_signup_is_request_only_and_returns_to_login(self):
        self.db["add_user"].return_value = True
        self.fill_signup()
        self.at.button(key="signup-submit").click().run()
        user, password = self.db["add_user"].call_args.args
        self.assertEqual((user["email"], user["agency"], user["phone"], password),
                         ("new@example.go.kr", "예시구청", "", "password-1"))
        self.assertEqual(self.at.session_state["auth-tab"], "로그인")
        self.assertEqual([s.value for s in self.at.success], [SIGNUP_DONE])
        self.assertEqual(self.at.text_input(key="signup-email").value, "")  # 가입 칸은 비운다
        self.assertEqual(len(self.at.metric), 0)  # 신청만으로는 대시보드가 열리지 않는다


if __name__ == "__main__":
    unittest.main()
