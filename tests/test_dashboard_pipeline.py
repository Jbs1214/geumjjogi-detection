# 담당: 장범수
"""dashboard/pipeline.py·report_pdf.py·mailer.py·workflow.py 테스트. 실제 분석(GPU)·메일 발송·영역 지정 창 없이,
명령 실행은 출력 파일만 만드는 가짜로, SMTP는 가짜 객체로 바꾼다. 실행: python -m unittest discover -s tests -t . -v"""
import json
import os
import subprocess
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pandas as pd
import yaml

from dashboard import app, mailer, pipeline, report_pdf

START = datetime(2026, 11, 22, 8, 0)


def tiny_video(path: Path) -> Path:
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for _ in range(10):
        out.write(np.full((48, 64, 3), 120, np.uint8))
    out.release()
    return path


def judged_rows(video: str, site: str) -> pd.DataFrame:
    return pd.DataFrame({"video": video, "event_id": [1, 2, 3], "intersection": site,
                         "wall_time": ["2026-11-22 08:01:00", "2026-11-22 08:30:00", "2026-11-22 09:05:00"],
                         "slot": ["08–09시", "08–09시", "09–10시"], "vehicle_type": "car",
                         "ped_result": ["yield", "no_yield", "exclude"], "red_result": ["na", "stopped", "not_stopped"],
                         "status": "auto", "clip": ""})


class Sandbox(unittest.TestCase):
    """data/dashboard·outputs·configs 대신 임시 폴더를 쓴다."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        for name, value in {"WORK": self.tmp / "work", "VIDEOS_FILE": self.tmp / "work" / "videos.json",
                            "JOBS": self.tmp / "work" / "jobs", "OUT": self.tmp / "outputs"}.items():
            p = patch.object(pipeline, name, value)
            p.start()
            self.addCleanup(p.stop)
        p = patch.object(pipeline, "site_path", lambda vid: self.tmp / "configs" / f"site_{vid}.yaml")
        p.start()
        self.addCleanup(p.stop)
        (self.tmp / "configs").mkdir()
        self.video = tiny_video(self.tmp / "내 영상.mp4")

    def write_site(self, vid: str, signal: bool) -> None:
        site = {"site": vid.split("_")[0], "signal_roi_px": [[1, 1], [2, 1], [2, 2], [1, 2]] if signal else []}
        pipeline.site_path(vid).write_text(yaml.safe_dump(site, allow_unicode=True), encoding="utf-8")


class VideoListTest(Sandbox):
    def test_add_and_validate(self):
        self.assertIn("밑줄", pipeline.add_video("강남_역", "남→동", START, str(self.video)))
        self.assertIn("파일이 없습니다", pipeline.add_video("강남역", "남→동", START, str(self.tmp / "없음.mp4")))
        not_video = self.tmp / "메모.txt"
        not_video.write_text("x", encoding="utf-8")
        self.assertIn("영상으로 읽을 수 없습니다", pipeline.add_video("강남역", "남→동", START, str(not_video)))
        self.assertEqual(pipeline.add_video("강남역", "남→동", START, f'"{self.video}"'), "")   # 탐색기 "경로로 복사"의 따옴표
        self.assertIn("이미 있습니다", pipeline.add_video("강남역", "남→동", START, str(self.video)))
        v = pipeline.load_videos()[0]
        self.assertEqual((v["id"], v["direction"], v["start"]), ("강남역-S_20261122_0800", "남→동", "2026-11-22 08:00"))
        self.assertEqual(pipeline.video_status(v), "영역 미설정")

    def test_video_id_matches_rule_time_parsing(self):
        vid = pipeline.video_id("A", "서→남", START)   # rules.rule은 첫 _ 뒤를 %Y%m%d_%H%M로 읽는다
        self.assertEqual(datetime.strptime(vid.split("_", 1)[1], "%Y%m%d_%H%M"), START)
        self.assertEqual(vid.split("_")[0], "A-W")

    def test_remove_keeps_files(self):
        pipeline.add_video("강남역", "남→동", START, str(self.video))
        pipeline.remove_video("강남역-S_20261122_0800")
        self.assertEqual(pipeline.load_videos(), [])
        self.assertTrue(self.video.exists())

    def test_weather_and_signal_condition_saved(self):
        pipeline.add_video("강남역", "남→동", START, str(self.video), "우천", True)
        v = pipeline.load_videos()[0]
        self.assertEqual((v["weather"], v["right_turn_signal"]), ("우천", True))

    def test_day_night_by_start_time(self):
        """configs/eval.yaml의 day_hours 07:00~17:30(끝 제외)이 낮이다(기술서 11.1)."""
        for hm, expected in (("06:59", "야간"), ("07:00", "낮"), ("17:29", "낮"), ("17:30", "야간")):
            start = datetime.strptime(f"2026-11-22 {hm}", "%Y-%m-%d %H:%M")
            self.assertEqual(pipeline.day_night(start), expected, hm)

    def test_same_file_registered_twice_is_duplicate(self):
        pipeline.add_video("강남역", "남→동", START, str(self.video))
        self.assertEqual(pipeline.duplicates(pipeline.load_videos()), set())
        pipeline.add_video("강남역", "동→북", START, str(self.video))   # 방향만 다르게 같은 파일을 또 넣음
        self.assertEqual(pipeline.duplicates(pipeline.load_videos()), {"강남역-S_20261122_0800", "강남역-E_20261122_0800"})

    def test_status_counts_include_last_job_errors(self):
        pipeline.add_video("강남역", "남→동", START, str(self.video))
        videos = pipeline.load_videos()
        self.assertEqual(pipeline.status_counts(videos), {"분석 완료": 0, "영역 미설정": 1, "오류": 0})
        job = pipeline.JOBS / "20261122_090000"
        job.mkdir(parents=True)
        pipeline.write_json(job / "status.json", {"state": "실패", "videos": {videos[0]["id"]: "실패"},
                                                 "errors": {videos[0]["id"]: "track 단계 실패"}, "created": "x"})
        self.assertEqual(pipeline.status_counts(videos)["오류"], 1)


class SiteTest(Sandbox):
    def test_patch_site_puts_intersection_and_direction(self):
        pipeline.add_video("강남역", "동→북", START, str(self.video))
        v = pipeline.load_videos()[0]
        self.write_site(v["id"], signal=False)
        pipeline.patch_site(v)
        cfg = yaml.safe_load(pipeline.site_path(v["id"]).read_text(encoding="utf-8"))
        self.assertEqual((cfg["site"], cfg["approach_from"]), ("강남역", "동→북"))
        self.assertEqual(pipeline.video_status(v), "영역 완료")

    def test_patch_site_puts_signal_condition(self):
        """우회전 전용 신호가 있다고 등록하면 site의 right_turn_signal이 켜져 적색 판정에서 빠진다(rules.rule)."""
        pipeline.add_video("강남역", "동→북", START, str(self.video), right_turn_signal=True)
        v = pipeline.load_videos()[0]
        self.write_site(v["id"], signal=False)
        pipeline.patch_site(v)
        self.assertTrue(yaml.safe_load(pipeline.site_path(v["id"]).read_text(encoding="utf-8"))["right_turn_signal"])

    def test_roi_tool_esc_means_not_saved(self):
        pipeline.add_video("강남역", "남→동", START, str(self.video))
        v = pipeline.load_videos()[0]
        with patch("subprocess.run"):   # 창을 띄우지 않고 저장도 안 한 경우
            self.assertFalse(pipeline.run_roi_tool(v))

    def test_step_commands_with_and_without_signal(self):
        v = {"id": "A-S_20261122_0800", "source": str(self.video)}
        self.write_site(v["id"], signal=False)
        self.assertEqual([n for n, _ in pipeline.step_commands(v)], ["track", "events", "features", "rules"])
        self.write_site(v["id"], signal=True)
        steps = dict(pipeline.step_commands(v))
        self.assertIn("signal", steps)
        self.assertIn("--signal", steps["features"])


def fake_run(out: Path, video_stem: str, fail: str = ""):
    """파이프라인 명령 대신 출력 CSV만 만든다. fail 단계는 실패(returncode 1)."""
    def run(cmd, **kwargs):
        module = cmd[2]
        step = {"tracking.track": "track", "tracking.events": "events", "traffic_light.color": "signal",
                "features.build": "features", "rules.rule": "rules"}[module]
        if step == fail:
            kwargs["stdout"].write("Traceback: 가짜 오류\n")
            return subprocess.CompletedProcess(cmd, 1)
        arg = lambda name: cmd[cmd.index(name) + 1]  # noqa: E731
        if step in ("track", "signal"):   # 원래 영상 이름으로 쓴다
            folder = "tracks" if step == "track" else "signal"
            (out / folder).mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"video": [video_stem], "t_start_s": [0.0]}).to_csv(out / folder / f"{video_stem}.csv", index=False)
        elif step == "events":
            vid = Path(arg("--tracks")).stem
            assert (out / "tracks" / f"{vid}.csv").exists(), "track 결과가 video id 이름으로 바뀌어 있어야 한다"
            (out / "events").mkdir(exist_ok=True)
            (out / "events" / f"{vid}.csv").write_text("x", encoding="utf-8")
        elif step == "features":
            vid = Path(arg("--events")).stem
            (out / "features").mkdir(exist_ok=True)
            (out / "features" / f"{vid}.csv").write_text("x", encoding="utf-8")
        else:
            vid = Path(arg("--features")).stem
            (out / "judged").mkdir(exist_ok=True)
            judged_rows(vid, vid.split("-")[0]).to_csv(out / "judged" / f"{vid}.csv", index=False, encoding="utf-8-sig")
        return subprocess.CompletedProcess(cmd, 0)
    return run


class JobTest(Sandbox):
    def make_job(self, email: str = "", signal: bool = True) -> tuple[Path, dict]:
        pipeline.add_video("강남역", "남→동", START, str(self.video))
        v = pipeline.load_videos()[0]
        self.write_site(v["id"], signal)
        job = pipeline.JOBS / "20261122_090000"
        job.mkdir(parents=True)
        pipeline.write_json(job / "job.json", {"videos": [v], "email": email, "red_mode": "보통"})
        pipeline.write_json(job / "status.json", {"state": "대기", "videos": {v["id"]: "대기"}, "errors": {}, "created": "x"})
        return job, v

    def test_full_job_renames_outputs_and_makes_report(self):
        job, v = self.make_job()
        with patch("subprocess.run", fake_run(pipeline.OUT, self.video.stem)), patch("dashboard.db.configured", return_value=False):
            pipeline.run_job(job)
        status = pipeline.read_status(job)
        self.assertEqual((status["state"], status["videos"][v["id"]]), ("완료", "완료"))
        self.assertTrue((pipeline.OUT / "tracks" / f"{v['id']}.csv").exists())
        self.assertFalse((pipeline.OUT / "tracks" / f"{self.video.stem}.csv").exists())
        signal = pd.read_csv(pipeline.OUT / "signal" / f"{v['id']}.csv")
        self.assertEqual(signal["video"].tolist(), [v["id"]])   # 신호 시간표의 video 칸도 바꾼다
        self.assertTrue(Path(status["report"]).read_bytes().startswith(b"%PDF"))
        self.assertIn("메일 주소 없음", status["mail"])
        self.assertEqual(pipeline.video_status(v), "분석 완료")

    def test_failed_step_recorded_and_mail_sent_only_with_report(self):
        job, v = self.make_job(email="a@b.co", signal=False)
        with patch("subprocess.run", fake_run(pipeline.OUT, self.video.stem, fail="events")), \
                patch("dashboard.db.configured", return_value=False):
            pipeline.run_job(job)
        status = pipeline.read_status(job)
        self.assertEqual(status["state"], "실패")
        self.assertIn("events 단계 실패", status["errors"][v["id"]])
        self.assertIn("가짜 오류", status["errors"][v["id"]])
        self.assertIn("분석에 성공한 영상이 없습니다", status["mail"])

    def test_mail_sent_after_report(self):
        job, _ = self.make_job(email="a@b.co")
        with patch("subprocess.run", fake_run(pipeline.OUT, self.video.stem)), patch("dashboard.db.configured", return_value=False), \
                patch.object(mailer, "configured", return_value=True), patch.object(mailer, "send_report") as send:
            pipeline.run_job(job)
        send.assert_called_once()
        self.assertEqual(send.call_args.args[0], "a@b.co")
        self.assertEqual(pipeline.read_status(job)["mail"], "보냄: a@b.co")

    def test_progress_from_track_log(self):
        job, v = self.make_job()
        status = pipeline.update_status(job, state="진행 중", videos={v["id"]: "track", "B-N_20261122_0800": "대기"})
        self.assertIsNone(pipeline.track_progress(job, v["id"]))   # 아직 진행 줄 없음
        log = ["=== B-N_20261122_0800 track: ...", "  추적 900/1000프레임 (90%)",   # 다른 영상 줄은 무시
               f"=== {v['id']} track: ...", "  추적 0/1645프레임 (0%)", "  추적 822/1645프레임 (50%)"]
        (job / "log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
        self.assertAlmostEqual(pipeline.track_progress(job, v["id"]), 822 / 1645)
        overall, per = pipeline.job_progress(job, status)
        self.assertAlmostEqual(per[v["id"]], 822 / 1645)
        self.assertEqual(per["B-N_20261122_0800"], 0.0)
        self.assertAlmostEqual(overall, 822 / 1645 / 2)

    def test_stale_running_job_marked_stopped(self):
        job, _ = self.make_job()
        pipeline.update_status(job, state="진행 중", pid=0)
        self.assertIsNone(pipeline.running_job())
        self.assertEqual(pipeline.read_status(job)["state"], "중단됨")


class ReportTest(unittest.TestCase):
    def test_pdf_has_pages(self):
        df = pd.concat([judged_rows("A-S_20261122_0800", "A"), judged_rows("B-N_20261122_0800", "B")])
        out = report_pdf.build_report(df, Path(tempfile.mkdtemp()) / "r.pdf", 30)
        data = out.read_bytes()
        self.assertTrue(data.startswith(b"%PDF"))
        self.assertEqual(data.count(b"/Type /Page\n") + data.count(b"/Type /Page "), 3)   # 요약 + 보행자 + 적색

    def test_no_events_still_makes_report(self):
        empty = judged_rows("A-S_20261122_0800", "A").iloc[:0]   # 우회전 이벤트 0건인 영상
        out = report_pdf.build_report(empty, Path(tempfile.mkdtemp()) / "r.pdf", 30, videos={"A-S_20261122_0800": "A"})
        self.assertTrue(out.read_bytes().startswith(b"%PDF"))

    def test_details_follow_available_columns(self):
        """제외 사유·횡단보도 칸이 있으면 표로 나오고, 없는 표(DB에서 읽은 것)로도 리포트가 만들어진다."""
        df = judged_rows("A-S_20261122_0800", "A").assign(
            ped_exclude_reason=[None, None, "camera_moved"], no_yield_at=[None, "near;exit", None])
        self.assertEqual(report_pdf.reason_rows(df, "ped_result"), [["카메라가 움직인 구간", "1건"]])
        self.assertEqual(sorted(report_pdf.crosswalk_rows(df)), [["진입 쪽 횡단보도", "1건"], ["출구 쪽 횡단보도", "1건"]])
        plain = judged_rows("A-S_20261122_0800", "A")   # DB의 events 표와 같은 칸
        self.assertEqual(report_pdf.detail_blocks(plain, "ped_result"), [])
        out = report_pdf.build_report(plain, Path(tempfile.mkdtemp()) / "r.pdf", 30, red_mode="엄격")
        self.assertTrue(out.read_bytes().startswith(b"%PDF"))

    def test_kinds_match_dashboard(self):
        self.assertEqual({k: v[:3] for k, v in app.KINDS.items()}, report_pdf.KINDS)
        self.assertEqual(app.NOTICE, report_pdf.NOTICE)


class MailerTest(unittest.TestCase):
    def setUp(self):
        self.pdf = Path(tempfile.mkdtemp()) / "report.pdf"
        self.pdf.write_bytes(b"%PDF-1.4 test")

    def test_message_has_pdf_and_notice(self):
        with patch.dict(os.environ, {"SMTP_USER": "sender@example.com"}):
            msg = mailer.build_message("to@example.com", self.pdf)
        self.assertEqual((msg["To"], msg["From"]), ("to@example.com", "sender@example.com"))
        files = [part.get_filename() for part in msg.iter_attachments()]
        self.assertEqual(files, ["report.pdf"])
        self.assertIn(report_pdf.NOTICE, msg.get_body().get_content())

    def test_starttls_or_ssl_by_port(self):
        env = {"SMTP_HOST": "smtp.example.com", "SMTP_USER": "u", "SMTP_PASSWORD": "p"}
        for port, cls in (("587", "SMTP"), ("465", "SMTP_SSL")):
            with patch.dict(os.environ, {**env, "SMTP_PORT": port}), patch.object(mailer, "load_env"), \
                    patch(f"smtplib.{cls}") as smtp:
                mailer.send_report("to@example.com", self.pdf)
            server = smtp.return_value
            server.login.assert_called_once_with("u", "p")
            server.send_message.assert_called_once()
            self.assertEqual(server.starttls.called, port == "587")


class PageTest(Sandbox):
    def test_analysis_page_renders(self):
        from streamlit.testing.v1 import AppTest
        pipeline.add_video("강남역", "남→동", START, str(self.video))
        with patch("dashboard.db.configured", return_value=False):
            at = AppTest.from_file(str(Path(app.__file__)), default_timeout=30).run()
            at.sidebar.radio[0].set_value("영상 분석").run()
        self.assertFalse(at.exception)
        self.assertEqual([s.value for s in at.main.subheader], ["① 영상 등록", "② 영역 설정", "③ 분석", "④ 리포트"])
        self.assertEqual([s.value for s in at.sidebar.subheader], ["공통 필터", "이 화면: 처리 현황"])
        self.assertIn("영역 설정 시작 (1개)", [b.label for b in at.sidebar.button])
        self.assertEqual([t.value for t in at.sidebar.text], ["분석 완료 0", "영역 미설정 1", "오류 0"])

    def test_register_marks_duplicate_and_filter_narrows_list(self):
        from streamlit.testing.v1 import AppTest
        pipeline.add_video("강남역", "남→동", START, str(self.video), "우천")
        with patch("dashboard.db.configured", return_value=False):
            at = AppTest.from_file(str(Path(app.__file__)), default_timeout=30).run()
            at.sidebar.radio[0].set_value("영상 분석").run()
            # 같은 파일을 방향만 바꿔 화면에서 한 번 더 넣는다(날씨는 기본값 맑음) → 두 줄 모두 "중복"
            next(s for s in at.main.selectbox if s.label.startswith("우회전 방향")).select("동→북").run()
            next(t for t in at.main.text_input if t.label == "영상 파일 경로").set_value(str(self.video)).run()
            next(b for b in at.main.button if b.label == "목록에 추가").click().run()
            both = [m.value for m in at.main.markdown if "badge" in m.value]
            next(g for g in at.sidebar.button_group if g.label == "날씨").select("우천").run()
            rainy = [m.value for m in at.main.markdown if "badge" in m.value]
        self.assertFalse(at.exception)
        self.assertEqual(len(pipeline.load_videos()), 2)
        self.assertEqual(both, [":orange-badge[영역 미설정] :red-badge[중복]"] * 2)
        self.assertEqual(rainy, [":orange-badge[영역 미설정] :red-badge[중복]"])   # 우천으로 등록한 한 줄만 남는다


if __name__ == "__main__":
    unittest.main()
