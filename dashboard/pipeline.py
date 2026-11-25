# 담당: 장범수
"""대시보드의 영상 분석 흐름 중 화면이 없는 부분: 영상 목록 → 영역 설정(tracking.roi_tool) → 분석 작업 → 리포트·메일.

영상은 사용자가 적은 PC 경로에 그대로 둔다(복사하지 않음). 파이프라인은 파일 이름 {교차로}_{YYYYMMDD}_{HHMM}에서
촬영 시각을 읽으므로(rules.rule), 영상마다 이름 "{교차로}-{방향 코드}_{YYYYMMDD}_{HHMM}"(video id)을 정하고
track·color가 원래 영상 이름으로 낸 CSV를 이 이름으로 바꿔 둔다. 그 뒤 단계(events·features·rules)는 CSV 이름을 따라간다.
site 파일은 configs/site_<video id>.yaml(영상마다 1개)이고, 영역 설정 뒤 site = 교차로 이름, approach_from = 우회전 방향,
right_turn_signal = 등록할 때 고른 신호 조건을 넣는다. 날씨는 목록에만 적고, 낮·야간은 촬영 시작 시각으로 계산한다(기술서 11.1).

분석 작업은 대시보드와 따로 도는 프로세스(python -m dashboard.pipeline run <작업 폴더>)라 화면을 새로 고쳐도 계속된다.
한 번에 하나만 돈다(GPU). 작업 폴더(data/dashboard/jobs/<시각>)에 job.json(영상·메일), status.json(진행), log.txt, report.pdf.
영상마다 track → events → color(신호등 영역이 있을 때) → features → rules → (DB 설정 시) events 표에 넣기,
다 끝나면 이번 영상들의 판정으로 리포트 PDF를 만들고 메일 주소가 있으면 보낸다(통계만, 영상·클립은 보내지 않음).
그다음 위반 의심 클립(블러, eval.clips)을 만들어 판정 CSV의 clip 칸(과 DB)을 채운다. 오래 걸리므로 리포트·메일 뒤에 하고, 실패해도 판정과 리포트는 그대로다.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import yaml

from eval.report import select_clip_events

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data" / "dashboard"
VIDEOS_FILE = WORK / "videos.json"
JOBS = WORK / "jobs"
OUT = ROOT / "outputs"
EVAL_CONFIG = ROOT / "configs" / "eval.yaml"
# 우회전 방향(들어오는 쪽 → 꺾어 나가는 쪽): video id에 넣는 코드
DIRECTIONS = {"남→동": "S", "동→북": "E", "북→서": "N", "서→남": "W"}
WEATHERS = ["맑음", "우천"]
# 신호 조건(표시 이름: site 파일의 right_turn_signal). 우회전 전용 신호가 있으면 적색 정지 판정에서 뺀다(기술서 2.3)
SIGNALS = {"직진 신호만": False, "우회전 전용 신호 있음 (정지선 제외)": True}
NAME_PATTERN = r"[^\s_/\\:*?\"<>|.]+"  # 교차로 이름: 파일 이름에 들어가므로 공백·밑줄·경로 문자 없이
STEPS = ["track", "events", "signal", "features", "rules", "clips"]  # 화면에 보여 줄 단계 이름(signal은 신호등 영역이 있을 때만)
TRACK_LINE = r"추적 (\d+)/(\d+)프레임"  # tracking.track이 1%마다 찍는 진행 줄
UNBUFFERED = {"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}  # 단계 출력이 바로 log.txt에 쓰이게(진행률·오류를 바로 보이게)


# ---------- 영상 목록 ----------

def load_videos() -> list[dict]:
    return json.loads(VIDEOS_FILE.read_text(encoding="utf-8")) if VIDEOS_FILE.exists() else []


def save_videos(videos: list[dict]) -> None:
    write_json(VIDEOS_FILE, videos)


def write_json(path: Path, data) -> None:
    """다른 프로세스가 반쯤 쓴 파일을 읽지 않게 임시 파일에 쓰고 바꿔 넣는다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def video_id(intersection: str, direction: str, start: datetime) -> str:
    return f"{intersection}-{DIRECTIONS[direction]}_{start:%Y%m%d_%H%M}"


def site_path(vid: str) -> Path:
    return ROOT / "configs" / f"site_{vid}.yaml"


def add_video(intersection: str, direction: str, start: datetime, source: str,
              weather: str = WEATHERS[0], right_turn_signal: bool = False) -> str:
    """영상을 목록에 넣는다. 문제가 있으면 이유를 돌려주고 넣지 않는다(문제없으면 "")."""
    intersection, source = intersection.strip(), source.strip().strip('"')
    if not re.fullmatch(NAME_PATTERN, intersection):
        return "교차로 이름에 공백·밑줄(_)·마침표·경로 문자(/ \\ : 등)를 쓸 수 없습니다."
    if direction not in DIRECTIONS:
        return "우회전 방향을 고르세요."
    if not Path(source).is_file():
        return f"영상 파일이 없습니다: {source}"
    cap = cv2.VideoCapture(source)
    ok = cap.isOpened() and cap.read()[0]
    cap.release()
    if not ok:
        return f"영상으로 읽을 수 없습니다: {source}"
    vid = video_id(intersection, direction, start)
    videos = load_videos()
    if any(v["id"] == vid for v in videos):
        return f"같은 교차로·방향·촬영 시각의 영상이 이미 있습니다: {vid}"
    videos.append({"id": vid, "intersection": intersection, "direction": direction,
                   "start": start.strftime("%Y-%m-%d %H:%M"), "source": str(Path(source).resolve()),
                   "weather": weather, "right_turn_signal": right_turn_signal})
    save_videos(videos)
    return ""


def remove_video(vid: str) -> None:
    """목록에서만 뺀다(영상·site 파일·결과는 그대로)."""
    save_videos([v for v in load_videos() if v["id"] != vid])


def video_status(video: dict) -> str:
    if (OUT / "judged" / f"{video['id']}.csv").exists():
        return "분석 완료"
    return "영역 완료" if site_path(video["id"]).exists() else "영역 미설정"


def day_night(start: datetime) -> str:
    """촬영 시작 시각이 configs/eval.yaml의 day_hours 사이(끝 제외)면 "낮", 그 밖은 "야간"(기술서 11.1)."""
    day_from, day_until = yaml.safe_load(EVAL_CONFIG.read_text(encoding="utf-8"))["day_hours"]
    return "낮" if day_from <= f"{start:%H:%M}" < day_until else "야간"


def duplicates(videos: list[dict]) -> set[str]:
    """같은 파일 이름이 두 번 이상 등록된 영상의 id(교차로·방향·시각을 다르게 적어 같은 영상을 또 넣은 경우)."""
    names = [Path(v["source"]).name for v in videos]
    return {v["id"] for v, name in zip(videos, names) if names.count(name) > 1}


def status_counts(videos: list[dict]) -> dict[str, int]:
    """처리 현황: 분석 완료·영역 미설정 영상 수와, 가장 최근 분석에서 실패한 영상 수."""
    states = [video_status(v) for v in videos]
    jobs = list_jobs()
    failed = set(jobs[0][1].get("errors", {})) if jobs else set()
    return {"분석 완료": states.count("분석 완료"), "영역 미설정": states.count("영역 미설정"),
            "오류": sum(v["id"] in failed for v in videos)}


# ---------- 영역 설정 ----------

def run_roi_tool(video: dict) -> bool:
    """영역 지정 창(tracking.roi_tool)을 이 PC에 띄우고 닫힐 때까지 기다린다. 저장했으면 교차로·방향을 넣고 True."""
    site = site_path(video["id"])
    before = site.stat().st_mtime if site.exists() else None
    subprocess.run([sys.executable, "-m", "tracking.roi_tool", "--video", video["source"], "--out", str(site)],
                   cwd=ROOT, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if not site.exists() or site.stat().st_mtime == before:  # Esc로 끝냄(저장 안 함)
        return False
    patch_site(video)
    return True


def patch_site(video: dict) -> None:
    """roi_tool은 site 값을 영상 파일 이름에서 가져오므로, 대시보드에 적은 교차로 이름과 우회전 방향으로 바꾼다.
    등록할 때 고른 신호 조건도 넣는다(예전에 등록해 그 값이 없는 영상은 roi_tool 값 그대로)."""
    site = site_path(video["id"])
    cfg = yaml.safe_load(site.read_text(encoding="utf-8"))
    cfg["site"], cfg["approach_from"] = video["intersection"], video["direction"]
    if "right_turn_signal" in video:
        cfg["right_turn_signal"] = video["right_turn_signal"]
    with open(site, "w", encoding="utf-8") as f:
        f.write(f"# tracking.roi_tool 로 생성, 대시보드에서 교차로·방향 반영: {Path(video['source']).name}\n")
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False, default_flow_style=None)


# ---------- 분석 작업 ----------

def start_job(videos: list[dict], email: str, red_mode: str = "보통") -> Path:
    """분석 작업 폴더를 만들고 대시보드와 따로 도는 프로세스로 시작한다. red_mode = 리포트의 적색 판정 모드."""
    job = JOBS / datetime.now().strftime("%Y%m%d_%H%M%S")
    job.mkdir(parents=True)
    write_json(job / "job.json", {"videos": videos, "email": email.strip(), "red_mode": red_mode})
    write_json(job / "status.json", {"state": "대기", "videos": {v["id"]: "대기" for v in videos}, "errors": {},
                                    "created": datetime.now().strftime("%Y-%m-%d %H:%M")})
    flags = {"creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" \
        else {"start_new_session": True}
    with open(job / "log.txt", "a", encoding="utf-8") as log:
        proc = subprocess.Popen([sys.executable, "-m", "dashboard.pipeline", "run", str(job)], cwd=ROOT, stdout=log,
                                stderr=subprocess.STDOUT, env={**os.environ, "PYTHONIOENCODING": "utf-8"}, **flags)
    update_status(job, pid=proc.pid)
    return job


def read_status(job: Path) -> dict:
    return json.loads((job / "status.json").read_text(encoding="utf-8"))


def update_status(job: Path, **changes) -> dict:
    status = {**read_status(job), **changes}
    write_json(job / "status.json", status)
    return status


def alive(pid: int | None) -> bool:
    if not pid:
        return False
    if os.name == "nt":  # os.kill(pid, 0)은 Windows에서 프로세스를 끝내 버리므로 tasklist로 본다
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def list_jobs() -> list[tuple[Path, dict]]:
    """작업 폴더와 진행 상태, 최근 것부터. 진행 중인데 프로세스가 없으면(PC 재시작 등) 중단으로 고친다."""
    jobs = []
    for job in sorted(JOBS.glob("*"), reverse=True) if JOBS.exists() else []:
        if not (job / "status.json").exists():
            continue
        status = read_status(job)
        if status["state"] in ("대기", "진행 중") and not alive(status.get("pid")):
            status = update_status(job, state="중단됨")
        jobs.append((job, status))
    return jobs


def running_job() -> Path | None:
    return next((job for job, status in list_jobs() if status["state"] in ("대기", "진행 중")), None)


def step_commands(video: dict) -> list[tuple[str, list[str]]]:
    """영상 하나의 (단계 이름, 명령) 목록. 신호등 영역이 없으면 signal 단계와 --signal을 뺀다(1단계만)."""
    vid, src, site = video["id"], video["source"], str(site_path(video["id"]))
    py = [sys.executable, "-m"]
    signal = bool(yaml.safe_load(Path(site).read_text(encoding="utf-8")).get("signal_roi_px"))
    tracks, events, features = (str(OUT / d / f"{vid}.csv") for d in ("tracks", "events", "features"))
    steps = [("track", py + ["tracking.track", "--video", src, "--site", site]),
             ("events", py + ["tracking.events", "--tracks", tracks, "--site", site])]
    if signal:
        steps.append(("signal", py + ["traffic_light.color", "--video", src, "--site", site]))
    steps += [("features", py + ["features.build", "--events", events, "--tracks", tracks, "--site", site]
               + (["--signal", str(OUT / "signal" / f"{vid}.csv")] if signal else [])),
              ("rules", py + ["rules.rule", "--features", features, "--site", site])]
    return steps


def rename_output(folder: str, video: dict) -> None:
    """track·color는 원래 영상 이름으로 CSV를 쓴다 → video id 이름으로 바꾼다(신호 시간표는 video 칸도)."""
    src = OUT / folder / f"{Path(video['source']).stem}.csv"
    dst = OUT / folder / f"{video['id']}.csv"
    if src != dst:
        os.replace(src, dst)
    if folder == "signal":
        df = pd.read_csv(dst, encoding="utf-8-sig")
        df.assign(video=video["id"]).to_csv(dst, index=False, encoding="utf-8-sig")


def run_video(job: Path, video: dict, log) -> str:
    """영상 하나를 끝까지 돌린다. 실패하면 이유(로그 마지막 줄들), 아니면 ""."""
    for name, cmd in step_commands(video):
        status = read_status(job)
        update_status(job, videos={**status["videos"], video["id"]: name}, step_started=datetime.now().isoformat(timespec="seconds"))
        log.write(f"\n=== {video['id']} {name}: {' '.join(cmd)}\n")
        log.flush()
        if subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **UNBUFFERED}).returncode != 0:
            log.flush()
            tail = (job / "log.txt").read_text(encoding="utf-8", errors="replace").strip().splitlines()[-3:]
            return f"{name} 단계 실패: " + " / ".join(tail)
        if name in ("track", "signal"):
            rename_output("tracks" if name == "track" else "signal", video)
    return ""


def run_clips(job: Path, video: dict, events: list[int], log) -> str:
    """고른 이벤트(events)의 위반 의심 클립을 만들고 DB를 쓰면 clip 칸까지 다시 넣는다. 실패하면 이유, 아니면 "". 판정 결과는 바뀌지 않는다."""
    from dashboard import db

    vid = video["id"]
    judged = OUT / "judged" / f"{vid}.csv"
    cmd = [sys.executable, "-m", "eval.clips", "--judged", str(judged), "--tracks", str(OUT / "tracks" / f"{vid}.csv"), "--video", video["source"],
           "--events", ",".join(map(str, events))]
    log.write(f"\n=== {vid} clips: {' '.join(cmd)}\n")
    log.flush()
    if subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **UNBUFFERED}).returncode != 0:
        log.flush()
        return "클립 생성 실패: " + " / ".join((job / "log.txt").read_text(encoding="utf-8", errors="replace").strip().splitlines()[-3:])
    if db.configured():
        try:
            db.save_events(pd.read_csv(judged, encoding="utf-8-sig", dtype={"clip": str}))
        except Exception as e:  # 판정은 이미 DB에 있으므로 클립 경로만 못 넣은 것으로 남긴다
            return f"클립 경로를 DB에 넣지 못함: {e}"
    return ""


def track_progress(job: Path, vid: str) -> float | None:
    """지금 검출·추적 중인 영상의 진행률(0~1). log.txt에서 그 영상 track 단계 뒤의 마지막 진행 줄을 읽는다. 아직 없으면 None."""
    log = (job / "log.txt").read_text(encoding="utf-8", errors="replace") if (job / "log.txt").exists() else ""
    part = log.rsplit(f"=== {vid} track:", 1)
    found = re.findall(TRACK_LINE, part[1]) if len(part) == 2 else []
    if not found:
        return None
    done, total = map(int, found[-1])
    return done / total if total else None


def job_progress(job: Path, status: dict) -> tuple[float, dict]:
    """(작업 전체 진행률 0~1, 영상별 진행률 {id: 0~1 또는 None}). 검출·추적이 거의 모든 시간을 차지하므로
    영상 하나 = track 진행률로 보고, 끝난 영상은 1, 아직 안 한 영상은 0."""
    per = {}
    for vid, step in status["videos"].items():
        per[vid] = 1.0 if step in ("완료", "실패") else (track_progress(job, vid) or 0.0) if step == "track" else             0.99 if step in STEPS else 0.0
    return (sum(per.values()) / len(per) if per else 0.0), per


def run_job(job: Path) -> None:
    """작업 프로세스 본체: 영상마다 파이프라인 → DB → 리포트 → 메일."""
    from dashboard import db, mailer, report_pdf  # 대시보드 화면에서 이 파일을 읽을 때는 필요 없다

    spec = json.loads((job / "job.json").read_text(encoding="utf-8"))
    update_status(job, state="진행 중", started=datetime.now().strftime("%Y-%m-%d %H:%M"))
    errors, judged = {}, []
    with open(job / "log.txt", "a", encoding="utf-8") as log:
        for video in spec["videos"]:
            error = run_video(job, video, log)
            status = read_status(job)
            if error:
                errors[video["id"]] = error
                update_status(job, videos={**status["videos"], video["id"]: "실패"}, errors=errors)
                continue
            df = pd.read_csv(OUT / "judged" / f"{video['id']}.csv", encoding="utf-8-sig", dtype={"clip": str})
            if db.configured():
                db.save_events(df)
            judged.append(df)
            update_status(job, videos={**status["videos"], video["id"]: "완료"})
    report = None
    if judged:
        min_n = yaml.safe_load(EVAL_CONFIG.read_text(encoding="utf-8"))["min_n_reference"]
        done = {v["id"]: v["intersection"] for v in spec["videos"] if v["id"] not in errors}
        report = report_pdf.build_report(pd.concat(judged, ignore_index=True), job / "report.pdf", min_n,
                                         videos=done, red_mode=spec["red_mode"])
    mail = "보내지 않음(메일 주소 없음)"
    if spec["email"] and report:
        if not mailer.configured():
            mail = "보내지 못함: 보내는 메일 계정 설정이 없습니다(.env SMTP_*)"
        else:
            try:
                mailer.send_report(spec["email"], report)
                mail = f"보냄: {spec['email']}"
            except Exception as e:  # 메일 서버 오류는 분석 결과와 상관없으니 상태에만 남긴다
                mail = f"보내지 못함: {e}"
    elif spec["email"]:
        mail = "보내지 못함: 분석에 성공한 영상이 없습니다"
    update_status(job, report=str(report) if report else "", mail=mail)
    clip_errors = {}
    finished = [v for v in spec["videos"] if v["id"] not in errors]
    chosen = {}  # 영상 id: 클립을 만들 이벤트 번호(이번 작업의 영상들에서 함께 고른다)
    if judged:
        cfg = yaml.safe_load(EVAL_CONFIG.read_text(encoding="utf-8"))
        merged = pd.concat([df.assign(job_video=v["id"]) for v, df in zip(finished, judged)], ignore_index=True)
        chosen = select_clip_events(merged, cfg, spec["red_mode"]).groupby("job_video")["event_id"].apply(list).to_dict()
    with open(job / "log.txt", "a", encoding="utf-8") as log:
        for video in finished:
            if video["id"] not in chosen:  # 위반 의심이 없거나 뽑히지 않은 영상
                continue
            update_status(job, videos={**read_status(job)["videos"], video["id"]: "clips"})
            error = run_clips(job, video, chosen[video["id"]], log)
            if error:
                clip_errors[video["id"]] = error
            update_status(job, videos={**read_status(job)["videos"], video["id"]: "완료"}, clip_errors=clip_errors)
    update_status(job, state="완료" if judged else "실패",
                  finished=datetime.now().strftime("%Y-%m-%d %H:%M"))


def main() -> None:
    parser = argparse.ArgumentParser(description="대시보드 분석 작업 실행(대시보드가 띄운다)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run").add_argument("job")
    args = parser.parse_args()
    run_job(Path(args.job))


if __name__ == "__main__":
    main()
