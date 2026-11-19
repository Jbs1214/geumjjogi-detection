# 담당: 육주영
"""대시보드의 MySQL 저장(선택 기능): 로그인 계정(users, 아이디는 이메일)과 판정 이벤트(events).
레포 루트 .env에 DB_HOST가 있을 때만 쓴다. 없으면 대시보드는 지금처럼 outputs/judged의 CSV를 읽는다.
.env 키(따옴표 없이 KEY=값): DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME

실행: python -m dashboard.db init                  # 표 만들기
      python -m dashboard.db admin                 # 첫 관리자 계정(비밀번호는 화면에 보이지 않게 입력)
      python -m dashboard.db import outputs/judged  # 판정 CSV를 events 표에 넣기(영상 단위로 다시 넣음)
"""
import argparse
import getpass
import os
from pathlib import Path

import mysql.connector
import pandas as pd

from dashboard.auth import PASSWORD_MIN, hash_password

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
USER_FIELDS = {  # 가입 때 받는 칸: 최대 글자 수
    "email": 255, "name": 50, "agency": 100, "department": 100, "job_position": 50, "phone": 30,
}
# 대시보드가 쓰는 칸과 식별 칸만 둔다(이름은 docs/csv_schema.md 5절). 단계별 CSV가 원본이다
EVENT_COLUMNS = ["video", "event_id", "intersection", "wall_time", "slot",
                 "vehicle_type", "ped_result", "red_result", "status", "clip"]
TABLES = [
    f"""CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        {", ".join(f"{name} VARCHAR({size}) NOT NULL DEFAULT ''" for name, size in USER_FIELDS.items())},
        password_hash VARCHAR(200) NOT NULL,
        role VARCHAR(10) NOT NULL DEFAULT 'viewer',
        status VARCHAR(10) NOT NULL DEFAULT 'pending',
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        approved_at DATETIME NULL,
        UNIQUE (email)
    ) CHARACTER SET utf8mb4""",
    """CREATE TABLE IF NOT EXISTS events (
        video VARCHAR(100) NOT NULL,
        event_id INT NOT NULL,
        intersection VARCHAR(50) NOT NULL,
        wall_time DATETIME NOT NULL,
        slot VARCHAR(20) NOT NULL,
        vehicle_type VARCHAR(20) NULL,
        ped_result VARCHAR(20) NULL,
        red_result VARCHAR(20) NULL,
        status VARCHAR(20) NULL,
        clip VARCHAR(500) NULL,
        PRIMARY KEY (video, event_id)
    ) CHARACTER SET utf8mb4""",
]


def load_env() -> None:
    """.env의 KEY=값 줄을 환경변수로 읽는다(이미 있는 환경변수는 두고)."""
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and not key.lstrip().startswith("#"):
            os.environ.setdefault(key.strip(), value.strip())


def configured() -> bool:
    load_env()
    return bool(os.environ.get("DB_HOST"))


def connect():
    load_env()
    return mysql.connector.connect(
        host=os.environ["DB_HOST"],
        port=int(os.environ["DB_PORT"]),
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        database=os.environ["DB_NAME"],
    )


def run(sql: str, params: tuple = (), many: bool = False) -> list[dict]:
    """SQL 하나(many면 params 줄마다)를 실행하고 커밋한다. SELECT면 줄들을 dict로 돌려준다."""
    conn = connect()
    try:
        cur = conn.cursor(dictionary=True)
        if many:
            cur.executemany(sql, params)
        else:
            cur.execute(sql, params)
        rows = cur.fetchall() if cur.with_rows else []
        conn.commit()
        return rows
    finally:
        conn.close()


def init_db() -> None:
    for sql in TABLES:
        run(sql)


def add_user(user: dict, password: str, role: str = "viewer", status: str = "pending") -> bool:
    """가입 신청을 넣는다. 이미 가입된 이메일이면 False."""
    columns = [*USER_FIELDS, "password_hash", "role", "status"]
    values = (*(user[name] for name in USER_FIELDS), hash_password(password), role, status)
    try:
        run(f"INSERT INTO users ({', '.join(columns)}) VALUES ({', '.join(['%s'] * len(columns))})", values)
    except mysql.connector.IntegrityError:
        return False
    return True


def get_user(email: str) -> dict | None:
    rows = run("SELECT * FROM users WHERE email = %s", (email,))
    return rows[0] if rows else None


def pending_users() -> list[dict]:
    return run("SELECT * FROM users WHERE status = 'pending' ORDER BY created_at")


def approve_user(user_id: int) -> None:
    run("UPDATE users SET status = 'approved', approved_at = NOW() WHERE id = %s AND status = 'pending'", (user_id,))


def reject_user(user_id: int) -> None:
    """승인 대기 중인 신청을 지운다."""
    run("DELETE FROM users WHERE id = %s AND status = 'pending'", (user_id,))


def save_events(judged: pd.DataFrame) -> int:
    """판정 이벤트를 넣는다. 같은 영상의 줄은 먼저 지워 여러 번 넣어도 같다. 넣은 줄 수를 돌려준다."""
    rows = judged.reindex(columns=EVENT_COLUMNS).astype(object)
    rows = rows.where(rows.notna(), None)  # 빈 값은 NULL
    conn = connect()
    try:
        cur = conn.cursor()
        cur.executemany("DELETE FROM events WHERE video = %s", [(video,) for video in sorted(set(rows["video"]))])
        cur.executemany(
            f"INSERT INTO events ({', '.join(EVENT_COLUMNS)}) VALUES ({', '.join(['%s'] * len(EVENT_COLUMNS))})",
            rows.values.tolist(),
        )
        conn.commit()
    finally:
        conn.close()
    return len(rows)


def fetch_events() -> pd.DataFrame:
    """events 표 전체. dashboard.app.load_judged와 같은 모양(date 칸, clip 빈 값은 "")이고 없으면 빈 표."""
    rows = run(f"SELECT {', '.join(EVENT_COLUMNS)} FROM events")
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["wall_time"]).dt.date
    df["clip"] = df["clip"].fillna("")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    sub.add_parser("admin")
    sub.add_parser("import").add_argument("judged_dir")
    args = parser.parse_args()

    if args.command == "init":
        init_db()
        print("표 준비 완료: users, events")
    elif args.command == "admin":
        user = dict.fromkeys(USER_FIELDS, "")
        user["email"], user["name"], user["agency"] = input("이메일(아이디): "), input("이름: "), input("소속 기관: ")
        password = getpass.getpass("비밀번호: ")
        if len(password) < PASSWORD_MIN:
            parser.exit(1, f"비밀번호는 {PASSWORD_MIN}자 이상이어야 한다.\n")
        created = add_user(user, password, role="admin", status="approved")
        print("관리자 계정을 만들었다." if created else "이미 가입된 이메일이다.")
    else:
        files = sorted(Path(args.judged_dir).glob("*.csv"))
        judged = pd.concat([pd.read_csv(p, encoding="utf-8-sig", dtype={"clip": str}) for p in files], ignore_index=True)
        print(f"{len(files)}개 파일, {save_events(judged)}건을 넣었다.")


if __name__ == "__main__":
    main()
