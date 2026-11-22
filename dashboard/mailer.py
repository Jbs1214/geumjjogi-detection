# 담당: 장범수
"""분석 리포트 PDF를 메일로 보낸다(표준 라이브러리 smtplib). 통계 PDF만 붙이고 영상·클립은 보내지 않는다.
보내는 계정은 레포 루트 .env(따옴표 없이 KEY=값, dashboard/db.py와 같은 파일):
  SMTP_HOST, SMTP_PORT(465 = SSL, 그 밖은 STARTTLS), SMTP_USER, SMTP_PASSWORD, SMTP_FROM(생략하면 SMTP_USER)
"""
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

from dashboard.db import load_env
from dashboard.report_pdf import NOTICE

SSL_PORT = 465


def configured() -> bool:
    load_env()
    return bool(os.environ.get("SMTP_HOST"))


def build_message(to: str, pdf: Path) -> EmailMessage:
    pdf = Path(pdf)
    msg = EmailMessage()
    msg["Subject"] = "[우회전 위반 의심 분석] 분석이 끝났습니다"
    msg["From"] = os.environ.get("SMTP_FROM") or os.environ["SMTP_USER"]
    msg["To"] = to
    msg.set_content(f"요청하신 영상 분석이 끝나 리포트를 보내 드립니다(첨부 PDF).\n\n※ {NOTICE}\n")
    msg.add_attachment(pdf.read_bytes(), maintype="application", subtype="pdf", filename=pdf.name)
    return msg


def send_report(to: str, pdf: Path) -> None:
    load_env()
    msg = build_message(to, pdf)
    host, port = os.environ["SMTP_HOST"], int(os.environ.get("SMTP_PORT", SSL_PORT))
    smtp = smtplib.SMTP_SSL(host, port) if port == SSL_PORT else smtplib.SMTP(host, port)
    with smtp:
        if port != SSL_PORT:
            smtp.starttls()
        smtp.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
        smtp.send_message(msg)
