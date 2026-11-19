# 담당: 육주영
"""대시보드 로그인용 비밀번호 해시와 가입 입력 검사. 비밀번호 원문은 저장하지 않고 scrypt 해시와 무작위 salt만 남긴다(표준 라이브러리)."""
import hashlib
import hmac
import re
import secrets

PASSWORD_MIN = 8  # 가입 때 받는 비밀번호 최소 길이
SCRYPT = {"n": 2**14, "r": 8, "p": 1}
REQUIRED_FIELDS = ("email", "name", "agency")
EMAIL_PATTERN = r"[^@\s]+@[^@\s]+\.[^@\s]+"
PHONE_PATTERN = r"01\d{8,9}"  # 휴대폰 번호: 숫자만(01011112222)


def signup_errors(user: dict, password: str, confirm: str) -> dict[str, str]:
    """가입 입력의 칸별 오류 메시지. 문제없는 칸은 들어 있지 않다. 키는 users 표의 칸 이름과 "password", "confirm"."""
    errors = {name: "필수 입력입니다." for name in REQUIRED_FIELDS if not user[name]}
    if user["email"] and not re.fullmatch(EMAIL_PATTERN, user["email"]):
        errors["email"] = "이메일 형식이 아닙니다."
    if not password:
        errors["password"] = "필수 입력입니다."
    elif len(password) < PASSWORD_MIN:
        errors["password"] = f"비밀번호는 {PASSWORD_MIN}자 이상이어야 합니다."
    if confirm != password:
        errors["confirm"] = "비밀번호 확인이 다릅니다."
    if user["phone"] and not re.fullmatch(PHONE_PATTERN, user["phone"]):
        errors["phone"] = "휴대폰 번호는 - 없이 숫자만 적어 주세요(예: 01011112222)."
    return errors


def format_phone(phone: str) -> str:
    """01011112222 -> 010-1111-2222 (관리자 화면 표시용). 숫자 10~11자리가 아니면 그대로 돌려준다."""
    parts = re.fullmatch(r"(\d{3})(\d{3,4})(\d{4})", phone)
    return "-".join(parts.groups()) if parts else phone


def hash_password(password: str) -> str:
    """저장할 문자열 "scrypt$<salt>$<hash>". 같은 비밀번호도 salt가 달라 매번 다르다."""
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, **SCRYPT)
    return f"scrypt${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    _, salt, digest = stored.split("$")
    candidate = hashlib.scrypt(password.encode("utf-8"), salt=bytes.fromhex(salt), **SCRYPT)
    return hmac.compare_digest(candidate.hex(), digest)
