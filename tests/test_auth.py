# 담당: 육주영
"""dashboard/auth.py 테스트. 실행: python -m unittest discover -s tests -t . -v"""
import unittest

from dashboard.auth import hash_password, signup_errors, verify_password

USER = {"email": "new@example.go.kr", "name": "김신청", "agency": "예시구청", "department": "", "job_position": "", "phone": ""}


class PasswordTest(unittest.TestCase):
    def test_hash_hides_password_and_differs_each_time(self):
        first, second = hash_password("password-1"), hash_password("password-1")
        self.assertNotIn("password-1", first)
        self.assertNotEqual(first, second)  # salt가 달라 같은 비밀번호도 다른 해시

    def test_only_right_password_passes(self):
        stored = hash_password("비밀번호-1")
        self.assertTrue(verify_password("비밀번호-1", stored))
        self.assertFalse(verify_password("비밀번호-2", stored))


class SignupErrorsTest(unittest.TestCase):
    def test_valid_input_has_no_errors(self):
        self.assertEqual(signup_errors(USER, "password-1", "password-1"), {})
        self.assertEqual(signup_errors({**USER, "phone": "010-1234-5678"}, "password-1", "password-1"), {})

    def test_each_rule(self):
        self.assertEqual(set(signup_errors({**USER, "email": "", "agency": ""}, "password-1", "password-1")), {"email", "agency"})
        for email in ["abc", "a@b", "a b@c.kr", "@c.kr"]:
            self.assertEqual(signup_errors({**USER, "email": email}, "password-1", "password-1"), {"email": "이메일 형식이 아닙니다."})
        self.assertEqual(set(signup_errors(USER, "short", "short")), {"password"})
        self.assertEqual(set(signup_errors(USER, "password-1", "password-2")), {"confirm"})
        for phone in ["전화", "010-12", "0101234567890"]:
            self.assertEqual(set(signup_errors({**USER, "phone": phone}, "password-1", "password-1")), {"phone"})


if __name__ == "__main__":
    unittest.main()
