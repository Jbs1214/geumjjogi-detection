# 담당: 김민영
"""픽셀 좌표를 위에서 본 미터 단위 평면 좌표로 바꾼다.

횡단보도 네 모서리(픽셀)와 실측 폭·길이(m)로 homography 행렬을 만든다.
"""
import numpy as np


def compute_homography(corners_px: list[list[float]], width_m: float, length_m: float) -> np.ndarray:
    """모서리 4점(좌상·우상·우하·좌하) → 3x3 행렬."""
    raise NotImplementedError


def to_meters(H: np.ndarray, points_px: np.ndarray) -> np.ndarray:
    """(N, 2) 픽셀 좌표 → (N, 2) 미터 좌표."""
    raise NotImplementedError
