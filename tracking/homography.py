# 담당: 김민영
"""픽셀 좌표를 위에서 본 미터 단위 평면 좌표로 바꾼다.

횡단보도 네 모서리(픽셀)와 실측 폭·길이(m)로 homography 행렬을 만든다.
"""
import cv2
import numpy as np


def compute_homography(corners_px: list[list[float]], width_m: float, length_m: float) -> np.ndarray:
    """모서리 4점(좌상·우상·우하·좌하) → 3x3 행렬."""
    src = np.asarray(corners_px, dtype=np.float32)
    if src.shape != (4, 2):
        raise ValueError(f"corners_px는 4점 [[x, y], ...]이어야 한다: shape {src.shape}")
    if width_m <= 0 or length_m <= 0:
        raise ValueError(f"width_m·length_m은 0보다 커야 한다: {width_m}, {length_m}")
    # 미터 평면의 같은 순서 모서리. 원점 = 좌상, x = 폭 방향, y = 길이 방향 (docs/csv_schema.md)
    dst = np.array([[0, 0], [width_m, 0], [width_m, length_m], [0, length_m]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def to_meters(H: np.ndarray, points_px: np.ndarray) -> np.ndarray:
    """(N, 2) 픽셀 좌표 → (N, 2) 미터 좌표."""
    pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    homog = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T  # [x, y, 1] → [X, Y, W]
    # 원근 때문에 W가 점마다 다르다. W로 나눠야 실제 평면 좌표가 된다.
    return homog[:, :2] / homog[:, 2:3]
