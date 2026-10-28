# 담당: 육주영
"""공통 평가 지표. 판정은 yield / no_yield 두 값만 비교한다(exclude 제외)."""


def no_yield_f1(y_true: list[str], y_pred: list[str]) -> float:
    """주 지표: 미양보 클래스 F1."""
    raise NotImplementedError


def extraction_recall(gt_events: str, pred_events: str) -> float:
    """정답 구간에서 파이프라인이 잡은 이벤트 비율."""
    raise NotImplementedError


def kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """이중 라벨링 일치도 (Cohen's kappa)."""
    raise NotImplementedError


def yield_rate_error(gt_rate: float, pred_rate: float) -> float:
    """양보율 추정 오차 (%p)."""
    raise NotImplementedError
