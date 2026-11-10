# 담당: 육주영
"""공통 평가 지표 (기술서 6). F1은 판정 대상만 비교한다(exclude·na 제외)."""


def no_yield_f1(y_true: list[str], y_pred: list[str]) -> float:
    """주 지표: 미양보 클래스 F1."""
    raise NotImplementedError


def extraction_recall(gt_events: str, pred_events: str) -> float:
    """정답 구간에서 파이프라인이 잡은 이벤트 비율.
    같은 영상에서 t_enter_s 차이가 match_window_s(configs/eval.yaml, 초기값 2초) 이내이고 cls(차량·오토바이)가 같으면 같은 이벤트로 본다(기술서 11.1)."""
    raise NotImplementedError


def not_stopped_f1(y_true: list[str], y_pred: list[str]) -> float:
    """2단계: 미정지 클래스 F1 (stopped / not_stopped)."""
    raise NotImplementedError


def signal_agreement(red_labels: str, signal: str) -> float:
    """신호 색 판정 일치율. data/labels/<영상>_red.csv와 outputs/signal/<영상>.csv를 시간 단위로 비교."""
    raise NotImplementedError


def bootstrap_ci(y_true: list[str], y_pred: list[str], n: int) -> tuple[float, float]:
    """F1의 부트스트랩 95% 구간. n은 configs/eval.yaml의 bootstrap_n."""
    raise NotImplementedError


def kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """이중 라벨링 일치도 (Cohen's kappa)."""
    raise NotImplementedError


def yield_rate_error(gt_rate: float, pred_rate: float) -> float:
    """위반율 추정 오차 (%p)."""
    raise NotImplementedError
