# 담당: 육주영
"""공통 평가 지표 (기술서 6). F1은 판정 대상만 비교한다(exclude·na 제외)."""
import math

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score


def no_yield_f1(y_true: list[str], y_pred: list[str]) -> float:
    """주 지표: 미양보 클래스 F1.
    정답이 exclude인 이벤트는 뺀다. 예측이 exclude면 미양보를 못 잡은 것으로 센다."""
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if t != "exclude"]
    true = [t == "no_yield" for t, _ in pairs]
    pred = [p == "no_yield" for _, p in pairs]
    return float(f1_score(true, pred, zero_division=0))


def match_events(gt: pd.DataFrame, pred: pd.DataFrame, match_window_s: float) -> list[tuple]:
    """정답 이벤트와 추출 이벤트를 1:1로 짝지어 (gt 인덱스, pred 인덱스) 목록을 돌려준다.
    같은 video·cls이고 t_enter_s 차이가 match_window_s 이내인 것을 차이가 작은 순으로 짝짓는다."""
    m = gt.reset_index().merge(pred.reset_index(), on=["video", "cls"], suffixes=("_gt", "_pred"))
    m["diff"] = (m["t_enter_s_gt"] - m["t_enter_s_pred"]).abs()
    m = m[m["diff"] <= match_window_s].sort_values("diff")
    pairs, used_gt, used_pred = [], set(), set()
    for g, p in zip(m["index_gt"], m["index_pred"]):
        if g not in used_gt and p not in used_pred:
            pairs.append((g, p))
            used_gt.add(g)
            used_pred.add(p)
    return pairs


def extraction_recall(gt_events: str, pred_events: str, match_window_s: float) -> float:
    """정답 구간에서 파이프라인이 잡은 이벤트 비율.
    같은 영상에서 t_enter_s 차이가 match_window_s(configs/eval.yaml, 초기값 2초) 이내이고 cls(차량·오토바이)가 같으면 같은 이벤트로 본다(기술서 11.1)."""
    gt = pd.read_csv(gt_events, encoding="utf-8-sig")
    pred = pd.read_csv(pred_events, encoding="utf-8-sig")
    return len(match_events(gt, pred, match_window_s)) / len(gt)


def not_stopped_f1(y_true: list[str], y_pred: list[str]) -> float:
    """2단계: 미정지 클래스 F1 (stopped / not_stopped).
    정답이 exclude·na인 이벤트는 뺀다. 예측이 exclude·na면 미정지를 못 잡은 것으로 센다.
    정답에 판정 대상이 하나도 없으면 nan."""
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if t in ("stopped", "not_stopped")]
    if not pairs:
        return math.nan
    true = [t == "not_stopped" for t, _ in pairs]
    pred = [p == "not_stopped" for _, p in pairs]
    return float(f1_score(true, pred, zero_division=0))


def signal_agreement(red_labels: str, signal: str) -> float:
    """신호 색 판정 일치율. data/labels/<영상>_red.csv와 outputs/signal/<영상>.csv를 시간 단위로 비교."""
    raise NotImplementedError


def bootstrap_ci(y_true: list[str], y_pred: list[str], n: int, f1=no_yield_f1) -> tuple[float, float]:
    """F1의 부트스트랩 95% 구간. n은 configs/eval.yaml의 bootstrap_n. f1은 no_yield_f1(기본) 또는 not_stopped_f1.
    재표본에 판정 대상이 없어 F1이 nan인 회차는 빼고 구한다."""
    rng = np.random.default_rng(0)
    true, pred = np.array(y_true), np.array(y_pred)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(true), len(true))
        scores.append(f1(true[idx], pred[idx]))
    low, high = np.nanpercentile(scores, [2.5, 97.5])
    return float(low), float(high)


def kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """이중 라벨링 일치도 (Cohen's kappa)."""
    return float(cohen_kappa_score(labels_a, labels_b))


def yield_rate_error(gt_rate: float, pred_rate: float) -> float:
    """위반율 추정 오차 (%p)."""
    return abs(gt_rate - pred_rate) * 100
