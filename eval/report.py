# 담당: 육주영
"""교차로·시간대별 위반율 = 미양보 / (양보 + 미양보)를 Wilson 95% 신뢰구간과 함께 표·그래프로 만든다.
판정 대상이 min_n_reference(configs/eval.yaml, 초기값 30) 미만인 칸은 "참고용"으로 표시한다. 적색 위반율(미정지 / (일시정지 + 미정지))은 따로 집계한다.
시간대는 영상 파일 이름({교차로}_{YYYYMMDD}_{HHMM})에서 읽는다.

입력: outputs/judged/*.csv, data/labels/*.csv (정답 구간 비교용), configs/eval.yaml
출력: outputs/report/yield_rate.csv, outputs/report/yield_rate.png, 지표 출력

지금은 1단계(보행자)만 집계한다. 적색 위반율은 MVP 뒤에 추가한다.
"""
import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 화면 없이 파일로만 저장
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from eval.metrics import bootstrap_ci, kappa, match_events, no_yield_f1, yield_rate_error


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(center - half, 0.0), min(center + half, 1.0)  # 계산 오차로 0~1을 벗어나지 않게


def violation_rate(results: pd.Series) -> float:
    """미양보 / (양보 + 미양보). 판정 대상이 없으면 nan."""
    n = results.isin(["yield", "no_yield"]).sum()
    return (results == "no_yield").sum() / n if n else math.nan


def rate_table(judged: pd.DataFrame, min_n_reference: int) -> pd.DataFrame:
    rows = []
    for (intersection, slot), g in judged.groupby(["intersection", "slot"]):
        n = int(g["ped_result"].isin(["yield", "no_yield"]).sum())
        k = int((g["ped_result"] == "no_yield").sum())
        low, high = wilson_ci(k, n)
        rows.append({
            "intersection": intersection,
            "slot": slot,
            "n_target": n,
            "n_no_yield": k,
            "violation_rate": k / n if n else math.nan,
            "ci_low": low,
            "ci_high": high,
            "note": "참고용" if n < min_n_reference else "",
        })
    return pd.DataFrame(rows)


def plot_rates(table: pd.DataFrame, path: Path) -> None:
    plt.rcParams["font.family"] = "Malgun Gothic"
    t = table.dropna(subset=["violation_rate"])
    x = t["intersection"] + " " + t["slot"]
    yerr = [t["violation_rate"] - t["ci_low"], t["ci_high"] - t["violation_rate"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, t["violation_rate"], yerr=yerr, capsize=4)
    ax.set_ylim(0, 1)
    ax.set_ylabel("위반율 (미양보 ÷ 판정 대상)")
    ax.set_title("교차로·시간대별 보행자 위반율 (Wilson 95% 신뢰구간)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate(judged: pd.DataFrame, labels_dir: str, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """라벨 파일마다 지표를 계산한다. (지표 표, 이중 라벨 kappa 표)를 돌려준다."""
    labels = {
        p.stem: pd.read_csv(p, encoding="utf-8-sig")
        for p in sorted(Path(labels_dir).glob("*.csv"))
        if not p.stem.endswith("_red")
    }
    rows = []
    for gt in labels.values():
        video = gt["video"].iloc[0]
        if video not in judged["video"].values:
            continue  # 아직 판정 결과가 없는 영상의 라벨표
        pairs = match_events(gt, judged, cfg["match_window_s"])
        y_true = [gt.loc[i, "ped_label"] for i, _ in pairs]
        y_pred = [judged.loc[j, "ped_result"] for _, j in pairs]
        # 짝이 하나도 없으면 F1을 계산할 수 없다(재현율 0으로만 남긴다)
        f1 = no_yield_f1(y_true, y_pred) if pairs else math.nan
        ci_low, ci_high = bootstrap_ci(y_true, y_pred, cfg["bootstrap_n"]) if pairs else (math.nan, math.nan)
        rows.append({
            "video": video,
            "labeler": gt["labeler"].iloc[0],
            "n_label": len(gt),
            "n_reviewed": len(pairs),
            "extraction_recall": len(pairs) / len(gt),
            "no_yield_f1": f1,
            "f1_ci_low": ci_low,
            "f1_ci_high": ci_high,
            "rate_error_pp": yield_rate_error(
                violation_rate(gt["ped_label"]),
                violation_rate(judged.loc[judged["video"] == video, "ped_result"]),
            ),
        })

    kappa_rows = []
    by_video: dict[str, list[pd.DataFrame]] = {}
    for gt in labels.values():
        by_video.setdefault(gt["video"].iloc[0], []).append(gt)
    for video, pair in by_video.items():
        if len(pair) == 2:
            a, b = pair
            pairs = match_events(a, b, cfg["match_window_s"])
            kappa_rows.append({
                "video": video,
                "n_both": len(pairs),
                "kappa": kappa(
                    [a.loc[i, "ped_label"] for i, _ in pairs],
                    [b.loc[j, "ped_label"] for _, j in pairs],
                ),
            })
    return pd.DataFrame(rows), pd.DataFrame(kappa_rows)


def report(judged_dir: str, labels_dir: str, config: str, out_dir: str = "outputs/report") -> None:
    with open(config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    judged = pd.concat(
        [pd.read_csv(p, encoding="utf-8-sig") for p in sorted(Path(judged_dir).glob("*.csv"))],
        ignore_index=True,
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table = rate_table(judged, cfg["min_n_reference"])
    table.to_csv(out / "yield_rate.csv", index=False, encoding="utf-8-sig")
    plot_rates(table, out / "yield_rate.png")
    print(table.to_string(index=False))

    metrics, kappas = evaluate(judged, labels_dir, cfg)
    if not metrics.empty:
        print(metrics.to_string(index=False))
    if not kappas.empty:
        print(kappas.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="교차로·시간대별 위반율 보고")
    parser.add_argument("--judged", default="outputs/judged")
    parser.add_argument("--labels", default="data/labels")
    parser.add_argument("--config", default="configs/eval.yaml")
    args = parser.parse_args()
    report(args.judged, args.labels, args.config)


if __name__ == "__main__":
    main()
