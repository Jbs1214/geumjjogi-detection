# 담당: 육주영
"""Random Forest·XGBoost 2클래스(yield / no_yield) 분류기를 영상 단위 6-fold로 학습·평가한다.
같은 영상의 이벤트가 학습과 평가에 섞이지 않게 영상 이름으로 나눈다.

입력: outputs/features/*.csv, data/labels/*.csv (exclude 라벨은 뺀다)
출력: fold별 F1 표, weights/cls_{rf|xgb}_v{번호}_{YYMMDD}.joblib
"""
import argparse


def train(features_dir: str, labels_dir: str, model: str) -> None:
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="양보 판정 분류기 학습")
    parser.add_argument("--features", default="outputs/features")
    parser.add_argument("--labels", default="data/labels")
    parser.add_argument("--model", choices=["rf", "xgb"], default="rf")
    args = parser.parse_args()
    train(args.features, args.labels, args.model)


if __name__ == "__main__":
    main()
