# 담당: 장범수
"""검출 모델을 3클래스(vehicle·motorcycle·pedestrian)로 미세조정한다. 성능이 부족할 때만 쓴다.

입력: YOLO 형식 data.yaml (data/boxes/), 시작 가중치
출력: runs/ 학습 결과 → weights/det_{모델}_v{번호}_{YYMMDD}.pt 로 이름을 바꿔 Drive에 올리고 docs/weights.md에 기록
"""
import argparse


def train(data: str, model: str, epochs: int) -> None:
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="검출 모델 미세조정")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    train(args.data, args.model, args.epochs)


if __name__ == "__main__":
    main()
