# 담당: 장범수
"""YOLO11n과 YOLO26s의 클래스별 검출 재현율·속도를 비교한다.

입력: 박스 라벨이 있는 프레임(data/boxes/)의 YOLO 형식 data.yaml, 비교할 모델 이름들
출력: 모델별 클래스 재현율·mAP·FPS 표 (docs/experiments.md에 옮겨 적는다)
"""
import argparse


def compare(models: list[str], data: str) -> None:
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="검출 모델 비교")
    parser.add_argument("--models", nargs="+", default=["yolo11n.pt", "yolo26s.pt"])
    parser.add_argument("--data", required=True, help="YOLO 형식 data.yaml")
    args = parser.parse_args()
    compare(args.models, args.data)


if __name__ == "__main__":
    main()
