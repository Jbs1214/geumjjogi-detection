# 개발환경 확인: GPU · 라이브러리가 쓸 수 있는 상태인지 검사한다.
# 실행: conda activate geumjjogi-detection 후 python check_env.py

from importlib import import_module


def check(name, func):
    try:
        print(f"[OK]   {name}: {func()}")
    except Exception as e:
        print(f"[FAIL] {name}: {e}")


def cuda_available():
    torch = import_module("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("CPU용 torch가 설치되었거나 드라이버 문제")
    return torch.version.cuda


def gpu_compute():
    torch = import_module("torch")
    x = torch.rand(1000, 1000, device="cuda")
    return f"{(x @ x).sum().item():.1f}"


for module in ["torch", "ultralytics", "cv2", "numpy", "pandas", "sklearn", "xgboost", "matplotlib", "yaml"]:
    check(f"{module} 버전", lambda m=module: import_module(m).__version__)
check("CUDA 사용 가능", cuda_available)
check("GPU 이름", lambda: import_module("torch").cuda.get_device_name(0))
check("GPU 계산", gpu_compute)
