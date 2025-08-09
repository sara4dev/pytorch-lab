import torch

print("torch.__version__", torch.__version__)
print("torch.cuda.is_available()", torch.cuda.is_available())
print("torch.cuda.get_device_name(0)", torch.cuda.get_device_name(0))
print("torch.cuda.get_device_capability(0)", torch.cuda.get_device_capability(0))

# 1) What float dtypes exist in *this* torch build?
names = [
    "float64", "float32", "bfloat16", "float16",
    "float8_e4m3fn", "float8_e5m2",           # common FP8 names
    "float8_e4m3fnuz", "float8_e5m2fnuz",     # some builds use these
]
available = [getattr(torch, n) for n in names if hasattr(torch, n)]
print("Floating dtypes in this torch build:",
      [str(dt).replace("torch.","") for dt in available])

# 2) Which of those can actually perform operations on CUDA?
cuda_ok = []
for dt in available:
    try:
        a = torch.tensor([1.0], device="cuda", dtype=dt)
        b = torch.tensor([2.0], device="cuda", dtype=dt)
        c = a + b  # Simple addition test
        cuda_ok.append(dt)
    except Exception:
        pass
print("Works on CUDA:",
      [str(dt).replace("torch.","") for dt in cuda_ok])

# (Optional) TF32 isn't a dtype—just a matmul mode for FP32:
print("TF32 allowed (backend):", torch.backends.cuda.matmul.allow_tf32)
