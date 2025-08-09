# PyTorch Lab

A testing environment for exploring PyTorch CUDA capabilities, data types, and GPU compatibility.

## Overview

This lab provides tools to test and verify PyTorch installations, particularly focusing on:
- CUDA availability and device information
- Available floating-point data types in your PyTorch build
- CUDA allocation compatibility for different data types
- FP8 format support (float8_e4m3fn, float8_e5m2, etc.)
- TF32 backend configuration

## Files

### `test-dtypes.py`

A comprehensive Python script that:
1. **System Information**: Displays PyTorch version, CUDA availability, and GPU details
2. **Data Type Discovery**: Identifies all available floating-point data types in your PyTorch build
3. **CUDA Compatibility Testing**: Tests which data types can be allocated on CUDA devices
4. **Backend Configuration**: Shows TF32 matmul backend status

**Sample Output:**
```
torch.__version__ 2.4.0
torch.cuda.is_available() True
torch.cuda.get_device_name(0) NVIDIA GeForce RTX 4090
torch.cuda.get_device_capability(0) (8, 9)
Floating dtypes in this torch build: ['float64', 'float32', 'bfloat16', 'float16', 'float8_e4m3fn', 'float8_e5m2']
Alloc works on CUDA: ['float64', 'float32', 'bfloat16', 'float16']
TF32 allowed (backend): True
```

### `pytoch-pod.yaml`

Kubernetes deployment configuration for running PyTorch workloads with GPU support:
- Uses NVIDIA's official PyTorch container (`nvcr.io/nvidia/pytorch:24.12-py3`)
- Requests 1 GPU resource
- Mounts workspace volume
- Includes configuration options for GPU Operator and node selection

## Usage

### Local Testing

Run the data type test script:
```bash
python test-dtypes.py
```

### Kubernetes Deployment

Deploy the PyTorch pod with GPU support:
```bash
kubectl apply -f pytoch-pod.yaml
```

Access the running pod:
```bash
kubectl exec -it deployment/pytorch-lab -- bash
```

Copy and run the test script inside the pod:
```bash
kubectl cp test-dtypes.py pytorch-lab-<pod-id>:/workspace/
kubectl exec -it deployment/pytorch-lab -- python /workspace/test-dtypes.py
```

## Requirements

### Local Environment
- Python 3.8+
- PyTorch with CUDA support
- NVIDIA GPU with compatible drivers

### Kubernetes Environment
- Kubernetes cluster with GPU nodes
- NVIDIA GPU Operator or equivalent GPU support
- kubectl configured to access your cluster

## Tested Data Types

The script tests for these floating-point data types:
- **Standard**: `float64`, `float32`, `float16`
- **Brain Float**: `bfloat16`
- **FP8 Formats**: `float8_e4m3fn`, `float8_e5m2`, `float8_e4m3fnuz`, `float8_e5m2fnuz`

## Notes

- FP8 support varies by PyTorch version and hardware
- TF32 is not a data type but a computation mode for float32 operations
- Some data types may be available but not CUDA-allocatable depending on your GPU architecture

## Troubleshooting

If you encounter issues:
1. Verify CUDA installation: `nvidia-smi`
2. Check PyTorch CUDA build: `python -c "import torch; print(torch.cuda.is_available())"`
3. For Kubernetes: ensure GPU nodes are properly labeled and the GPU Operator is running
