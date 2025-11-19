# GPU Installation Guide

This guide explains how to install GPU dependencies for running the lab with cuML acceleration.

## Prerequisites

- NVIDIA GPU with CUDA Compute Capability 6.0 or higher
- CUDA Toolkit 11.0+ or 12.0+
- cuDNN (usually comes with CUDA installation)

## Method 1: Installation via pip (Recommended)

### For CUDA 11.x:
```bash
pip install cuml-cu11 --extra-index-url=https://pypi.nvidia.com
```

### For CUDA 12.x:
```bash
pip install cuml-cu12 --extra-index-url=https://pypi.nvidia.com
```

### Install base dependencies:
```bash
pip install -r requirements.txt
```

## Method 2: Installation via conda (Recommended for full RAPIDS stack)

### Create a new conda environment with RAPIDS:
```bash
# For CUDA 11.8
conda create -n rapids-env -c rapidsai -c conda-forge -c nvidia \
    cuml=23.12 python=3.10 cudatoolkit=11.8

# For CUDA 12.0
conda create -n rapids-env -c rapidsai -c conda-forge -c nvidia \
    cuml=23.12 python=3.10 cudatoolkit=12.0
```

### Activate the environment:
```bash
conda activate rapids-env
```

### Install remaining dependencies:
```bash
pip install -r requirements.txt
```

## Verify Installation

Check if cuML is available:
```python
try:
    from cuml.ensemble import RandomForestRegressor
    from cuml.linear_model import LogisticRegression
    print("✓ cuML is installed and ready!")
except ImportError:
    print("✗ cuML is not available")
```

## Check CUDA Version

```bash
nvidia-smi
```

Or in Python:
```python
import subprocess
result = subprocess.run(['nvidia-smi', '--query-gpu=cuda_version', '--format=csv,noheader'], 
                       capture_output=True, text=True)
print(f"CUDA Version: {result.stdout.strip()}")
```

## Usage

In the notebook, set `USE_GPU = True` to enable GPU acceleration:
```python
USE_GPU = True
main(dataset_id=DATASET_ID, max_rows=MAX_ROWS, use_gpu=USE_GPU)
```

## Troubleshooting

### Issue: "CUDA not found"
- Ensure CUDA toolkit is installed and in PATH
- Check `nvidia-smi` works
- Verify CUDA version matches cuML requirements

### Issue: "cuML import fails"
- Reinstall cuML matching your CUDA version
- Check Python version compatibility (3.8-3.11 recommended)

### Issue: "Out of memory"
- Reduce `MAX_ROWS` parameter
- Use CPU mode if GPU memory is insufficient

## Notes

- cuML requires significant GPU memory (4GB+ recommended)
- For small datasets, CPU mode may be faster due to overhead
- GPU acceleration is most beneficial for large datasets and many combinations

