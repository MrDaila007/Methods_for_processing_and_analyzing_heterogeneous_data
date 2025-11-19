#!/bin/bash
# Script to install GPU dependencies for Lab1

set -e

echo "=== GPU Dependencies Installation Script ==="
echo ""

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ Error: nvidia-smi not found. Please install NVIDIA drivers first."
    exit 1
fi

echo "✓ NVIDIA driver detected"
echo ""

# Install base dependencies
echo "Installing base dependencies..."
pip install -r requirements.txt

# Try to detect CUDA version
CUDA_VERSION=$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader | head -n 1 | cut -d'.' -f1,2 | sed 's/\.//')

if [ -z "$CUDA_VERSION" ]; then
    echo "⚠️  Could not detect CUDA version automatically."
    echo "Please choose your CUDA version:"
    echo "1) CUDA 11.x (install cuml-cu11)"
    echo "2) CUDA 12.x (install cuml-cu12)"
    read -p "Enter choice (1 or 2): " choice
    
    case $choice in
        1)
            CUDA_PKG="cuml-cu11"
            ;;
        2)
            CUDA_PKG="cuml-cu12"
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac
else
    if [[ "$CUDA_VERSION" == "11"* ]] || [[ "$CUDA_VERSION" == "10"* ]]; then
        CUDA_PKG="cuml-cu11"
        echo "Detected CUDA 11.x, installing cuml-cu11..."
    elif [[ "$CUDA_VERSION" == "12"* ]]; then
        CUDA_PKG="cuml-cu12"
        echo "Detected CUDA 12.x, installing cuml-cu12..."
    else
        echo "⚠️  Unsupported CUDA version: $CUDA_VERSION"
        echo "Defaulting to cuml-cu11. If this doesn't work, install manually."
        CUDA_PKG="cuml-cu11"
    fi
fi

# Install cuML
echo ""
echo "Installing $CUDA_PKG from NVIDIA PyPI..."
pip install $CUDA_PKG --extra-index-url=https://pypi.nvidia.com

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Verifying installation..."

python3 << EOF
try:
    from cuml.ensemble import RandomForestRegressor
    from cuml.linear_model import LogisticRegression
    print("✓ cuML is installed and ready!")
except ImportError as e:
    print(f"✗ cuML import failed: {e}")
    print("Please check your CUDA installation and try again.")
EOF

echo ""
echo "To use GPU acceleration, set USE_GPU = True in the notebook."

