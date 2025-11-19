#!/usr/bin/env python3
"""Script to install GPU dependencies for Lab1."""

import subprocess
import sys
import re

def run_command(cmd, check=True):
    """Run a shell command and return the output."""
    try:
        result = subprocess.run(
            cmd, shell=True, check=check, capture_output=True, text=True
        )
        return result.stdout.strip(), result.returncode
    except subprocess.CalledProcessError as e:
        return e.stdout.strip() + e.stderr.strip(), e.returncode

def check_nvidia_driver():
    """Check if NVIDIA driver is available."""
    output, code = run_command("nvidia-smi --query-gpu=name --format=csv,noheader", check=False)
    if code != 0:
        print("❌ Error: nvidia-smi not found. Please install NVIDIA drivers first.")
        return False
    print(f"✓ NVIDIA GPU detected: {output.split(chr(10))[0]}")
    return True

def detect_cuda_version():
    """Try to detect CUDA version from nvidia-smi."""
    output, code = run_command("nvidia-smi --query-gpu=cuda_version --format=csv,noheader", check=False)
    if code != 0 or not output:
        return None
    
    version = output.split(chr(10))[0].strip()
    # Extract major.minor version
    match = re.match(r'(\d+)\.(\d+)', version)
    if match:
        major = int(match.group(1))
        minor = int(match.group(2))
        return (major, minor)
    return None

def install_base_dependencies():
    """Install base CPU dependencies."""
    print("\n📦 Installing base dependencies...")
    output, code = run_command("pip install -r requirements.txt", check=False)
    if code != 0:
        print(f"❌ Failed to install base dependencies:\n{output}")
        return False
    print("✓ Base dependencies installed")
    return True

def install_cuml(cuda_version=None):
    """Install cuML based on CUDA version."""
    if cuda_version is None:
        print("\n⚠️  Could not detect CUDA version automatically.")
        print("Please choose your CUDA version:")
        print("1) CUDA 11.x (install cuml-cu11)")
        print("2) CUDA 12.x (install cuml-cu12)")
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            cuda_pkg = "cuml-cu11"
        elif choice == "2":
            cuda_pkg = "cuml-cu12"
        else:
            print("Invalid choice. Exiting.")
            return False
    else:
        major, minor = cuda_version
        if major == 11 or major == 10:
            cuda_pkg = "cuml-cu11"
            print(f"\n📦 Detected CUDA {major}.{minor}, installing cuml-cu11...")
        elif major == 12:
            cuda_pkg = "cuml-cu12"
            print(f"\n📦 Detected CUDA {major}.{minor}, installing cuml-cu12...")
        else:
            print(f"⚠️  Unsupported CUDA version: {major}.{minor}")
            print("Defaulting to cuml-cu11. If this doesn't work, install manually.")
            cuda_pkg = "cuml-cu11"
    
    print(f"Installing {cuda_pkg} from NVIDIA PyPI...")
    cmd = f"pip install {cuda_pkg} --extra-index-url=https://pypi.nvidia.com"
    output, code = run_command(cmd, check=False)
    
    if code != 0:
        print(f"❌ Failed to install {cuda_pkg}:\n{output}")
        print("\nYou may need to:")
        print("1. Check your CUDA installation")
        print("2. Install manually: pip install cuml-cu11 --extra-index-url=https://pypi.nvidia.com")
        print("3. Or use conda: conda install -c rapidsai -c conda-forge cuml")
        return False
    
    print(f"✓ {cuda_pkg} installed successfully")
    return True

def verify_installation():
    """Verify that cuML can be imported."""
    print("\n🔍 Verifying installation...")
    try:
        from cuml.ensemble import RandomForestRegressor
        from cuml.linear_model import LogisticRegression
        print("✓ cuML is installed and ready!")
        return True
    except ImportError as e:
        print(f"✗ cuML import failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check that CUDA toolkit is installed")
        print("2. Verify CUDA version matches cuML package")
        print("3. Try reinstalling: pip uninstall cuml-cu11 cuml-cu12 && pip install cuml-cu11 --extra-index-url=https://pypi.nvidia.com")
        return False

def main():
    """Main installation function."""
    print("=== GPU Dependencies Installation Script ===\n")
    
    # Check NVIDIA driver
    if not check_nvidia_driver():
        sys.exit(1)
    
    # Detect CUDA version
    cuda_version = detect_cuda_version()
    if cuda_version:
        print(f"✓ CUDA version detected: {cuda_version[0]}.{cuda_version[1]}")
    
    # Install base dependencies
    if not install_base_dependencies():
        sys.exit(1)
    
    # Install cuML
    if not install_cuml(cuda_version):
        sys.exit(1)
    
    # Verify installation
    if verify_installation():
        print("\n✅ Installation complete!")
        print("\nTo use GPU acceleration, set USE_GPU = True in the notebook.")
    else:
        print("\n⚠️  Installation completed but verification failed.")
        print("The code will fall back to CPU mode if GPU is not available.")

if __name__ == "__main__":
    main()

