#!/bin/bash

# RLLaVA One-Click Installation Script
#
# Usage:
#   ./install.sh [--fast|-f]
#
# Environment Variables:
#   PYPI_MIRROR: Specify a custom PyPI mirror source
#     Examples:
#       export PYPI_MIRROR="https://mirrors.aliyun.com/pypi/simple/"
#       export PYPI_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
#       export PYPI_MIRROR="https://mirrors.cloud.tencent.com/pypi/simple"
#       export PYPI_MIRROR="https://mirrors.163.com/pypi/simple"
#
# Options:
#   --fast, -f: Skip version checks and force reinstall packages
#
# The script will automatically try multiple mirror sources if the primary one fails.

set -e  # Exit immediately if a command exits with a non-zero status

echo "🚀 Starting RLLaVA installation..."

# Check for fast mode
FAST_MODE=false
if [ "$1" = "--fast" ] || [ "$1" = "-f" ]; then
    FAST_MODE=true
    echo "⚡ Fast mode enabled - skipping version checks"
fi

# PyPI mirror configuration
# Default mirror sources (can be overridden by environment variable PYPI_MIRROR)
DEFAULT_MIRRORS=(
    "https://mirrors.aliyun.com/pypi/simple/"
    "https://pypi.tuna.tsinghua.edu.cn/simple"
    "https://mirrors.cloud.tencent.com/pypi/simple"
    "https://mirrors.163.com/pypi/simple"
)

# Set mirror source from environment variable or use default
if [ -n "$PYPI_MIRROR" ]; then
    PYPI_INDEX_URL="$PYPI_MIRROR"
    echo "🔧 Using custom PyPI mirror: $PYPI_INDEX_URL"
else
    PYPI_INDEX_URL="${DEFAULT_MIRRORS[0]}"
    echo "🔧 Using default PyPI mirror: $PYPI_INDEX_URL"
fi

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Install package with mirror fallback
install_with_mirror_fallback() {
    local package=$1
    local extra_flags=$2
    
    # Try primary mirror first
    if pip install $package $extra_flags -i "$PYPI_INDEX_URL"; then
        return 0
    fi
    
    # If primary mirror fails, try other mirrors
    print_warning "Failed to install $package from primary mirror, trying fallback mirrors..."
    for mirror in "${DEFAULT_MIRRORS[@]}"; do
        if [ "$mirror" != "$PYPI_INDEX_URL" ]; then
            print_info "Trying mirror: $mirror"
            if pip install $package $extra_flags -i "$mirror"; then
                print_success "Successfully installed $package from $mirror"
                return 0
            fi
        fi
    done
    
    # If all mirrors fail, try default PyPI
    print_warning "All mirrors failed, trying default PyPI..."
    if pip install $package $extra_flags; then
        print_success "Successfully installed $package from default PyPI"
        return 0
    fi
    
    print_error "Failed to install $package from all sources"
    return 1
}

# Check Python version
check_python_version() {
    print_info "Checking Python version..."
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    required_version="3.11"
    
    if python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)"; then
        print_success "Python version check passed: $python_version"
    else
        print_error "Python version does not meet requirements. Need >= 3.11, current version: $python_version"
        exit 1
    fi
}

# Check CUDA environment
check_cuda() {
    print_info "Checking CUDA environment..."
    if command -v nvidia-smi &> /dev/null; then
        cuda_version=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
        print_success "Detected CUDA version: $cuda_version"
        
        # Set CUDA_HOME if not already set or if it's invalid
        if [ -z "$CUDA_HOME" ] || [ ! -f "$CUDA_HOME/bin/nvcc" ]; then
            # Build a list of candidate paths to search
            candidate_paths=(
                "/usr/local/cuda"                    # Generic symlink (most common)
                "/opt/cuda"                          # Alternative location
                "/usr/local/cuda-${cuda_version}"    # Version-specific path
            )
            
            # Add all /usr/local/cuda-* directories dynamically
            if [ -d "/usr/local" ]; then
                for cuda_dir in /usr/local/cuda-*; do
                    if [ -d "$cuda_dir" ]; then
                        candidate_paths+=("$cuda_dir")
                    fi
                done
            fi
            
            # Add all /opt/cuda-* directories dynamically
            if [ -d "/opt" ]; then
                for cuda_dir in /opt/cuda-*; do
                    if [ -d "$cuda_dir" ]; then
                        candidate_paths+=("$cuda_dir")
                    fi
                done
            fi
            
            # Try each candidate path
            for cuda_path in "${candidate_paths[@]}"; do
                if [ -d "$cuda_path" ] && [ -f "$cuda_path/bin/nvcc" ]; then
                    export CUDA_HOME="$cuda_path"
                    export PATH="$CUDA_HOME/bin:$PATH"
                    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
                    print_info "Found CUDA installation at: $CUDA_HOME"
                    break
                fi
            done
        fi
        
        # Verify CUDA installation
        if [ -n "$CUDA_HOME" ] && [ -f "$CUDA_HOME/bin/nvcc" ]; then
            print_success "CUDA installation verified at: $CUDA_HOME"
            
            # Make CUDA_HOME available for all subprocesses
            export CUDA_HOME
            export PATH="$CUDA_HOME/bin:$PATH"
            export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
            
            # Check if PyTorch can see CUDA
            if python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
                print_success "PyTorch CUDA support available"
            else
                print_warning "PyTorch CUDA support not available"
            fi
        return 0
        else
            print_warning "CUDA installation found but nvcc not available"
            return 1
        fi
    else
        print_warning "CUDA environment not detected, will install CPU version of PyTorch"
        return 1
    fi
}

# Compare version numbers (returns 0 if v1 >= v2, 1 otherwise)
compare_versions() {
    local v1=$1
    local v2=$2
    
    # Use Python to compare versions properly
    python3 -c "
import sys
from packaging import version
try:
    if version.parse('$v1') >= version.parse('$v2'):
        sys.exit(0)
    else:
        sys.exit(1)
except:
    # Fallback to string comparison if packaging is not available
    if '$v1' >= '$v2':
        sys.exit(0)
    else:
        sys.exit(1)
" 2>/dev/null
}

# Check if package is already installed
check_package_installed() {
    local package_name=$1
    local required_version=$2
    
    # Extract package name without version
    local name=$(echo "$package_name" | cut -d'=' -f1)
    
    # Check if package is installed and get version
    local installed_version=$(pip show "$name" 2>/dev/null | grep "^Version:" | cut -d' ' -f2)
    
    if [ -n "$installed_version" ]; then
        if [ "$installed_version" = "$required_version" ]; then
            return 0  # Already installed with correct version
        else
            # Check if installed version is compatible (>= required)
            if compare_versions "$installed_version" "$required_version"; then
                # For packages that are already installed with newer versions, skip reinstallation
                # unless we're in fast mode
                if [ "$FAST_MODE" = false ]; then
                    print_info "Package $name is installed with version $installed_version (>= required $required_version), skipping..."
                    return 0  # Skip reinstallation
                else
                    print_info "Package $name is installed with version $installed_version, but need $required_version"
                    return 1  # Force reinstallation in fast mode
                fi
            else
                print_info "Package $name is installed with version $installed_version, but need $required_version"
                return 1  # Need to upgrade
            fi
        fi
    else
        return 1  # Not installed
    fi
}

# Install basic dependencies
install_basic_deps() {
    print_info "Installing basic dependencies..."
    
    # Basic packages (no compilation required)
    basic_packages=(
        "deprecated==1.2.18"
        "torch==2.6.0"
        "torchvision==0.21.0" 
        "torchaudio==2.6.0"
        "transformers==4.51.1"
        "accelerate==1.10.1"
        "peft==0.15.2"
        "trl==0.14.0"
        "torchdata==0.11.0"
        "datasets==3.3.2"
        "omegaconf==2.3.0"
        "tensordict==0.9.1"
        "codetiming==1.4.0"
        "tokenizers==0.21.4"
        "sentencepiece==0.2.0"
        "safetensors==0.5.3"
        "huggingface-hub==0.34.4"
        "numpy==1.26.4"
        "pandas==2.2.3"
        "pillow==11.1.0"
        "megfile==4.1.6"
        "qwen_vl_utils==0.0.10"
        "opencv-python-headless>=4.11.0"
        "av==14.2.0"
        "decord==0.6.0"
        "gradio==5.31.0"
        "fastapi==0.115.12"
        "uvicorn==0.34.0"
        "pydantic==2.10.6"
        "requests==2.32.5"
        "tqdm==4.67.1"
        "pyyaml==6.0.2"
        "scikit-learn==1.6.1"
        "scipy==1.15.2"
        "matplotlib==3.10.3"
        "seaborn==0.13.2"
        "wandb==0.18.3"
        "tensorboard==2.19.0"
        "jupyterlab==4.4.9"
        "ipython==9.2.0"
        "notebook==7.4.7"
    )
    
    local total=${#basic_packages[@]}
    local current=0
    
    for package in "${basic_packages[@]}"; do
        current=$((current + 1))
        local name=$(echo "$package" | cut -d'=' -f1)
        local version=$(echo "$package" | cut -d'=' -f2)
        
        if [ "$FAST_MODE" = false ] && check_package_installed "$name" "$version"; then
            print_info "✓ [$current/$total] $package is already installed"
        else
            print_info "[$current/$total] Installing $package..."
            install_with_mirror_fallback "$package" || {
                print_warning "Failed to install $package, continuing with next package..."
            }
        fi
    done
    
    print_success "Basic dependencies installation completed"
}

# Install compilation dependencies
install_compile_deps() {
    print_info "Installing compilation dependencies..."
    
    # Install compilation tools
    compile_packages=(
        "ninja==1.11.1.4"
        "einops==0.8.1"
    )
    
    for package in "${compile_packages[@]}"; do
        local name=$(echo "$package" | cut -d'=' -f1)
        local version=$(echo "$package" | cut -d'=' -f2)
        
        if check_package_installed "$name" "$version"; then
            print_info "✓ $package is already installed"
        else
            print_info "Installing $package..."
            install_with_mirror_fallback "$package" || {
                print_warning "Failed to install $package, continuing with next package..."
            }
        fi
    done
}

# Try to install optional packages
install_optional_packages() {
    print_info "Installing optional packages..."
    
    # Ensure CUDA environment is properly set up for compilation packages
    if [ -n "$CUDA_HOME" ] && [ -f "$CUDA_HOME/bin/nvcc" ]; then
        print_info "CUDA environment ready for compilation packages"
        export CUDA_HOME
        export PATH="$CUDA_HOME/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
    else
        print_warning "CUDA environment not properly configured, some packages may fail to install"
    fi
    
    # Install vllm dependencies first (before vllm itself)
    print_info "Installing vllm dependencies..."
    
    # Fix lark version conflict first (vllm requires lark==1.2.2, but jupyterlab installed 1.3.0)
    print_info "Fixing lark version conflict (downgrading to 1.2.2 for vllm compatibility)..."
    install_with_mirror_fallback "lark==1.2.2" || {
        print_warning "Failed to downgrade lark, continuing..."
    }
    
    vllm_deps=(
        "blake3"
        "cachetools"
        "compressed-tensors==0.9.3"
        "depyf==0.18.0"
        "gguf>=0.13.0"
        "lm-format-enforcer<0.11,>=0.10.11"
        "mistral_common[opencv]>=1.5.4"
        "msgspec"
        "numba==0.61.2"
        "openai>=1.52.0"
        "opentelemetry-api<1.27.0,>=1.26.0"
        "opentelemetry-exporter-otlp<1.27.0,>=1.26.0"
        "opentelemetry-sdk<1.27.0,>=1.26.0"
        "opentelemetry-semantic-conventions-ai<0.5.0,>=0.4.1"
        "outlines==0.1.11"
        "partial-json-parser"
        "prometheus-fastapi-instrumentator>=7.0.0"
        "ray[cgraph]!=2.44.*,>=2.43.0"
        "tiktoken>=0.6.0"
        "watchfiles"
    )
    
    # Platform-specific dependencies
    PLATFORM=$(uname -m)
    if [ "$PLATFORM" = "x86_64" ] || [ "$PLATFORM" = "arm64" ] || [ "$PLATFORM" = "aarch64" ]; then
        vllm_deps+=("llguidance<0.8.0,>=0.7.9")
        if [ "$PLATFORM" = "x86_64" ] || [ "$PLATFORM" = "aarch64" ]; then
            vllm_deps+=("xgrammar==0.1.18")
        fi
    fi
    
    for package in "${vllm_deps[@]}"; do
        local name=$(echo "$package" | cut -d'=' -f1 | cut -d'[' -f1)
        # Extract version requirement if present
        local version_req=$(echo "$package" | grep -oE '[>=<]+[0-9.]+' | head -1 || echo "")
        
        # Check if package is already installed (skip check for version ranges as they're complex)
        if [ -z "$version_req" ]; then
            if pip show "$name" &>/dev/null; then
                print_info "✓ vllm dependency $name is already installed, skipping..."
                continue
            fi
        fi
        
        print_info "Installing vllm dependency: $package..."
        install_with_mirror_fallback "$package" || {
            print_warning "Failed to install $package, continuing..."
        }
    done
    
    # Optional packages array
    optional_packages=(
        "xformers==0.0.29.post2"
        "bitsandbytes==0.45.3"
        "vllm==0.8.5.post1"
        "sglang==0.4.6.post1"
        "mathruler==0.1.0"
        "pylatexenc==2.10"
    )
    
    for package in "${optional_packages[@]}"; do
        local name=$(echo "$package" | cut -d'=' -f1)
        local version=$(echo "$package" | cut -d'=' -f2)
        
        if check_package_installed "$name" "$version"; then
            print_info "✓ $package is already installed"
        else
            print_info "Installing $package..."
            
            # Special handling for vllm: install without --no-deps since we already installed dependencies
            if [ "$name" = "vllm" ]; then
                install_with_mirror_fallback "$package" || {
                    print_warning "$package installation failed, skipping..."
                }
            else
                # Use --no-deps to avoid reinstalling dependencies that may cause conflicts
                install_with_mirror_fallback "$package" "--no-deps" || {
                    print_warning "$package installation failed with --no-deps, trying with dependencies..."
                    install_with_mirror_fallback "$package" || {
                        print_warning "$package installation failed, skipping..."
                    }
                }
            fi
        fi
    done
    
    # DeepSpeed (special handling due to CUDA requirements)
    if check_package_installed "deepspeed" "0.15.4"; then
        print_info "✓ deepspeed==0.15.4 is already installed"
    else
        print_info "Installing DeepSpeed..."
        # Check if CUDA_HOME is properly set
        if [ -n "$CUDA_HOME" ] && [ -f "$CUDA_HOME/bin/nvcc" ]; then
            print_info "Using CUDA_HOME: $CUDA_HOME"
            CUDA_HOME="$CUDA_HOME" install_with_mirror_fallback "deepspeed==0.15.4" || {
                print_warning "DeepSpeed installation failed, trying with no-build-isolation..."
                CUDA_HOME="$CUDA_HOME" install_with_mirror_fallback "deepspeed==0.15.4" "--no-build-isolation" || {
                    print_warning "DeepSpeed installation failed, skipping..."
                }
            }
        else
            print_warning "CUDA_HOME not properly configured, skipping DeepSpeed installation"
            print_info "To install DeepSpeed manually:"
            echo "  export CUDA_HOME=/path/to/cuda"
            echo "  pip install deepspeed==0.15.4"
        fi
    fi
    
    # Flash Attention (special handling)
    if check_package_installed "flash-attn" "2.7.4.post1"; then
        print_info "✓ flash-attn==2.7.4.post1 is already installed"
    else
        print_info "Installing Flash Attention..."
        
        # Check if CUDA_HOME is properly set
        if [ -n "$CUDA_HOME" ] && [ -f "$CUDA_HOME/bin/nvcc" ]; then
            print_info "Using CUDA_HOME: $CUDA_HOME"
            
            # Detect Python version for the correct wheel
            PYTHON_VERSION=$(python3 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
            print_info "Detected Python version tag: $PYTHON_VERSION"
            
            # Important notice before starting the installation
            echo ""
            print_info "⚠️  IMPORTANT: Flash Attention installation can be slow when building from source."
            print_info "To speed up installation, you can download the pre-built wheel file:"
            echo ""
            echo "  1. Download the wheel file:"
            echo "     wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
            echo ""
            echo "  2. Install from the downloaded wheel:"
            echo "     pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
            echo ""
            print_info "Press Ctrl+C now if you want to use the faster method above."
            print_info "Otherwise, the installation will continue automatically in 5 seconds..."
            sleep 5
            echo ""
            
            # Try multiple strategies to install Flash Attention
            print_info "Attempting Flash Attention installation from source (this may take several hours)..."
            
            # Try installing Flash Attention with mirror fallback
            if install_with_mirror_fallback "flash-attn==2.7.4.post1" "--no-build-isolation"; then
                print_success "Flash Attention installed successfully"
            else
                # All attempts failed
                print_warning "Flash Attention installation failed after multiple attempts"
                print_warning "This is optional and the project can run without it (with reduced performance)"
                print_info "You can try installing it manually later with:"
                echo "  Method 1 (Recommended - Using pre-built wheel):"
                echo "    wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
                echo "    pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
                echo ""
                echo "  Method 2 (Build from source):"
                echo "    export CUDA_HOME=$CUDA_HOME"
                echo "    pip install flash-attn==2.7.4.post1 --no-build-isolation"
            fi
        else
            print_warning "CUDA_HOME not properly configured, skipping Flash Attention installation"
            print_info "To install Flash Attention manually:"
            echo "  export CUDA_HOME=/path/to/cuda"
            echo "  pip install flash-attn==2.7.4.post1 --no-build-isolation"
        fi
    fi
}

# Install current package
install_package() {
    print_info "Installing rllava package..."
    install_with_mirror_fallback "-e ." "--no-deps" || {
        print_warning "Failed to install rllava package, continuing..."
    }
}

# Verify installation
verify_installation() {
    print_info "Verifying installation..."
    
    # Set CUDA environment for verification
    if [ -n "$CUDA_HOME" ]; then
        export CUDA_HOME="$CUDA_HOME"
        export PATH="$CUDA_HOME/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
    fi
    
    python3 -c "
import torch
import transformers
import accelerate
import peft
import trl
import datasets
import omegaconf
import tensordict
import gradio
import fastapi
import uvicorn
import pydantic
import numpy
import pandas
import PIL
import cv2
import av
import decord
import matplotlib
import seaborn
import sklearn
import scipy
import wandb
import tensorboard
import jupyterlab
import IPython
import notebook

print('✅ Core packages imported successfully')
print(f'PyTorch version: {torch.__version__}')
print(f'Transformers version: {transformers.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'Current GPU: {torch.cuda.get_device_name(0)}')
"
    
    print_success "Installation verification completed!"
}

# Main function
main() {
    print_info "RLLaVA installation script started"
    
    # Check environment
    check_python_version
    check_cuda
    
    # Upgrade pip
    print_info "Upgrading pip..."
    install_with_mirror_fallback "--upgrade pip" || {
        print_warning "Failed to upgrade pip, continuing..."
    }
    
    print_info "Starting smart installation with duplicate checking..."
    print_warning "Note: The script will skip packages that are already installed with correct versions"
    
    # Step-by-step installation
    install_basic_deps
    install_compile_deps
    install_optional_packages
    install_package
    
    # Verification
    verify_installation
    
    print_success "🎉 RLLaVA installation completed!"
    print_info "If some optional packages failed to install, you can install them separately later:"
    echo "  pip install flash-attn==2.7.4.post1 --no-build-isolation"
    echo "  pip install xformers==0.0.29.post2"
    echo "  pip install bitsandbytes==0.45.3"
    echo "  pip install vllm==0.8.5.post1"
    echo "  pip install sglang==0.4.6.post1"
}

# Run main function
main "$@"