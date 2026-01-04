#!/bin/bash
# Setup script for TFLite export environment
# Run this on an amd64 Linux machine (e.g., RunPod, AWS, GCP)
#
# Usage:
#   chmod +x scripts/setup_tflite_export.sh
#   ./scripts/setup_tflite_export.sh
#
# Then export:
#   python -m yolo.cli export --checkpoint model.ckpt --format tflite

set -e

echo "=============================================="
echo "🚀 Setting up TFLite Export Environment"
echo "=============================================="

# Check architecture
ARCH=$(uname -m)
echo "Architecture: $ARCH"
if [[ "$ARCH" != "x86_64" ]]; then
    echo "⚠️  Warning: This script is optimized for x86_64 (amd64)"
    echo "   TFLite export may have issues on ARM architecture"
fi

# Install system dependencies
echo ""
echo "📦 Installing system dependencies..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y python3-pip python3-venv
fi

# Create virtual environment
echo ""
echo "🐍 Creating Python virtual environment..."
python3 -m venv .venv-tflite
source .venv-tflite/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CPU version for export)
echo ""
echo "🔥 Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install core dependencies
echo ""
echo "📚 Installing core dependencies..."
pip install \
    lightning>=2.0 \
    jsonargparse[signatures]>=4.27.7 \
    omegaconf \
    einops \
    numpy \
    Pillow \
    loguru \
    rich

# Install ONNX export dependencies
echo ""
echo "📦 Installing ONNX dependencies..."
pip install \
    onnx>=1.14.0 \
    onnxsim>=0.4.0

# Install TFLite export dependencies
echo ""
echo "🔧 Installing TFLite dependencies..."
pip install \
    tensorflow>=2.15.0 \
    onnx2tf>=1.25.0

# Install the package
echo ""
echo "📦 Installing YOLO package..."
pip install -e .

# Verify installation
echo ""
echo "=============================================="
echo "🔍 Verifying installation..."
echo "=============================================="

python -c "
import sys
print(f'Python: {sys.version}')

try:
    import torch
    print(f'✅ torch: {torch.__version__}')
except Exception as e:
    print(f'❌ torch: {e}')

try:
    import onnx
    print(f'✅ onnx: {onnx.__version__}')
except Exception as e:
    print(f'❌ onnx: {e}')

try:
    import tensorflow as tf
    print(f'✅ tensorflow: {tf.__version__}')
except Exception as e:
    print(f'❌ tensorflow: {e}')

try:
    import onnx2tf
    print(f'✅ onnx2tf: OK')
except Exception as e:
    print(f'❌ onnx2tf: {e}')

print('')
print('To activate this environment later:')
print('  source .venv-tflite/bin/activate')
print('')
print('To export to TFLite:')
print('  python -m yolo.cli export --checkpoint model.ckpt --format tflite')
print('  python -m yolo.cli export --checkpoint model.ckpt --format tflite --quantization fp16')
print('  python -m yolo.cli export --checkpoint model.ckpt --format tflite --quantization int8 --calibration-images ./images/')
"

echo ""
echo "=============================================="
echo "✅ Setup complete!"
echo "=============================================="
