#!/bin/bash
# =============================================================================
# RunPod Setup Script for Gemma 3 User Modeling Experiments
# =============================================================================
# Usage:
#   1. Create a RunPod pod with:
#      - GPU: 1x H100 80GB SXM (or H200)
#      - Template: RunPod PyTorch 2.x (CUDA 12.x)
#      - Container Disk: 20GB
#      - Volume Disk: 100GB+ (model weights cache)
#   2. SSH into the pod or open a terminal
#   3. Run: bash runpod_setup.sh
# =============================================================================

set -e

echo "============================================"
echo "  RunPod Setup: Gemma 3 User Modeling"
echo "============================================"

# --- 1. Check GPU ---
echo ""
echo "[1/5] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# --- 2. Set HuggingFace Token ---
echo "[2/5] Setting up HuggingFace authentication..."
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN environment variable not set."
    echo ""
    echo "Gemma 3 models are gated. You need a HuggingFace token."
    echo "Get one at: https://huggingface.co/settings/tokens"
    echo "Then accept the Gemma license at: https://huggingface.co/google/gemma-3-1b-it"
    echo ""
    echo "Set it with:"
    echo "  export HF_TOKEN=hf_your_token_here"
    echo "  bash runpod_setup.sh"
    echo ""
    echo "Or add it to RunPod pod environment variables."
    exit 1
fi
echo "HF_TOKEN is set."

# --- 3. Install uv and Dependencies ---
echo ""
echo "[3/5] Installing uv and Python dependencies..."

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Now authenticate with HuggingFace (uv is available for fallback)
huggingface-cli login --token "$HF_TOKEN" 2>/dev/null || (uv pip install --system huggingface_hub && huggingface-cli login --token "$HF_TOKEN")
echo "Authenticated with HuggingFace."

uv pip install --system \
    "transformers>=4.50.0" \
    "accelerate>=0.30.0" \
    torch \
    "scikit-learn>=1.3.0" \
    "matplotlib>=3.7.0" \
    "numpy>=1.24.0" \
    "pandas>=2.0.0" \
    "tqdm>=4.66.0" \
    "huggingface_hub>=0.20.0" \
    "jupyterlab>=4.0.0" \
    "ipywidgets>=8.0.0"

echo "Dependencies installed via uv."

# --- 4. Clone Repository ---
echo ""
echo "[4/5] Setting up workspace..."
WORKSPACE="/workspace"
REPO_DIR="$WORKSPACE/ai-safety"

if [ -d "$REPO_DIR" ]; then
    echo "Repository already exists at $REPO_DIR. Pulling latest..."
    cd "$REPO_DIR" && git pull
else
    echo "Cloning repository..."
    cd "$WORKSPACE"
    git clone https://github.com/canivel/ai-safety.git
fi

NOTEBOOK_DIR="$REPO_DIR/research-idea-6/experiments/notebooks"
RESULTS_DIR="$REPO_DIR/research-idea-6/experiments/results/gemma3_gender_detection"
mkdir -p "$RESULTS_DIR"

echo "Workspace ready."
echo "  Notebook: $NOTEBOOK_DIR/user_modeling_gender_detection_gemma3.ipynb"
echo "  Results:  $RESULTS_DIR/"

# --- 5. Pre-download Models (Optional) ---
echo ""
echo "[5/5] Pre-downloading Gemma 3 models..."
echo "This downloads model weights to cache so notebook runs are faster."
echo ""

python3 -c "
from huggingface_hub import snapshot_download
import os

models = [
    'google/gemma-3-1b-it',
    'google/gemma-3-4b-it',
    'google/gemma-3-12b-it',
    'google/gemma-3-27b-it',
]

for m in models:
    print(f'Downloading {m}...')
    try:
        snapshot_download(m, ignore_patterns=['*.gguf', '*.bin'])
        print(f'  Done: {m}')
    except Exception as e:
        print(f'  Warning: {m} failed: {e}')
        print(f'  (Will download on first use in notebook)')
print('Model download complete.')
"

# --- Done ---
echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Open JupyterLab (RunPod provides a URL)"
echo "  2. Navigate to: research-idea-6/experiments/notebooks/"
echo "  3. Open: user_modeling_gender_detection_gemma3.ipynb"
echo "  4. Set MODEL_ID in the config cell (start with 'google/gemma-3-1b-it')"
echo "  5. Run All Cells"
echo "  6. Change MODEL_ID to next model and repeat"
echo ""
echo "Estimated total runtime: ~2-3 hours for all 4 models"
echo "  1B: ~10 min | 4B: ~20 min | 12B: ~40 min | 27B: ~90 min"
echo ""
