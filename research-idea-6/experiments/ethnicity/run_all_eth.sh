#!/bin/bash
# Batch runner for all ethnicity extraction jobs on RunPod
# Self-contained: installs deps, fixes line endings, validates, then runs.
#
# Upload entire experiments/ dir to /workspace/experiments/ then:
#   nohup bash /workspace/experiments/ethnicity/run_all_eth.sh > /workspace/eth_run.log 2>&1 &
#   tail -f /workspace/eth_run.log
set -e

export HF_HOME=/workspace/.cache/huggingface
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN environment variable before running}"
cd /workspace/experiments

echo "=========================================="
echo "  Ethnicity Extraction — Full Batch Run"
echo "  $(date)"
echo "=========================================="

# --- Step 0: Fix Windows line endings on ALL files ---
echo ""
echo "[SETUP] Fixing line endings..."
find /workspace/experiments -name "*.py" -o -name "*.json" -o -name "*.sh" | xargs sed -i 's/\r//g'
echo "  Done."

# --- Step 1: Install dependencies ---
echo ""
echo "[SETUP] Installing dependencies..."
pip install --break-system-packages -q accelerate transformers torch scikit-learn tqdm huggingface_hub 2>&1 | tail -3
echo "  Done."

# --- Step 2: Validate setup ---
echo ""
echo "[SETUP] Validating data..."
python3 << 'PYEOF'
import sys; sys.path.insert(0, '.')
from shared.load_data import load_ethnicity_names, load_questions, VALID_COMPARISONS
eth = load_ethnicity_names()
qs = load_questions()
print(f'  Groups: {list(eth.keys())}')
print(f'  Questions: {len(qs)}')
print(f'  Comparisons: {VALID_COMPARISONS}')
for g in eth:
    if g != 'ambiguous':
        print(f'    {g}: {len(eth[g]["names"])} names')

import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  GPU: {torch.cuda.get_device_name(0)}')
print(f'  CUDA available: {torch.cuda.is_available()}')

import accelerate
print(f'  accelerate: {accelerate.__version__}')
print('  SETUP OK')
PYEOF

# --- Step 3: Run all extraction jobs ---
# Models ordered by size (smallest first for faster iteration)
MODELS="gemma1b gemma4b qwen7b mistral7b gemma12b"
COMPARISONS="white_vs_black white_vs_hispanic white_vs_asian white_vs_native_american white_vs_pacific_islander"

TOTAL=25
COUNT=0

for model in $MODELS; do
    for comp in $COMPARISONS; do
        COUNT=$((COUNT + 1))
        echo ""
        echo "=========================================="
        echo "  JOB $COUNT/$TOTAL: $model / $comp"
        echo "  $(date)"
        echo "=========================================="

        python3 ethnicity/extract_hidden_states_eth.py "$model" "$comp"

        echo ""
        echo "  JOB $COUNT/$TOTAL COMPLETE: $model / $comp"
        echo "  $(date)"
        echo "=========================================="
    done
done

echo ""
echo "=========================================="
echo "  ALL $TOTAL JOBS COMPLETE"
echo "  $(date)"
echo "=========================================="

# List output files
echo ""
echo "Output files:"
find results/ethnicity_probing -type f \( -name "*.npz" -o -name "*.json" \) | sort
echo ""
echo "Total files: $(find results/ethnicity_probing -type f \( -name "*.npz" -o -name "*.json" \) | wc -l)"
