#!/bin/bash
# Run remaining gemma12b jobs only (jobs 21-25)
set -e

export HF_HOME=/workspace/.cache/huggingface
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN environment variable before running}"
cd /workspace/experiments

echo "=========================================="
echo "  Gemma 12B — Remaining 5 Jobs"
echo "  $(date)"
echo "=========================================="

# Clean incomplete model downloads
echo "Cleaning incomplete downloads..."
find /workspace/.cache/huggingface -name "*.incomplete" -delete 2>/dev/null
echo "  Done."

# Pre-download the model before running extraction
echo ""
echo "Pre-downloading gemma-3-12b-it..."
python3 << 'PYEOF'
from huggingface_hub import snapshot_download
import os
path = snapshot_download("google/gemma-3-12b-it", token=os.environ.get("HF_TOKEN"))
print(f"  Downloaded to: {path}")
PYEOF
echo "  Model download complete."

COMPARISONS="white_vs_black white_vs_hispanic white_vs_asian white_vs_native_american white_vs_pacific_islander"
COUNT=0
TOTAL=5

for comp in $COMPARISONS; do
    COUNT=$((COUNT + 1))
    echo ""
    echo "=========================================="
    echo "  JOB $COUNT/$TOTAL: gemma12b / $comp"
    echo "  $(date)"
    echo "=========================================="

    python3 ethnicity/extract_hidden_states_eth.py gemma12b "$comp"

    echo ""
    echo "  JOB $COUNT/$TOTAL COMPLETE: gemma12b / $comp"
    echo "  $(date)"
    echo "=========================================="
done

echo ""
echo "=========================================="
echo "  ALL GEMMA12B JOBS COMPLETE"
echo "  $(date)"
echo "=========================================="

echo ""
echo "All output files:"
find results/ethnicity_probing -type f \( -name "*.npz" -o -name "*.json" \) | sort
echo ""
echo "Total: $(find results/ethnicity_probing -type f \( -name '*.npz' -o -name '*.json' \) | wc -l) files"
