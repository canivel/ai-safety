#!/bin/bash
# Setup script for RunPod - sets environment and validates
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN environment variable before running}"
export HF_HOME=/workspace/.cache/huggingface
mkdir -p /workspace/.cache/huggingface

echo "=== Environment ==="
echo "HF_HOME=$HF_HOME"
echo "HF_TOKEN set: $([ -n "$HF_TOKEN" ] && echo YES || echo NO)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "VRAM: $(python3 -c 'import torch; print(f"{torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")')"

echo ""
echo "=== Data validation ==="
cd /workspace/experiments
python3 -c "
import sys; sys.path.insert(0, '.')
from shared.load_data import load_ethnicity_names, load_questions, VALID_COMPARISONS
eth = load_ethnicity_names()
qs = load_questions()
print(f'Groups: {list(eth.keys())}')
print(f'Questions: {len(qs)}')
print(f'Comparisons: {VALID_COMPARISONS}')
for g in eth:
    if g != 'ambiguous':
        print(f'  {g}: {len(eth[g][\"names\"])} names')
print('SETUP OK')
"
