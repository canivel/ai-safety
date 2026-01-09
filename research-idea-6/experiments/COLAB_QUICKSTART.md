# Google Colab Quick Start Guide

## Step 1: Open the Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

1. Go to [Google Colab](https://colab.research.google.com/)
2. File → Upload Notebook
3. Upload `notebooks/01_model_diffing_setup.ipynb`

## Step 2: Enable GPU

1. Runtime → Change runtime type
2. Hardware accelerator: **GPU** (T4 is sufficient)
3. Save

## Step 3: Get HuggingFace Token

1. Go to https://huggingface.co/google/gemma-2-2b
2. Click "Agree and access repository"
3. Go to https://huggingface.co/settings/tokens
4. Create new token with "Read" access
5. Copy the token

## Step 4: Run the Notebook

The notebook will:
1. Install dependencies (~2 min)
2. Load models (~5 min first time, cached after)
3. Run analysis (~10 min)

## Expected Runtime

| Step | Time |
|------|------|
| Install dependencies | 2-3 min |
| Load Base model | 3-5 min |
| Load Chat model | 3-5 min |
| Run analysis | 5-10 min |
| **Total** | **15-25 min** |

## Memory Usage

With Colab's free T4 GPU (15GB):
- Gemma 2 2B Base: ~5GB
- Gemma 2 2B Chat: ~5GB
- Cross-Coder: ~2GB
- **Total**: ~12GB ✓

## Troubleshooting

### "Your session crashed"
- This usually means OOM
- Restart runtime and try again
- Make sure you're using GPU, not CPU

### "You must be authenticated"
- Run the login cell again
- Check your token has read access

### "Rate limit exceeded"
- HuggingFace has download limits
- Wait a few minutes and try again
- Or use `cache_dir` to cache models locally

## Saving Your Work

Results are saved to `/content/` in Colab. To keep them:

```python
# Download results
from google.colab import files
files.download('results/day1_summary.json')
```

Or save to Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')

# Copy results
!cp -r results /content/drive/MyDrive/sycophancy_research/
```

## Next Steps

After completing Day 1:

1. **Day 2**: Run sycophancy trigger analysis
2. **Day 3**: Causal intervention experiments
3. **Day 4**: Circuit tracing with attention heads
