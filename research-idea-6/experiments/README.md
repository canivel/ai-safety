# Model Diffing Experiments for Sycophancy Research

This directory contains the experimental setup for Research Idea 6: "From Inference to Pandering: User Modeling and Sycophancy Circuits."

## Quick Start

### Option 1: Google Colab (Recommended for Quick Start)

1. Upload `notebooks/01_model_diffing_setup.ipynb` to Google Colab
2. Run the first cell to install dependencies
3. Follow the notebook instructions

### Option 2: Local Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install dictionary_learning (for Cross-Coder support)
pip install dictionary-learning
# or: git clone https://github.com/saprmarks/dictionary_learning.git

# Run Jupyter
jupyter lab notebooks/
```

## Directory Structure

```
experiments/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── docs/
│   └── observable_signal_experiment.md  # Output-only detection experiment
├── notebooks/
│   └── 01_model_diffing_setup.ipynb  # Day 1 notebook
├── data/
│   └── user_persona_prompts.json     # Prompt dataset
│   └── observable_signal_minimal_pairs.json  # Minimal-pair prompts
├── src/
│   ├── __init__.py           # Package init
│   ├── model_diffing.py      # Core stereo model runner
│   ├── cross_coder.py        # Cross-Coder wrapper
│   ├── analysis.py           # Latent analysis utilities
│   ├── observable_signals.py # Output-only scoring utilities
│   └── visualization.py      # Plotting and dashboards
├── results/                  # Output directory (created on run)
└── figures/                  # Saved figures (created on run)
```

## Key Components

### 1. Stereo Model Runner (`src/model_diffing.py`)

Runs both Base (gemma-2-2b) and Chat (gemma-2-2b-it) models side-by-side:

```python
from src.model_diffing import StereoModelRunner

runner = StereoModelRunner(device="cuda", target_layers=[13])
runner.load_models(hf_token="your_token")

base_cache, chat_cache = runner.run_stereo("Your prompt here")
```

### 2. Cross-Coder Wrapper (`src/cross_coder.py`)

Loads and uses Cross-Coders for feature extraction:

```python
from src.cross_coder import CrossCoderWrapper

cc = CrossCoderWrapper.from_pretrained("Butanium/gemma-2-2b-crosscoder-l13")
latents = cc.encode(concatenated_activations)
chat_specific = cc.get_chat_specific_latents(activations)
```

### 3. Latent Analyzer (`src/analysis.py`)

Finds User Model and Sycophancy features:

```python
from src.analysis import LatentAnalyzer, SycophancyDetector

analyzer = LatentAnalyzer(cross_coder)
user_model_features = analyzer.find_user_model_features(prompt_pairs, pipeline)
```

## Models Used

| Model | HuggingFace ID | Purpose |
|-------|---------------|---------|
| Base | `google/gemma-2-2b` | Document completion (no RLHF) |
| Chat | `google/gemma-2-2b-it` | Helpful assistant (RLHF-trained) |
| Cross-Coder | `Butanium/gemma-2-2b-crosscoder-l13` | Feature extraction |

## Required Authentication

Gemma models require accepting the license:

1. Go to https://huggingface.co/google/gemma-2-2b
2. Accept the license agreement
3. Create a token at https://huggingface.co/settings/tokens
4. Use `huggingface-cli login` or pass token to functions

## Memory Requirements

| Setup | GPU Memory | Notes |
|-------|------------|-------|
| Gemma 2 2B (fp16) | ~5GB | Per model |
| Both models loaded | ~10GB | For stereo analysis |
| + Cross-Coder | ~12GB | Full setup |
| Colab Free | 15GB T4 | Sufficient for 2B models |

## Experiment Workflow

### Day 1: Find User Model Features

1. Load both models (stereo setup)
2. Run User Persona prompts (Novice vs Expert)
3. Compare activations / Cross-Coder latents
4. Identify consistent "User is Novice" features

### Day 2: Sycophancy Analysis

1. Run Sycophancy trigger prompts (Biased vs Neutral)
2. Find features that correlate with sycophantic behavior
3. Cross-reference with User Model features

### Day 3: Causal Intervention

1. Clamp identified features
2. Measure response changes
3. Test if ablation reduces sycophancy

## Dataset: User Persona Prompts

Located at `data/user_persona_prompts.json`:

- **Expertise Level Pairs**: Novice vs Expert versions of same questions
- **Sycophancy Triggers**: Biased vs Neutral versions
- **Implicit Gender**: Gender cues without explicit statements
- **Implicit Age**: Age cues for response adaptation
- **Emotional State**: Distressed vs Calm framing

## Expected Outputs

After running the Day 1 notebook:

1. `results/day1_summary.json` - Analysis summary
2. `results/user_model_candidates.png` - Top candidate features
3. `results/feature_verification.png` - Verification results

## Troubleshooting

### "CUDA out of memory"
- Use fp16: `dtype=torch.float16`
- Reduce batch size
- Use Colab with GPU runtime

### "ModuleNotFoundError: dictionary_learning"
```bash
pip install dictionary-learning
# or
git clone https://github.com/saprmarks/dictionary_learning.git
```

### "Access denied" for Gemma models
- Accept license at https://huggingface.co/google/gemma-2-2b
- Ensure token has read access

## References

- [Gemma 2 Technical Report](https://arxiv.org/abs/2408.00118)
- [Cross-Coders for Model Diffing](https://github.com/saprmarks/dictionary_learning)
- [TransformerLens Documentation](https://transformerlensorg.github.io/TransformerLens/)
- [A Pragmatic Vision for Interpretability](https://neelnanda.io/pragmatic-interpretability)

---

*Created: January 2026*
*Research Idea 6: User Modeling & Sycophancy Circuits*
