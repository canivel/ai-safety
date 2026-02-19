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
├── shared/                   # Shared data files (gender + ethnicity)
│   ├── questions.json               # 200 questions used across all experiments
│   ├── gender_names.json            # Male/female/ambiguous name lists
│   ├── ethnicity_names.json         # EEOC ethnicity name lists (6 groups)
│   ├── model_registry.json          # Model configs
│   └── load_data.py                 # Data loader module
├── notebooks/                # Gender experiment scripts
│   ├── 01_model_diffing_setup.ipynb    # Day 1 notebook (early exploration)
│   ├── extract_hidden_states.py        # GPU: hidden state extraction for all models
│   ├── analyze_probing_v4.py           # CPU: 4-variant probing analysis
│   ├── run_probing_v3.py              # GPU: v3 probing (predecessor to v4)
│   ├── run_probing_v4.py              # GPU: v4 probing pipeline
│   ├── run_all_models.py              # GPU: multi-model extraction runner
│   ├── run_causal_mediation.py        # GPU: causal mediation experiment
│   ├── run_kl_strength_sweep.py       # GPU: KL strength sweep (dose-response)
│   ├── run_circuit_tracing.py         # GPU: attention head circuit tracing
│   └── run_sae_analysis.py           # GPU: SAE feature analysis with Gemma Scope 2
├── ethnicity/                # Ethnicity experiment scripts
│   ├── extract_hidden_states_eth.py   # GPU: ethnicity hidden state extraction
│   ├── analyze_probing_eth.py         # CPU: ethnicity probing analysis
│   ├── run_kl_strength_sweep_eth.py   # GPU: ethnicity causal mediation
│   └── run_circuit_tracing_eth.py     # GPU: ethnicity circuit tracing
├── data/
│   └── user_persona_prompts.json     # Prompt dataset
├── src/
│   ├── __init__.py           # Package init
│   ├── model_diffing.py      # Core stereo model runner
│   ├── cross_coder.py        # Cross-Coder wrapper
│   ├── analysis.py           # Latent analysis utilities
│   ├── observable_signals.py # Output-only scoring utilities
│   └── visualization.py      # Plotting and dashboards
├── results/
│   ├── gemma3_gender_detection/       # v1-v3 results (Gemma 3 family)
│   ├── cross_family_probing_v4/       # v4: 5 models, 200 questions (gender)
│   ├── ethnicity_probing/             # Ethnicity probing results
│   │   ├── white_vs_black/            # 5 models × probing + steering JSONs
│   │   ├── white_vs_hispanic/
│   │   ├── white_vs_asian/
│   │   ├── white_vs_native_american/
│   │   └── white_vs_pacific_islander/
│   ├── causal_mediation/              # Causal mediation + KL sweep results
│   ├── circuit_tracing/               # Attention head analysis results
│   └── sae_analysis/                  # SAE feature analysis results
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

### Phase 1: Hidden State Extraction (GPU)

Extract hidden states for all 5 models using `extract_hidden_states.py`:

```bash
python extract_hidden_states.py --model gemma1b
python extract_hidden_states.py --model gemma4b
python extract_hidden_states.py --model gemma12b
python extract_hidden_states.py --model qwen7b
python extract_hidden_states.py --model mistral7b
```

### Phase 2: Probing Analysis (CPU)

Run 4-variant probing with `analyze_probing_v4.py`:

```bash
python analyze_probing_v4.py all
```

Produces: last-token accuracy, question-only accuracy, held-out generalization, steering KL ratios, permutation null tests, and ambiguous name classification.

### Phase 3: Mechanistic Experiments (GPU)

```bash
# Causal mediation: KL strength sweep with dose-response curve
python run_kl_strength_sweep.py gemma4b

# Circuit tracing: attention head ablation study
python run_circuit_tracing.py gemma4b

# SAE analysis: Gemma Scope 2 feature decomposition
python run_sae_analysis.py
```

## Dataset: User Persona Prompts

Located at `data/user_persona_prompts.json`:

- **Expertise Level Pairs**: Novice vs Expert versions of same questions
- **Sycophancy Triggers**: Biased vs Neutral versions
- **Implicit Gender**: Gender cues without explicit statements
- **Implicit Age**: Age cues for response adaptation
- **Emotional State**: Distressed vs Calm framing

## Results Summary

### Gender Probing — Cross-Family v4 (5 models, 200 questions)

| Model | Last-Token | Q-Only | Held-Out | KL Ratio | CoT Signal |
|-------|-----------|--------|---------|---------|-----------|
| Gemma 3-1B | 88.3% | 99.8% | 100.0% | 1.11x | 0/16 |
| Gemma 3-4B | 96.8% | 100.0% | 100.0% | 5.23x | 0/16 |
| Gemma 3-12B | 100.0% | 100.0% | 100.0% | 5.31x | 0/16 |
| Qwen 2.5-7B | 99.8% | 100.0% | 100.0% | 1.18x | 0/16 |
| Mistral 7B | 99.5% | 100.0% | 98.8% | 1.81x | 0/16 |

Embedding-layer accuracy: **50.0% (chance)** for all models — confirming signal is from transformer processing.

### Gender Mechanistic Experiments (Gemma 3 4B)

- **Causal mediation**: 48.3% first-token KL reduction at strength=1.0; random direction control only 9.6% change vs 980%
- **Circuit tracing**: 20 heads (7.4% of 272) cause 21.5% accuracy drop. Three-phase circuit: L4-8 encoding → L14 propagation → L30 aggregation
- **SAE analysis**: 0/16,384 Gemma Scope 2 features show significant gender differential — gender encoded in superposition

### Ethnicity Probing — 5 models, 5 EEOC comparisons (25 experiments)

#### Last-Token Probing Accuracy

| Model | W vs Black | W vs Hispanic | W vs Asian | W vs Nat.Am. | W vs Pac.Isl. |
|-------|-----------|---------------|------------|-------------|---------------|
| Gemma 3-1B | 86.3% | 89.2% | 93.5% | 95.8% | 93.8% |
| Gemma 3-4B | 94.0% | 95.5% | 97.5% | 97.0% | 97.7% |
| Gemma 3-12B | 98.0% | 99.0% | 98.8% | 97.8% | 97.5% |
| Qwen 2.5-7B | 95.5% | 98.0% | 98.5% | 98.5% | 97.5% |
| Mistral 7B | 91.5% | 94.5% | 97.3% | 95.3% | 95.3% |

#### Question-Only Probing Accuracy

| Model | W vs Black | W vs Hispanic | W vs Asian | W vs Nat.Am. | W vs Pac.Isl. |
|-------|-----------|---------------|------------|-------------|---------------|
| Gemma 3-1B | 94.8% | 96.3% | 97.3% | 96.8% | 96.8% |
| Gemma 3-4B | 97.8% | 99.0% | 99.7% | 99.3% | 99.5% |
| Gemma 3-12B | 98.8% | 100.0% | 100.0% | 99.7% | 99.7% |
| Qwen 2.5-7B | 99.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Mistral 7B | 98.8% | 99.7% | 100.0% | 99.3% | 100.0% |

#### Held-Out Name Generalization

| Model | W vs Black | W vs Hispanic | W vs Asian | W vs Nat.Am. | W vs Pac.Isl. |
|-------|-----------|---------------|------------|-------------|---------------|
| Gemma 3-1B | 90.0% | 97.5% | 98.8% | 97.5% | 96.3% |
| Gemma 3-4B | 91.3% | 98.8% | 100.0% | 100.0% | 100.0% |
| Gemma 3-12B | 96.3% | 100.0% | 100.0% | 100.0% | 100.0% |
| Qwen 2.5-7B | 93.8% | 100.0% | 100.0% | 100.0% | 100.0% |
| Mistral 7B | 96.3% | 100.0% | 100.0% | 100.0% | 100.0% |

All embedding baselines: **50.0% (chance)**. All p-values: **0.0**.

### Ethnicity Experiment Workflow

```bash
cd experiments/ethnicity

# GPU: Extract hidden states (~20 min per model×comparison on A40)
python extract_hidden_states_eth.py gemma4b white_vs_black

# CPU: Run probing analysis
python analyze_probing_eth.py gemma4b white_vs_black
```

## Troubleshooting

### "CUDA out of memory"
- Use bf16: `dtype=torch.bfloat16` (default for Gemma 3)
- A40 (48GB) fits all models up to 12B
- 27B requires H100 80GB+

### "Access denied" for Gemma models
- Accept license at https://huggingface.co/google/gemma-3-4b-it
- Set `HF_TOKEN` environment variable

### Gemma 3 architecture notes
- **1B**: Text-only → `Gemma3ForCausalLM` + `AutoTokenizer`
- **4B/12B/27B**: Multimodal → `Gemma3ForConditionalGeneration` + `AutoProcessor`
- Hidden states: pass `output_hidden_states=True` in forward call, NOT via config

## References

- [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786)
- [Gemma Scope 2: SAEs for Gemma 3](https://huggingface.co/google/gemma-scope-2-4b-it)
- [What Kind of User Are You? (ICML 2025)](https://openreview.net/forum?id=si1XJoQeaO)
- [Demographic Probing Construct Validity (Tonneau et al.)](https://arxiv.org/abs/2601.18486)

---

*Created: January 2026*
*Gender experiments completed: February 2026*
*Ethnicity extension completed: February 2026*
*Research Idea 6: Implicit User Modeling — Gender & Ethnicity Mechanistic Evidence*
