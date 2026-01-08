# Cross-Coder Technical Guide for Model Diffing

## Overview

Cross-Coders are a specialized tool for comparing two related models (e.g., Base vs Chat) by learning a shared sparse representation that captures both common and model-specific features.

---

## What are Cross-Coders?

### Standard SAE vs Cross-Coder

| Feature | Standard SAE | Cross-Coder |
|---------|--------------|-------------|
| **Input** | Single model's activations | Two models' activations |
| **Output** | Sparse features for one model | Shared + model-specific features |
| **Use case** | Feature discovery | Model comparison / diffing |
| **Key benefit** | Interpretability | Isolate training-induced changes |

### Why Cross-Coders for This Project?

1. **Isolate RLHF Effects**: User modeling is likely added during RLHF
2. **Cleaner Signal**: Features unique to Chat model are our targets
3. **Direct Comparison**: Same prompt, two models, identify differences

---

## Cross-Coder Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CROSS-CODER ARCHITECTURE                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Base Model Activations    Chat Model Activations           │
│         x_base                    x_chat                    │
│            │                         │                       │
│            ▼                         ▼                       │
│  ┌──────────────────────────────────────────────────┐       │
│  │              SHARED ENCODER                       │       │
│  │                                                   │       │
│  │    f_shared = Encoder(concat(x_base, x_chat))    │       │
│  │                                                   │       │
│  └──────────────────────────────────────────────────┘       │
│                         │                                    │
│                         ▼                                    │
│  ┌──────────────────────────────────────────────────┐       │
│  │              SPARSE FEATURES                      │       │
│  │                                                   │       │
│  │  • Common features (active in both)              │       │
│  │  • Base-specific features                        │       │
│  │  • Chat-specific features  ← OUR TARGET          │       │
│  │                                                   │       │
│  └──────────────────────────────────────────────────┘       │
│                         │                                    │
│                         ▼                                    │
│  ┌──────────────────────────────────────────────────┐       │
│  │              DECODERS                             │       │
│  │                                                   │       │
│  │    x_base' = Decoder_base(f)                     │       │
│  │    x_chat' = Decoder_chat(f)                     │       │
│  │                                                   │       │
│  └──────────────────────────────────────────────────┘       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Feature Categories

| Category | Description | Relevance |
|----------|-------------|-----------|
| **Common** | Active in both Base and Chat | General language features |
| **Base-only** | Active only in Base | Rarely useful for our task |
| **Chat-only** | Active only in Chat | **Primary targets** - RLHF-added |

---

## Loading Gemma Scope Cross-Coders

### Prerequisites

```bash
pip install sae_lens transformer_lens torch transformers
```

### Loading Cross-Coder

```python
from sae_lens import SAE

# Load Cross-Coder for Base vs Chat comparison
# Note: Exact release name may vary - check Gemma Scope 2 documentation
cross_coder = SAE.from_pretrained(
    release="gemma-scope-9b-crosscoder",  # Example name
    sae_id="layer_24/width_65k",
    device="cuda"
)
```

### Loading Both Models

```python
from transformer_lens import HookedTransformer

# Load Base model
base_model = HookedTransformer.from_pretrained(
    "google/gemma-2-9b",
    torch_dtype=torch.bfloat16,
    device="cuda"
)

# Load Chat model
chat_model = HookedTransformer.from_pretrained(
    "google/gemma-2-9b-it",
    torch_dtype=torch.bfloat16,
    device="cuda"
)
```

---

## Usage Patterns

### Basic Feature Extraction

```python
def get_cross_coder_features(base_model, chat_model, cross_coder, prompt, layer=24):
    """
    Extract Cross-Coder features for a prompt run through both models.
    """
    tokens = base_model.to_tokens(prompt)

    # Get activations from both models
    _, base_cache = base_model.run_with_cache(tokens)
    _, chat_cache = chat_model.run_with_cache(tokens)

    # Get residual stream at target layer
    base_act = base_cache[f"blocks.{layer}.hook_resid_post"]
    chat_act = chat_cache[f"blocks.{layer}.hook_resid_post"]

    # Encode through Cross-Coder
    # The exact API depends on the Cross-Coder implementation
    features = cross_coder.encode(base_act, chat_act)

    return features
```

### Identifying Chat-Specific Features

```python
def find_chat_specific_features(features_base, features_chat, threshold=0.1):
    """
    Find features that are significantly more active in Chat vs Base.
    """
    # Compute activation difference
    diff = features_chat - features_base

    # Find features with large positive difference (Chat > Base)
    chat_specific = (diff > threshold).nonzero()

    return chat_specific, diff
```

### Sycophancy-Correlated Features

```python
def find_sycophancy_features(cross_coder, base_model, chat_model,
                              novice_prompts, expert_prompts):
    """
    Find Chat-specific features that correlate with expertise framing.
    """
    results = []

    for novice, expert in zip(novice_prompts, expert_prompts):
        # Get features for both framings
        novice_feats = get_cross_coder_features(
            base_model, chat_model, cross_coder, novice
        )
        expert_feats = get_cross_coder_features(
            base_model, chat_model, cross_coder, expert
        )

        # Compute differential
        diff = novice_feats - expert_feats
        results.append(diff)

    # Average across prompts
    mean_diff = torch.stack(results).mean(dim=0)

    # Top features by differential magnitude
    top_features = mean_diff.abs().topk(k=20)

    return top_features
```

---

## Intervention with Cross-Coder Features

### Ablating Chat-Specific Features

```python
def ablate_chat_features(chat_model, cross_coder, feature_indices, prompt):
    """
    Ablate specific Chat-only features during inference.
    """
    def ablation_hook(activation, hook):
        # Encode to Cross-Coder features
        features = cross_coder.encode_chat(activation)

        # Zero out target features
        features[:, :, feature_indices] = 0

        # Decode back
        return cross_coder.decode_chat(features)

    # Add hook and run
    with chat_model.hooks([(f"blocks.{layer}.hook_resid_post", ablation_hook)]):
        output = chat_model.generate(prompt, max_new_tokens=200)

    return output
```

### Steering with Chat Features

```python
def steer_toward_base(chat_model, cross_coder, steering_strength, prompt):
    """
    Steer Chat model toward Base-like behavior by reducing Chat-specific features.
    """
    def steering_hook(activation, hook):
        features = cross_coder.encode_chat(activation)

        # Identify Chat-specific features (those with low Base activation)
        chat_specific_mask = cross_coder.get_chat_specific_mask()

        # Reduce their activation
        features[:, :, chat_specific_mask] *= (1 - steering_strength)

        return cross_coder.decode_chat(features)

    with chat_model.hooks([(f"blocks.{layer}.hook_resid_post", steering_hook)]):
        output = chat_model.generate(prompt, max_new_tokens=200)

    return output
```

---

## Workflow for This Project

### Step 1: Confirm Behavioral Difference

```python
# Run same prompt through both models
prompt = "I'm new to programming. Is this code correct? x = 1/0"

base_response = base_model.generate(prompt)
chat_response = chat_model.generate(prompt)

# Base should be more direct
# Chat should be more gentle/sycophantic
print("Base:", base_response)
print("Chat:", chat_response)
```

### Step 2: Extract Cross-Coder Features

```python
# For expertise pairs
for novice, expert in expertise_pairs:
    novice_features = get_cross_coder_features(base_model, chat_model, cross_coder, novice)
    expert_features = get_cross_coder_features(base_model, chat_model, cross_coder, expert)

    # Store for analysis
    store_features(novice_features, expert_features)
```

### Step 3: Identify Target Features

```python
# Find features that are:
# 1. Chat-specific (added by RLHF)
# 2. Correlate with expertise framing

target_features = find_sycophancy_features(
    cross_coder, base_model, chat_model,
    novice_prompts, expert_prompts
)
```

### Step 4: Ablation Experiments

```python
# Ablate target features and measure sycophancy reduction
for feature_idx in target_features:
    ablated_response = ablate_chat_features(
        chat_model, cross_coder, [feature_idx], novice_prompt
    )

    # Measure sycophancy
    score = compute_sycophancy_score(ablated_response)
    print(f"Feature {feature_idx}: Sycophancy score = {score}")
```

---

## Comparison: SAE-Only vs Cross-Coder

| Aspect | SAE-Only (Simple Track) | Cross-Coder (Advanced Track) |
|--------|------------------------|------------------------------|
| **Setup complexity** | Low | Medium-High |
| **Compute** | 1 model | 2 models |
| **Feature origin** | Unknown | Known (Common/Base/Chat) |
| **Signal clarity** | May find non-RLHF features | Directly isolates RLHF features |
| **Documentation** | Good | Limited |
| **Recommended for** | Initial exploration | Confirmatory analysis |

---

## Troubleshooting

### Common Issues

**Issue**: OOM when loading both models
```python
# Solution: Load models sequentially, clear cache
base_response = get_response(base_model, prompt)
del base_model
torch.cuda.empty_cache()

chat_response = get_response(chat_model, prompt)
```

**Issue**: Cross-Coder features sparse/unclear
```python
# Solution: Look at multiple layers
for layer in [15, 20, 25, 30]:
    features = get_cross_coder_features(..., layer=layer)
    print(f"Layer {layer}: {(features > 0.1).sum()} active features")
```

**Issue**: No clear Chat-specific features
```python
# Solution: Lower threshold or look at feature combinations
chat_specific = find_chat_specific_features(features_base, features_chat, threshold=0.05)
```

---

## Resources

### Documentation
- [Gemma Scope 2 Announcement](https://www.alignmentforum.org/posts/YQro5LyYjDzZrBCdb/announcing-gemma-scope-2)
- [Cross-Coder Paper](https://arxiv.org/abs/...) (if available)
- [What We Learned Trying to Diff Base and Chat Models](https://www.lesswrong.com/posts/...)

### Code Examples
- SAELens documentation
- Gemma Scope 2 tutorial notebooks

### Community
- Alignment Forum discussions
- MATS Slack (if applicable)

---

*Guide created: January 2026*
