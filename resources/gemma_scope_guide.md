# Gemma Scope 2 Technical Guide

## Overview

Gemma Scope 2 is the latest suite of interpretability tools from Google DeepMind, providing pre-trained Sparse Autoencoders (SAEs) and Transcoders for the Gemma 2 model family.

---

## What's Included

### Models Covered
| Model | Parameters | SAEs | Transcoders |
|-------|------------|------|-------------|
| Gemma 2 2B | 2B | ✓ All layers | ✓ All MLP layers |
| Gemma 2 9B | 9B | ✓ All layers | ✓ All MLP layers |
| Gemma 2 27B | 27B | ✓ All layers | ✓ All MLP layers |

### SAE Types
- **Residual Stream SAEs**: Decompose activations at residual stream
- **Attention Output SAEs**: Decompose attention layer outputs
- **MLP Output SAEs**: Decompose MLP layer outputs

### Transcoders
- Map MLP input → MLP output through sparse features
- Enable circuit tracing between layers
- JumpReLU activation for strict sparsity

---

## Installation & Loading

### Prerequisites
```bash
pip install sae_lens transformer_lens torch transformers
```

### Loading SAEs with SAELens

```python
from sae_lens import SAE

# Load SAE for specific layer
sae = SAE.from_pretrained(
    release="gemma-scope-2b-pt-res",  # or "gemma-scope-9b-pt-res"
    sae_id="layer_20/width_16k/average_l0_71",
    device="cuda"
)

# Available releases:
# - gemma-scope-2b-pt-res (residual stream)
# - gemma-scope-2b-pt-mlp (MLP output)
# - gemma-scope-9b-pt-res
# - gemma-scope-9b-pt-mlp
# - gemma-scope-27b-pt-res
# - gemma-scope-27b-pt-mlp
```

### Loading Transcoders

```python
from sae_lens import SAE

# Transcoders use similar interface
transcoder = SAE.from_pretrained(
    release="gemma-scope-9b-pt-transcoders",
    sae_id="layer_24/width_131k",
    device="cuda"
)
```

---

## Key Parameters

### SAE Widths
| Width | Features | Use Case |
|-------|----------|----------|
| 16k | 16,384 | Fast iteration, less granular |
| 65k | 65,536 | Balanced |
| 131k | 131,072 | Most granular, slower |
| 1M | 1,048,576 | Maximum granularity (select layers) |

### Sparsity (L0)
- **L0 ~70**: Good balance of sparsity and reconstruction
- **L0 ~140**: More features active, better reconstruction
- **L0 ~30**: Very sparse, may miss details

---

## Usage Patterns

### Basic Feature Extraction

```python
import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE

# Load model
model = HookedTransformer.from_pretrained("google/gemma-2-9b-it")

# Load SAE
sae = SAE.from_pretrained(
    release="gemma-scope-9b-pt-res",
    sae_id="layer_24/width_65k/average_l0_72",
    device="cuda"
)

# Get activations
prompt = "Let me solve this step by step..."
tokens = model.to_tokens(prompt)
_, cache = model.run_with_cache(tokens)

# Get residual stream at target layer
residual = cache["resid_post", 24]  # [batch, seq, d_model]

# Encode to sparse features
features = sae.encode(residual)  # [batch, seq, n_features]

# Get active features
active_features = (features > 0).nonzero()
```

### Attribution to Features

```python
def attribute_to_features(model, sae, tokens, target_token_idx, target_logit):
    """
    Compute attribution of output logit to SAE features.
    """
    # Forward pass with gradients
    model.zero_grad()
    logits = model(tokens)
    target = logits[0, target_token_idx, target_logit]
    target.backward()

    # Get gradient at SAE input
    residual = cache["resid_post", sae.cfg.hook_layer]
    residual.requires_grad_(True)

    # Encode and get feature gradients
    features = sae.encode(residual)
    feature_grads = torch.autograd.grad(target, features)[0]

    # Attribution = activation * gradient
    attribution = features * feature_grads
    return attribution
```

### Feature Ablation

```python
def ablate_feature(model, sae, feature_idx, tokens):
    """
    Ablate specific feature by zeroing its contribution.
    """
    def hook_fn(activation, hook):
        # Encode to features
        features = sae.encode(activation)

        # Zero target feature
        features[:, :, feature_idx] = 0

        # Decode back
        reconstructed = sae.decode(features)
        return reconstructed

    # Add hook at SAE input location
    hook_name = f"blocks.{sae.cfg.hook_layer}.hook_resid_post"

    with model.hooks([(hook_name, hook_fn)]):
        logits = model(tokens)

    return logits
```

### Feature Steering

```python
def steer_feature(model, sae, feature_idx, strength, tokens):
    """
    Steer by adding activation to specific feature.
    """
    def hook_fn(activation, hook):
        features = sae.encode(activation)

        # Add steering vector
        features[:, :, feature_idx] += strength

        reconstructed = sae.decode(features)
        return reconstructed

    hook_name = f"blocks.{sae.cfg.hook_layer}.hook_resid_post"

    with model.hooks([(hook_name, hook_fn)]):
        logits = model(tokens)

    return logits
```

---

## Transcoder-Specific Usage

### Circuit Tracing with Transcoders

```python
def trace_circuit(model, transcoder, tokens):
    """
    Trace computation through transcoder to find feature interactions.
    """
    _, cache = model.run_with_cache(tokens)

    # Get MLP input (transcoder input)
    mlp_in = cache["pre", transcoder.cfg.hook_layer]

    # Get sparse features
    features = transcoder.encode(mlp_in)

    # Decode to get contribution to output
    mlp_out_reconstructed = transcoder.decode(features)

    # Compare to actual MLP output
    mlp_out_actual = cache["post", transcoder.cfg.hook_layer]

    return features, mlp_out_reconstructed, mlp_out_actual
```

### Building Attribution Graphs

```python
def build_attribution_graph(model, transcoders, tokens, output_feature_idx):
    """
    Build graph of feature interactions across layers.
    """
    graph = {}

    for layer_idx, transcoder in enumerate(transcoders):
        # Get features at this layer
        features = transcoder.encode(cache[f"pre", layer_idx])

        # Compute attribution to downstream features
        # (simplified - actual implementation more complex)
        for feat_idx in range(features.shape[-1]):
            if features[0, -1, feat_idx] > 0:
                graph[(layer_idx, feat_idx)] = {
                    'activation': features[0, -1, feat_idx].item(),
                    'downstream': []  # filled by next layer analysis
                }

    return graph
```

---

## Feature Interpretation with Neuronpedia

### Looking Up Features

1. Go to [neuronpedia.org](https://neuronpedia.org)
2. Select model (e.g., "gemma-2-9b-it")
3. Select SAE (e.g., "layer_24/width_65k")
4. Enter feature index

### What to Look For

| Property | Interpretation |
|----------|----------------|
| **Max Activating Examples** | What contexts trigger this feature |
| **Top Tokens** | What the feature represents |
| **Logit Lens** | What output tokens it promotes |
| **Neuron Alignment** | Which neurons it corresponds to |

### Labeling Features

When labeling features for your research:

1. **Look at 10+ max activating examples**
2. **Identify common theme** (concept, syntax, task)
3. **Check for polysemanticity** (multiple meanings)
4. **Note activation position** (start, middle, end of text)
5. **Document confidence level** (high/medium/low)

---

## Compute Requirements

### Memory Estimates

| Model | SAE Width | Approx. VRAM |
|-------|-----------|--------------|
| Gemma 2 2B | 16k | ~8 GB |
| Gemma 2 2B | 65k | ~12 GB |
| Gemma 2 9B | 16k | ~24 GB |
| Gemma 2 9B | 65k | ~32 GB |
| Gemma 2 9B | 131k | ~40 GB |
| Gemma 2 27B | 65k | ~80 GB |

### Optimization Tips

1. **Use bf16 precision** for model and SAE
2. **Stream SAEs** rather than loading all at once
3. **Cache activations** to disk for repeated analysis
4. **Use 2B model** for initial experiments
5. **Subsample sequences** for long contexts

---

## Common Issues & Solutions

### Issue: OOM on 9B model
```python
# Solution: Use 2B or load in 8-bit
model = HookedTransformer.from_pretrained(
    "google/gemma-2-9b-it",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

### Issue: SAE version mismatch
```python
# Solution: Check SAELens version
pip install sae_lens --upgrade
```

### Issue: Slow feature analysis
```python
# Solution: Batch processing
features_batch = sae.encode(residual_batch)  # Process multiple at once
```

### Issue: Poor reconstruction
```python
# Solution: Use higher width or different L0
sae = SAE.from_pretrained(
    release="gemma-scope-9b-pt-res",
    sae_id="layer_24/width_131k/average_l0_140",  # Higher L0
    device="cuda"
)
```

---

## Resources

### Official Documentation
- [Gemma Scope 2 Announcement](https://www.alignmentforum.org/posts/YQro5LyYjDzZrBCdb/announcing-gemma-scope-2)
- [Gemma Scope Technical Paper](https://arxiv.org/pdf/2408.05147)
- [SAELens Documentation](https://jbloomaus.github.io/SAELens/)

### Tutorials
- [Gemma Scope 2 Tutorial Colab](https://colab.sandbox.google.com/drive/1NhWjg7n0nhfW--CjtsOdw5A5J_-Bzn4r)
- [TransformerLens Documentation](https://neelnanda-io.github.io/TransformerLens/)

### Community
- [Neuronpedia](https://neuronpedia.org) - Feature visualization
- [Alignment Forum](https://alignmentforum.org) - Research discussion

---

*Guide created: January 2026*
