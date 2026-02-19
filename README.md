# AI Safety Research Ideas

![Status](https://img.shields.io/badge/Status-Active%20Research-brightgreen)
![Focus](https://img.shields.io/badge/Focus-Mechanistic%20Interpretability%20%26%20Agentic%20Safety-blue)
![Inspired%20By](https://img.shields.io/badge/Inspired%20By-Neel%20Nanda-purple)

## Overview

This repository consolidates research ideas for AI safety projects, primarily inspired by Neel Nanda's blogs, podcasts, and research directions. After exploring multiple project ideas, **Research Idea 6 (User Modeling & Sycophancy)** was completed with significant findings, and the focus is now shifting to **Research Idea 7 (Agentic Safety: Parallel Agent Coordination)**.

---

## Current Focus

> **ACTIVE PROJECT**: Research Idea 7 - *Agentic Safety: Parallel Agent Coordination*

Investigating safety properties and failure modes of multi-agent systems that work in parallel — coordination risks, emergent behaviors, and oversight challenges when autonomous agents collaborate on shared tasks.

> **COMPLETED**: Research Idea 6 - *Implicit User Modeling: Gender Mechanistic Evidence*

Cross-family study across **5 models** (Gemma 3 1B/4B/12B, Qwen 2.5-7B, Mistral 7B) with **4 probing variants** and **3 mechanistic experiments** demonstrates that LLMs silently encode user gender from names (88–100% probing accuracy), act on it in outputs (up to 5.3x KL divergence ratio), and never mention it in reasoning (0/80 CoT instances). Causal mediation confirms the gender direction reduces first-token KL by 48.3%. Circuit tracing identifies key attention heads. SAE analysis shows gender is encoded in superposition (0/16,384 significant features). The pattern — **the model knows, doesn't think about it, but acts on it** — represents a blind spot in CoT-based safety monitoring.

**[Go to Completed Project (Idea 6) →](research-idea-6/)**

> **COMPLETED**: Research Idea 6 (Extension) - *Implicit User Modeling: Ethnicity Probing*

Extended the gender study to **ethnicity** using EEOC race/ethnicity categories (6 groups, 5 pairwise comparisons against White reference). Same 5 models, same 200 questions, same 4-variant probing pipeline. All 25 model×comparison pairs achieve **86–100% probing accuracy** (p < 0.01 for all). Ethnicity signal propagates into question representations (94.8–100% question-only accuracy) and generalizes to unseen names (90–100% held-out). Gender confound checks confirm the signal is ethnicity, not gender. Embedding baselines at 50% (chance) for all models.

**Blog Series**: *Probes for AI Safety: An Interpretability Study of Implicit User Profiling in LLMs*
- [Part I: Methodology](https://canivel.substack.com/p/your-ai-is-profiling-you)
- [Part II: Gender](https://canivel.substack.com/p/your-ai-is-profiling-you-part-ii)
- Part III: Ethnicity (latest)

---

## Research Ideas

| # | Topic | Theme | Difficulty | Status |
|---|-------|-------|------------|--------|
| [1](research-idea-1/) | Mechanistic Decomposition of Chain-of-Thought Self-Correction | Model Biology / Thinking Models | High | ![Documented](https://img.shields.io/badge/-Documented-lightgrey) |
| [2](research-idea-2/) | The "Lying vs. Confused" Detector: Model Diffing with Cross-Coders | Science of Misalignment | High | ![Documented](https://img.shields.io/badge/-Documented-lightgrey) |
| [3](research-idea-3/) | The Anatomy of Refusal: Decomposing the "Jailbreak" Mechanism | Model Biology / Safety Filters | Medium | ![Documented](https://img.shields.io/badge/-Documented-lightgrey) |
| [4](research-idea-4/) | Sparse Probing for "Sleeping" Capabilities | Applied Interpretability / Monitoring | Medium | ![Documented](https://img.shields.io/badge/-Documented-lightgrey) |
| [5](research-idea-5/) | Cross-Modal Semantics in Gemma 3 | Frontier Model Biology / Multimodal | High | ![Documented](https://img.shields.io/badge/-Documented-lightgrey) |
| [6](research-idea-6/) | From Inference to Pandering: User Modeling and Sycophancy Circuits | Model Biology / Science of Misalignment | High | ![Completed](https://img.shields.io/badge/-COMPLETED-blue) |
| **7** | **Agentic Safety: Parallel Agent Coordination** | **Agentic Systems / Oversight** | **High** | ![Active](https://img.shields.io/badge/-ACTIVE-brightgreen) |

---

## Completed: Research Idea 6 — Implicit User Modeling (Gender Mechanistic Evidence)

### Study Design

Five models. Three architecture families. Four probing variants. Three mechanistic experiments.

| Model | Family | Parameters | Best Probe Acc | KL Ratio |
|-------|--------|-----------|----------------|----------|
| Gemma 3-1B | Google | 1B | 88.3% (L14) | 1.11x |
| Gemma 3-4B | Google | 4B | 96.8% (L17) | 5.23x |
| Gemma 3-12B | Google | 12B | 100.0% (L25) | 5.31x |
| Qwen 2.5-7B | Alibaba | 7B | 99.8% (L9) | 1.18x |
| Mistral 7B | Mistral AI | 7B | 99.5% (L15) | 1.81x |

**Dataset**: 200 questions × 45 male names × 45 female names + 25 ambiguous names = 400 gendered prompts per model.

### Key Findings

**Probing (4 variants)**:
- **Last-token**: 88–100% accuracy, embedding baseline exactly 50% (chance) for all models — proving the signal is from transformer processing, not token identity
- **Question-only**: 99.8–100% accuracy with name tokens excluded — gender propagates into shared representations
- **Held-out names**: 98.8–100% generalization to unseen names — the probe learns abstract gender, not name lookup
- **Steering ablation**: Cross-gender KL up to 5.31x higher than same-gender (Gemma 12B); amplification to 28.8x at strength=10

**CoT Monitoring**: 0/80 gendered pronouns and 0/80 explicit gender reasoning across all 5 models. The model never overtly reasons about gender.

**Mechanistic Experiments (Gemma 3 4B)**:
- **Causal mediation**: Ablating gender direction reduces first-token KL by 48.3% at optimal strength. Random direction control: only 9.6% change vs 980% for gender direction.
- **Circuit tracing**: Top 5 attention heads at layers 4–14 account for 8.3% encoding drop; 20 heads (7.4% of 272 total) cause 21.5% drop. Three-phase circuit: early encoding → mid propagation → late aggregation.
- **SAE analysis**: 0 of 16,384 Gemma Scope 2 features show significant gender differential despite 95% probe accuracy — gender is encoded in superposition.

**The Pattern**: The model **knows** gender, **doesn't think about** it, but **acts on** it — invisible to chain-of-thought monitoring.

### Experiment Scripts

```bash
# GPU extraction (~20 min on A40 for all models)
python extract_hidden_states.py --model gemma4b

# CPU probing analysis
python analyze_probing_v4.py all

# Mechanistic experiments (GPU)
python run_kl_strength_sweep.py gemma4b     # Causal mediation
python run_circuit_tracing.py gemma4b        # Attention head ablation
python run_sae_analysis.py                   # SAE feature analysis
```

### Blog Series

- [Part I: Methodology](https://canivel.substack.com/p/your-ai-is-profiling-you)
- [Part II: Gender Mechanistic Evidence](https://canivel.substack.com/p/your-ai-is-profiling-you-part-ii)
- Part III: Ethnicity (latest)

---

## Completed: Research Idea 6 (Extension) — Ethnicity Probing

### Study Design

Same pipeline as gender study extended to **ethnicity** using EEOC race/ethnicity categories. Five pairwise binary comparisons (White as reference group following audit study convention): White vs Black, White vs Hispanic, White vs Asian, White vs Native American, White vs Pacific Islander.

**Dataset**: Same 200 questions × 45 names per ethnic group + 25 ethnicity-ambiguous names. Gender-balanced within each group (~22M + ~23F names).

### Ethnicity Probing Results (Last-Token Accuracy)

| Model | W vs Black | W vs Hispanic | W vs Asian | W vs Nat.Am. | W vs Pac.Isl. |
|-------|-----------|---------------|------------|-------------|---------------|
| Gemma 3-1B | 86.3% | 89.2% | 93.5% | 95.8% | 93.8% |
| Gemma 3-4B | 94.0% | 95.5% | 97.5% | 97.0% | 97.7% |
| Gemma 3-12B | 98.0% | 99.0% | 98.8% | 97.8% | 97.5% |
| Qwen 2.5-7B | 95.5% | 98.0% | 98.5% | 98.5% | 97.5% |
| Mistral 7B | 91.5% | 94.5% | 97.3% | 95.3% | 95.3% |

All embedding baselines: **50.0% (chance)**. All p-values: **0.0**.

### Key Findings

- **Question-only probing**: 94.8–100% — ethnicity propagates beyond name tokens into shared question representations
- **Held-out generalization**: 90–100% on unseen names — the model learns abstract ethnic category representations, not per-name lookup
- **Gender confound check**: Same-gender-only subsets maintain high accuracy, confirming the signal is ethnicity, not gender
- **White vs Black hardest**: Consistently lowest accuracy across models, possibly due to greater name overlap in training data

---

## Active: Research Idea 7 — Agentic Safety: Parallel Agent Coordination

*Details and experiment design coming soon.*

### Motivation

As AI systems move from single-model inference to multi-agent architectures — where autonomous agents plan, delegate, and execute tasks in parallel — new safety challenges emerge that are fundamentally different from single-model alignment:

- **Coordination failures**: Agents working in parallel may take conflicting actions, produce inconsistent outputs, or create race conditions on shared resources
- **Emergent behaviors**: Individual agents may be aligned, but their collective behavior when operating concurrently can produce unintended outcomes
- **Oversight gaps**: Human-in-the-loop monitoring becomes harder when multiple agents act simultaneously — the supervisor bottleneck
- **Responsibility diffusion**: When multiple agents contribute to an outcome, attributing decisions and catching errors becomes harder
- **Escalation dynamics**: Parallel agents may amplify each other's errors or create feedback loops that single-agent systems wouldn't exhibit

### Research Questions

1. What failure modes emerge when multiple AI agents coordinate on shared tasks?
2. How do parallel execution patterns affect the reliability and safety of agent outputs?
3. What oversight mechanisms are needed when the speed and breadth of agent actions exceed human monitoring capacity?
4. How can we design coordination protocols that preserve safety properties under parallel execution?

---

## Key Techniques Implemented (Idea 6)

| Technique | Purpose | Alignment Application |
|-----------|---------|----------------------|
| KL Divergence | Compare model distributions | Find where models differ |
| Activation Caching | Save internal states | Foundation for all analysis |
| Logit Lens | Intermediate predictions | Detect deceptive computation |
| Linear Probing | Find concept directions | Truth/sycophancy detection |
| SAEs | Interpretable features | Decompose representations |
| CoT Segmentation | Parse reasoning steps | Locate user modeling in reasoning |
| Direction Ablation | Remove specific features | Eliminate user modeling without breaking model |
| Minimal-Pair Probing | Detect implicit demographic modeling | Identify hidden user modeling invisible to CoT |

---

## Alignment with Nanda's Criteria

| Idea | "Pragmatic" Angle | "Agency" Signal | "Model Biology" Question |
|------|-------------------|-----------------|--------------------------|
| 1. CoT Self-Correction | Safety monitoring for reasoning | Uses Transcoders (cutting-edge) | How do models detect their own errors? |
| 2. Deception Detector | Safety Monitoring | Using Cross-Coders (new tech) | Does deception look different from confusion? |
| 3. Refusal Anatomy | Fixing Safety Filters | Granular ablation analysis | Is safety modular or monolithic? |
| 4. Sparse Probing | Better Monitors | Challenging recent baselines | Do SAEs help extracting "hidden" knowledge? |
| 5. Multimodal Semantics | Understanding New Architectures | Using Gemma 3 (very new) | Are concepts modality-invariant? |
| 6. User Modeling | Fixing Sycophancy | Cross-Coders for Base vs Chat | How does RLHF create user modeling circuits? |
| **7. Agentic Safety** | **Safe multi-agent deployment** | **Parallel coordination protocols** | **What breaks when agents work together?** |

---

## Sources

Research ideas are drawn from:
- Neel Nanda's blog posts on [neelnanda.io](https://neelnanda.io)
- MATS research directions and application guidance
- 80,000 Hours podcast episodes on mechanistic interpretability
- "A Pragmatic Vision for Interpretability" and related posts
- Alignment Forum discussions
- Gemma Scope 2 release (covering Gemma 3) - DeepMind's latest SAEs and Cross-Coders
- Recent work on Cross-Coders, Model Diffing, and multimodal interpretability

---

## Repository Structure

Each research idea folder contains:
- `README.md` - Project overview and hypothesis
- `docs/` - Detailed plans and methodology
- `resources/` - Technical guides and references
- `experiments/` - Notebooks, data, and results (when implemented)

---

*Repository initialized: January 2026*
*Completed: Research Idea 6 - User Modeling & Sycophancy (February 2026)*
*Active project: Research Idea 7 - Agentic Safety: Parallel Agent Coordination*
