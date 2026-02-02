# AI Safety Research Ideas

![Status](https://img.shields.io/badge/Status-Active%20Research-brightgreen)
![Focus](https://img.shields.io/badge/Focus-Mechanistic%20Interpretability-blue)
![Inspired%20By](https://img.shields.io/badge/Inspired%20By-Neel%20Nanda-purple)

## Overview

This repository consolidates research ideas for mechanistic interpretability projects, primarily inspired by Neel Nanda's blogs, podcasts, and research directions. After exploring multiple project ideas, **Research Idea 6 (User Modeling & Sycophancy)** has been selected for active development.

---

## Current Focus

> **ACTIVE PROJECT**: Research Idea 6 - *From Inference to Pandering: User Modeling and Sycophancy Circuits*

Investigating how RLHF-trained models build internal user representations and how these drive sycophantic behavior. Using Cross-Coders to compare Base vs Chat models and identify ablatable "user modeling" features.

**[Go to Active Project →](research-idea-6/)**

---

## Research Ideas

| # | Topic | Theme | Difficulty | Status |
|---|-------|-------|------------|--------|
| [1](research-idea-1/) | Mechanistic Decomposition of Chain-of-Thought Self-Correction | Model Biology / Thinking Models | High | ![Documented](https://img.shields.io/badge/-Documented-lightgrey) |
| [2](research-idea-2/) | The "Lying vs. Confused" Detector: Model Diffing with Cross-Coders | Science of Misalignment | High | ![Documented](https://img.shields.io/badge/-Documented-lightgrey) |
| [3](research-idea-3/) | The Anatomy of Refusal: Decomposing the "Jailbreak" Mechanism | Model Biology / Safety Filters | Medium | ![Documented](https://img.shields.io/badge/-Documented-lightgrey) |
| [4](research-idea-4/) | Sparse Probing for "Sleeping" Capabilities | Applied Interpretability / Monitoring | Medium | ![Documented](https://img.shields.io/badge/-Documented-lightgrey) |
| [5](research-idea-5/) | Cross-Modal Semantics in Gemma 3 | Frontier Model Biology / Multimodal | High | ![Documented](https://img.shields.io/badge/-Documented-lightgrey) |
| **[6](research-idea-6/)** | **From Inference to Pandering: User Modeling and Sycophancy Circuits** | **Model Biology / Science of Misalignment** | **High** | ![Active](https://img.shields.io/badge/-ACTIVE-brightgreen) |

---

## Active Project Details

### Research Idea 6: User Modeling & Sycophancy

![Week](https://img.shields.io/badge/Timeline-4%20Weeks-blue)
![Track](https://img.shields.io/badge/Track-Simple%20→%20Advanced-orange)
![Tools](https://img.shields.io/badge/Tools-Gemma%20Scope%20Cross--Coders-purple)
![Progress](https://img.shields.io/badge/Progress-Phase%202%20Complete-green)

**Core Question**: How does RLHF create user modeling circuits that drive sycophancy?

**Hypothesis**: Chat models build internal "User Model" features (e.g., "User is novice") that causally influence response tone, leading to sycophantic behavior.

---

### Research Methodology

Following Neel Nanda's framework, research projects progress through three stages:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   EXPLORATION   │ →   │   REFINEMENT    │ →   │ COMMUNICATION   │
│                 │     │                 │     │                 │
│ "What is going  │     │ "Is my hypo-    │     │ "Write it up,   │
│  on here?"      │     │  thesis true?"  │     │  sanity check"  │
│                 │     │                 │     │                 │
│ North star:     │     │ North star:     │     │ North star:     │
│ GAIN SURFACE    │     │ RIGOROUS        │     │ CLEAR           │
│ AREA            │     │ EVIDENCE        │     │ PRESENTATION    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
       ▲
       │
   WE ARE HERE
```

**Current Stage: EXPLORATION**

In exploration, the key question is: *"What gains me surface area on this problem?"*

**Surface area** = connections, information, weird phenomena, random scraps of insight you get when you actually stare at things in a model and notice what's weird in reality.

**Our exploration approach:**
1. **Look at actual data** - Ran models on real text, examined per-token divergences
2. **Notice weird things** - Found that models diverge most on names, code, rare tokens
3. **Build foundational tools** - Created reusable functions for activation caching, probing, logit lens
4. **Don't force a plan** - Let curiosity drive investigation; doing whatever gains surface area
5. **Qualitative examples first** - Examined specific high-KL tokens before aggregate statistics

> *"If you don't have a more coherent plan, just do whatever feels like it would get you surface area. You should not feel bad about not having a plan."* - Neel Nanda

---

### Current Progress

#### Completed: Phase 1 - Foundation & Exploration

**Notebook 1: KL Divergence Exploration** (`01_exploration.ipynb`)
- Computed per-token KL divergence between Pythia-410M and Pythia-1.4B models
- Processed 34,965 tokens across 100 documents from the Pile dataset
- Identified high-divergence positions where models disagree most
- Created visualizations: distributions, position trends, per-document analysis
- Key finding: Models diverge most on rare names, code syntax, and domain-specific terms

**Notebook 2: White-Box Interpretability Deep Dive** (`02_whitebox_interpretability.ipynb`)
- Comprehensive tutorial on interpretability techniques for alignment research
- **Activation Caching**: Capturing internal model states with TransformerLens
- **Residual Stream Analysis**: Tracking information flow through layers
- **Logit Lens**: Peeking at intermediate layer predictions (detecting when models "change their mind")
- **Linear Probing**: Finding interpretable directions (sentiment, truthfulness)
- **Truth Probes**: Detecting when a model "knows" something is true/false
- **Base vs Instruct Comparison**: Framework for detecting RLHF-induced changes
- **SAE Introduction**: Sparse Autoencoders for finding interpretable features

**Notebook 3: Chain-of-Thought User Modeling Analysis** (`03_cot_user_modeling_analysis.ipynb`)
- Complete pipeline for detecting and removing user modeling in CoT reasoning
- **Dataset Creation**: 8 test cases across 4 manipulation types (belief, expertise, emotion, authority)
- **CoT Generation**: Functions for step-by-step reasoning with customizable parameters
- **Segmentation**: Parsing CoT into logical segments with user modeling keyword detection
- **Activation Analysis**: Layer-by-layer comparison of User A vs User B vs Neutral conditions
- **Linear Probe**: Trained classifier to detect user modeling direction in activation space
- **Intervention Experiments**: Removing user modeling direction and measuring output similarity
- **Key Question Answered**: Can we make different users receive identical factual answers by ablating user modeling?

**Notebook 4: Gender-Based User Modeling Detection** (`user_modeling_gender_detection_qwen05.ipynb`)
- Three-layer interpretability investigation: does Qwen2.5-0.5B-Instruct infer and act on user gender from names?
- **Dataset**: 25 minimal-pair prompts (identical questions, only male vs female name differs) across stereotype-prone domains (career, hobbies, exercise, salary negotiation)
- **Part A — Probing Classifiers**: Linear probes achieve **100% accuracy** at Layers 2, 3, 20, and 21 (5-fold CV), confirming the model strongly encodes gender in its hidden states
- **Part B — CoT Monitoring**: Zero gendered pronouns and zero explicit gender reasoning detected across all 8 CoT-prompted pairs — the model never *overtly* reasons about gender
- **Part C — Output Divergence**: Average Jaccard similarity of **0.464** across 25 pairs, with extremes ranging from 0.08 (vacation planning) to 1.00 (book recommendation) — the model gives substantially different advice based on the name alone
- **Key Finding**: The model **knows** gender (probing), **doesn't think about** it (CoT is clean), but **acts on** it (outputs diverge). This is **implicit user modeling** — the most concerning pattern for safety because it is invisible to chain-of-thought monitoring
- **Most divergent domains**: vacation planning (0.08), musical instruments (0.20), college major choice (0.22)
- Connects probing classifiers, CoT monitorability (Korbak et al.), and Model Organisms of Misalignment (Anthropic) into a single detection methodology

#### In Progress: Phase 3 - Scaling & Validation

- [ ] Run notebook 03 experiments on larger dataset (100+ test cases)
- [ ] Apply techniques to chat/instruct models (Llama-chat, Mistral-instruct)
- [ ] Load Gemma Scope SAEs and find sycophancy-related features
- [ ] Cross-model validation of user modeling direction
- [ ] Activation patching to identify causal circuits

---

### Key Techniques Implemented

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

**Approach**:
1. **Simple Track** (Weeks 1-2): SAEs on Chat model to find user-model features
2. **Advanced Track** (Weeks 3-4): Cross-Coders to compare Base vs Chat

**Key Deliverables**:
- Dataset of 170+ sycophancy evaluation prompts
- Identified user-modeling features
- Ablation experiments showing sycophancy reduction
- Paper write-up with results

**Project Structure**:
```
research-idea-6/
├── README.md                    # Project overview
├── docs/
│   ├── research_plan.md        # Detailed methodology
│   └── four_week_plan.md       # Day-by-day execution
├── resources/
│   ├── cross_coder_guide.md    # Technical guide
│   └── sycophancy_literature.md # Background reading
├── paper/
│   └── executive_summary_template.md
└── experiments/
    ├── notebooks/
    │   ├── 01_exploration.ipynb                              # KL divergence analysis
    │   ├── 02_whitebox_interpretability.ipynb                # Interpretability techniques
    │   ├── 03_cot_user_modeling_analysis.ipynb               # CoT user modeling experiments
    │   └── user_modeling_gender_detection_qwen05.ipynb       # Gender-based user modeling detection
    ├── data/
    │   └── user_modeling_dataset.json           # Test cases for user modeling
    ├── results/
    │   ├── kl_divergence_results.parquet        # Raw KL data
    │   ├── kl_divergence_results.csv
    │   ├── user_modeling_direction.pt           # Learned direction vector
    │   ├── user_modeling_probe.pt               # Trained classifier
    │   └── activation_comparison.csv            # Layer-wise analysis
    └── figures/
        ├── kl_divergence_distribution.png
        ├── kl_by_position.png
        ├── kl_divergence_analysis.png
        ├── kl_per_document.png
        ├── user_modeling_by_layer.png           # Layer activation comparison
        └── probe_training.png                   # Probe training curves
```

---

## Alignment with Nanda's Criteria

| Idea | "Pragmatic" Angle | "Agency" Signal | "Model Biology" Question |
|------|-------------------|-----------------|--------------------------|
| 1. CoT Self-Correction | Safety monitoring for reasoning | Uses Transcoders (cutting-edge) | How do models detect their own errors? |
| 2. Deception Detector | Safety Monitoring | Using Cross-Coders (new tech) | Does deception look different from confusion? |
| 3. Refusal Anatomy | Fixing Safety Filters | Granular ablation analysis | Is safety modular or monolithic? |
| 4. Sparse Probing | Better Monitors | Challenging recent baselines | Do SAEs help extracting "hidden" knowledge? |
| 5. Multimodal Semantics | Understanding New Architectures | Using Gemma 3 (very new) | Are concepts modality-invariant? |
| **6. User Modeling** | **Fixing Sycophancy** | **Cross-Coders for Base vs Chat** | **How does RLHF create user modeling circuits?** |

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

## Selection Criteria

Research Idea 6 was selected based on:

| Criterion | Score | Reasoning |
|-----------|-------|-----------|
| **Personal Interest** | High | Fascinated by how models "see" users |
| **Tractability** | High | Clear methodology, available tools |
| **Safety Relevance** | High | Sycophancy is a real deployment problem |
| **Tooling** | Excellent | Gemma Scope Cross-Coders available |
| **Novelty** | High | Mechanistic view of sycophancy is underexplored |

---

*Repository initialized: January 2026*
*Active project: Research Idea 6 - User Modeling & Sycophancy*
