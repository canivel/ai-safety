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

**Core Question**: How does RLHF create user modeling circuits that drive sycophancy?

**Hypothesis**: Chat models build internal "User Model" features (e.g., "User is novice") that causally influence response tone, leading to sycophantic behavior.

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
    ├── data/
    ├── results/
    └── figures/
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
