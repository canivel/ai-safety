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

> **COMPLETED**: Research Idea 6 - *From Inference to Pandering: User Modeling and Sycophancy Circuits*

Successfully demonstrated that even small RLHF-tuned models (Qwen2.5-0.5B-Instruct) implicitly profile users by gender: 100% probing accuracy, zero chain-of-thought signal, and substantially divergent outputs. This pattern — **the model knows, doesn't think about it, but acts on it** — represents a blind spot in CoT-based safety monitoring. Scaling to larger models is likely but hard to verify without weight access, motivating a shift toward higher-level agentic safety research.

**[Go to Completed Project (Idea 6) →](research-idea-6/)**

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

## Completed: Research Idea 6 — User Modeling & Sycophancy

### Key Findings

We built a three-layer detection framework to catch implicit user modeling in RLHF-tuned models:

**Layer 1 — Probing Classifiers**: Linear probes achieve **100% accuracy** at detecting gender from hidden states at Layers 2, 3, 20, and 21 of Qwen2.5-0.5B-Instruct. The model perfectly encodes user gender from names alone.

**Layer 2 — CoT Monitoring**: Zero gendered pronouns and zero explicit gender reasoning across all chain-of-thought outputs. The model never overtly reasons about gender.

**Layer 3 — Output Divergence**: Average Jaccard similarity of **0.464** across 25 minimal-pair questions. The most extreme case (vacation planning) scored 0.08 — nearly completely different responses for the same question.

**The Pattern**: The model **knows** gender, **doesn't think about** it, but **acts on** it. This is implicit user modeling — invisible to chain-of-thought monitoring, the primary safety technique proposed for advanced AI oversight.

### Why We're Moving On

The user modeling findings are strong for small open-weight models. However, scaling this research faces a fundamental barrier: **larger frontier models are increasingly closed-weight**. Without access to hidden states, probing classifiers can't run, and the mechanistic evidence that makes the case undeniable is unavailable. Output divergence alone can still be measured, but it lacks the explanatory power of the full three-layer framework.

This motivates a shift toward **agentic safety** — a domain where the risks are observable at the system level without requiring model internals, and where the safety challenges are rapidly becoming urgent as multi-agent deployments scale.

### Notebooks

- `01_exploration.ipynb` — KL divergence analysis between Pythia models
- `02_whitebox_interpretability.ipynb` — Comprehensive interpretability techniques tutorial
- `03_cot_user_modeling_analysis.ipynb` — CoT user modeling pipeline with direction ablation
- `user_modeling_gender_detection_qwen05.ipynb` — Gender-based implicit user modeling detection

### Observable Signal Experiment

Additionally, we built an observable-signal framework testing user modeling beyond gender — measuring conclusion stability, agreement gradients, confidence-evidence mismatch, and counterfactual resistance across Base vs Chat models. The Chat model showed higher instability (0.250 vs 0.208) and higher counterfactual resistance (0.80 vs 0.60), consistent with RLHF amplifying user-signal sensitivity.

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
