# AI Safety Research Ideas

## Overview

This repository consolidates research ideas for mechanistic interpretability projects, primarily inspired by Neel Nanda's blogs, podcasts, and research directions. I'm exploring multiple project ideas before committing to one for deeper investigation.

## Status

Currently in the **exploration phase** - 5 research ideas documented. All ideas are designed for the 20-hour application sprint format and address Neel Nanda's specific requests for "evidence for and against" hypotheses regarding safety-relevant behaviors.

## Research Ideas

| Folder | Topic | Theme | Difficulty | Status |
|--------|-------|-------|------------|--------|
| [research-idea-1](research-idea-1/) | Mechanistic Decomposition of Chain-of-Thought Self-Correction | Model Biology / Thinking Models | High | Documented |
| [research-idea-2](research-idea-2/) | The "Lying vs. Confused" Detector: Model Diffing with Cross-Coders | Science of Misalignment | High | Documented |
| [research-idea-3](research-idea-3/) | The Anatomy of Refusal: Decomposing the "Jailbreak" Mechanism | Model Biology / Safety Filters | Medium | Documented |
| [research-idea-4](research-idea-4/) | Sparse Probing for "Sleeping" Capabilities | Applied Interpretability / Monitoring | Medium | Documented |
| [research-idea-5](research-idea-5/) | Cross-Modal Semantics in Gemma 3 | Frontier Model Biology / Multimodal | High | Documented |

### Alignment with Nanda's Criteria

| Idea | "Pragmatic" Angle | "Agency" Signal | "Model Biology" Question |
|------|-------------------|-----------------|--------------------------|
| 1. CoT Self-Correction | Safety monitoring for reasoning | Uses Transcoders (cutting-edge) | How do models detect their own errors? |
| 2. Deception Detector | Safety Monitoring | Using Cross-Coders (new tech) | Does deception look different from confusion? |
| 3. Refusal Anatomy | Fixing Safety Filters | Granular ablation analysis | Is safety modular or monolithic? |
| 4. Sparse Probing | Better Monitors | Challenging recent baselines | Do SAEs help extracting "hidden" knowledge? |
| 5. Multimodal Semantics | Understanding New Architectures | Using Gemma 3 (very new) | Are concepts modality-invariant? |

## Sources

Research ideas are drawn from:
- Neel Nanda's blog posts on [neelnanda.io](https://neelnanda.io)
- MATS research directions and application guidance
- 80,000 Hours podcast episodes on mechanistic interpretability
- "A Pragmatic Vision for Interpretability" and related posts
- Alignment Forum discussions
- Gemma Scope 2 release (covering Gemma 3) - DeepMind's latest SAEs and Cross-Coders
- Recent work on Cross-Coders, Model Diffing, and multimodal interpretability

## Structure

Each research idea folder contains:
- `README.md` - Project overview and hypothesis
- `docs/` - Detailed plans and methodology
- `resources/` - Technical guides and references
- `experiments/` - Notebooks, data, and results (when implemented)

## Goal

Select the most promising research direction based on:
1. Personal interest and curiosity
2. Tractability for self-directed learning
3. Relevance to AI safety
4. Available tooling (Gemma Scope, TransformerLens, etc.)

