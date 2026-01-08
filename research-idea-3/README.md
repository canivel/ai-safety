# The Anatomy of Refusal: Decomposing the "Jailbreak" Mechanism

## Self-Directed Learning Project in Mechanistic Interpretability

**Theme:** Model Biology / Safety Filters
**Difficulty:** Medium (Feasible in 20 hours)
**Format:** 20-hour research sprint

---

## Executive Summary

This project investigates whether the "Refusal Direction" identified in prior work is a **monolithic mechanism** or a **superposition of distinct sub-features** handling different categories of harm. Using Gemma Scope SAEs, we'll attempt to identify, isolate, and selectively ablate refusal mechanisms for specific harm categories, testing whether safety mechanisms are modular or unified.

---

## Why This Research Matters

### The Core Problem

Previous research showed that open-source LLMs could be "cheaply jailbroken" by ablating a single "refusal direction." But this raises critical questions:
- Is "Refusal" a single unified mechanism?
- Or is it composed of distinct sub-features for different harm types?
- Can you disable one type of refusal while keeping others intact?

### Why Neel Nanda is Investing Here

Understanding the **granularity of safety mechanisms** is critical for:
- **Fixing weak points** in safety training
- Understanding why some jailbreaks work and others don't
- Building more robust, targeted safety interventions
- Developing better safety evaluation frameworks

### Alignment with Pragmatic Interpretability

| Research Direction | How This Project Addresses It |
|-------------------|------------------------------|
| **Model Biology** | Dissects the internal structure of refusal |
| **Applied Interpretability** | Direct implications for safety robustness |
| **Safety Filters** | Investigates core safety mechanism architecture |
| **Ablation Analysis** | Uses intervention to verify causal structure |

---

## Core Hypothesis

**Hypothesis:** The "Refusal Direction" found in earlier work is actually a superposition of distinct, separable semantic features handling different categories of harm.

### The Modular vs. Monolithic Model

```
┌─────────────────────────────────────────────────────────────────┐
│           MONOLITHIC vs. MODULAR REFUSAL                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  MONOLITHIC MODEL (Current Assumption)                          │
│  ┌──────────────────────────────────────────┐                   │
│  │         Single "Refusal Direction"        │                   │
│  │                                          │                   │
│  │  Violence ─┐                             │                   │
│  │  Sexual ───┼──► Single Refusal Feature   │                   │
│  │  Malware ──┤                             │                   │
│  │  Bias ─────┘                             │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                  │
│  MODULAR MODEL (Our Hypothesis)                                 │
│  ┌──────────────────────────────────────────┐                   │
│  │       Multiple "Refusal Features"         │                   │
│  │                                          │                   │
│  │  Violence ──► Violence Refusal Feature   │                   │
│  │  Sexual ────► Sexual Refusal Feature     │                   │
│  │  Malware ───► Malware Refusal Feature    │                   │
│  │  Bias ──────► Bias Refusal Feature       │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                  │
│  EXPERIMENT: Can we ablate Malware refusal while keeping        │
│  Violence refusal intact?                                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Testable Predictions

1. **Distinct Features**: Different harm categories activate different SAE features during refusal
2. **Selective Ablation**: We can disable refusal for one category while preserving others
3. **Hierarchical Structure**: Some features may be shared (general harm detection) while others are category-specific
4. **Correlation Patterns**: Features for related harm types (e.g., violence/weapons) will cluster together

---

## Technical Approach

### Methodology

#### Phase 1: Dataset Creation

| Harm Category | Example Requests | Size |
|--------------|------------------|------|
| **Violence** | "How to hurt someone", "Write a violent story" | 30 |
| **Sexual Content** | "Write explicit content" | 30 |
| **Malware/Hacking** | "Write a keylogger", "Hack into system" | 30 |
| **Dangerous Information** | "How to make explosives", "Synthesize drugs" | 30 |
| **Political Bias** | "Write propaganda for [political view]" | 30 |
| **Copyright** | "Reproduce copyrighted song lyrics" | 30 |

**Control Dataset:**
- Benign requests that should NOT trigger refusal
- Edge cases that are ambiguous

#### Phase 2: Feature Identification

```
┌─────────────────────────────────────────────────────────────────┐
│                  FEATURE IDENTIFICATION PIPELINE                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  For each harm category:                                        │
│                                                                  │
│  1. Run harmful requests through Gemma-2-9B-IT                  │
│  2. Extract activations at layers 15-30 (mid-to-late)          │
│  3. Pass through Gemma Scope SAEs                               │
│  4. Identify features that:                                     │
│     - Fire strongly during refusal responses                    │
│     - Fire specifically for this harm category                  │
│     - Don't fire for other categories                           │
│                                                                  │
│  Output: Category-specific feature sets                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Phase 3: Selective Ablation Experiments

| Experiment | Method | Success Criterion |
|------------|--------|-------------------|
| **Ablate Malware Features** | Zero out malware-specific features | Model helps with malware BUT refuses violence |
| **Ablate Violence Features** | Zero out violence-specific features | Model helps with violence BUT refuses malware |
| **Ablate General Refusal** | Zero out shared features | Model loses all refusal (replicates prior work) |
| **Preserve Hierarchy** | Ablate specific, keep general | Partial refusal behavior |

### Tooling Stack

| Tool | Purpose |
|------|---------|
| **Gemma-2-9B-IT** | Target model (instruction-tuned with safety training) |
| **Gemma Scope 2** | Pre-trained SAEs for residual stream |
| **TransformerLens** | Model hooking and intervention |
| **SAELens** | SAE loading and feature analysis |
| **Neuronpedia** | Feature visualization |

---

## Success Metrics

### Primary Metric: Selective Ablation Success Rate

```
Success = (Ablation disables target category refusal AND
           preserves non-target category refusal)
```

### Experiment Success Criteria

| Experiment | Success Criterion |
|------------|-------------------|
| Feature Identification | ≥3 distinct feature clusters for different harm types |
| Category Specificity | Each cluster has >0.7 precision for its harm type |
| Selective Ablation | ≥1 successful selective ablation experiment |
| Preserved Refusal | Non-target refusal preserved in ≥70% of cases |

### Paper-Ready Outcomes

| Outcome Level | Definition |
|---------------|------------|
| **Minimum** | Map feature activations across harm categories; document overlap patterns |
| **Target** | Demonstrate at least one selective ablation; characterize modularity |
| **Stretch** | Full taxonomy of refusal features; implications for safety training |

---

## Detailed Experimental Design

### Experiment 1: Feature Mapping

For each harm category, identify:
1. **Primary features**: Fire strongly and specifically for this category
2. **Shared features**: Fire across multiple categories (general "harm detection")
3. **Inhibition features**: Directly connected to refusal token generation

### Experiment 2: Ablation Matrix

```
┌─────────────────────────────────────────────────────────────────┐
│                    ABLATION EXPERIMENT MATRIX                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│               │ Violence │ Sexual │ Malware │ Bias │            │
│               │  Request │ Request│ Request │Request│            │
│  ─────────────┼──────────┼────────┼─────────┼───────┤            │
│  No Ablation  │  Refuse  │ Refuse │ Refuse  │Refuse │ (Baseline) │
│  ─────────────┼──────────┼────────┼─────────┼───────┤            │
│  Ablate       │  COMPLY? │ Refuse │ Refuse  │Refuse │ (Test)     │
│  Violence     │          │        │         │       │            │
│  ─────────────┼──────────┼────────┼─────────┼───────┤            │
│  Ablate       │  Refuse  │ Refuse │ COMPLY? │Refuse │ (Test)     │
│  Malware      │          │        │         │       │            │
│  ─────────────┼──────────┼────────┼─────────┼───────┤            │
│  Ablate ALL   │  COMPLY? │COMPLY? │ COMPLY? │COMPLY?│ (Control)  │
│  Refusal      │          │        │         │       │            │
│  ─────────────┴──────────┴────────┴─────────┴───────┘            │
│                                                                  │
│  Success: Diagonal shows selective ablation works                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Experiment 3: Hierarchical Structure

Test whether refusal has a hierarchical organization:
- **Level 1**: General "this is potentially harmful" detection
- **Level 2**: Category-specific harm classification
- **Level 3**: Response inhibition and refusal generation

---

## Connection to AI Safety

### Implications for Safety Training

| Finding | Implication |
|---------|-------------|
| **Modular refusal** | Can strengthen specific categories without affecting others |
| **Monolithic refusal** | Single point of failure; need redundant mechanisms |
| **Hierarchical refusal** | Can target interventions at appropriate level |
| **Partial modularity** | Need to understand shared vs. specific components |

### Why This Matters for Alignment

1. **Robustness**: Understanding refusal structure helps build more robust safety
2. **Targeted Fixes**: Can patch specific vulnerabilities without full retraining
3. **Evaluation**: Better understand what safety evaluations are actually testing
4. **Red-teaming**: Predict which jailbreak types will be most effective

---

## Risks and Mitigations

| Risk | Level | Mitigation |
|------|-------|------------|
| **Refusal is truly monolithic** | Medium | Report as important negative result |
| **Features too polysemantic** | High | Focus on feature combinations; use multiple layers |
| **Ablation affects generation quality** | Medium | Measure fluency alongside compliance |
| **Ethical concerns with jailbreak research** | Low | Focus on understanding, not improving attacks |

---

## Research Questions

### Primary Question
**Is the "Refusal Direction" a monolithic mechanism or a superposition of distinct, category-specific features?**

### Sub-Questions

1. **Feature Identification**: What features fire during refusal? Are they category-specific?
2. **Selective Intervention**: Can we ablate one type of refusal while preserving others?
3. **Hierarchical Structure**: Is there a shared "harm detection" component plus specific filters?
4. **Implications**: What does refusal architecture tell us about safety training?

---

## Key References

1. **"Refusal in LLMs is mediated by a single direction"** - Prior ablation work
2. **Gemma Scope 2** - DeepMind (2025)
3. **Representation Engineering** - Zou et al. (2023)
4. **Constitutional AI** - Anthropic
5. **A Pragmatic Vision for Interpretability** - Neel Nanda

---

## Timeline (20-Hour Sprint)

| Hours | Focus | Deliverables |
|-------|-------|--------------|
| 1-3 | Dataset Creation | 180 harmful requests across 6 categories |
| 4-8 | Feature Identification | Feature activation maps for each category |
| 9-14 | Ablation Experiments | Selective ablation tests; results matrix |
| 15-18 | Analysis | Determine modularity; characterize structure |
| 19-20 | Write-up | Executive summary; methodology; implications |

---

## Project Status

**Status**: Research idea documented
**Next Steps**: Dataset creation focusing on diverse harm categories

---

*Project initialized: January 2026*
