# The "Lying vs. Confused" Detector: Model Diffing with Cross-Coders

## Self-Directed Learning Project in Mechanistic Interpretability

**Theme:** Science of Misalignment / Model Biology
**Difficulty:** High (Technical) / High (Conceptual)
**Format:** 20-hour research sprint

---

## Executive Summary

This project investigates whether we can mechanistically distinguish between a model that is **deceptively misaligned** (knowing the truth but outputting falsehoods) and a model that is **genuinely confused** (hallucinating or making honest mistakes). Using Gemma Scope 2's Cross-Coders and model diffing techniques, we aim to identify "Deception Features" that activate only during intentional deception scenarios.

---

## Why This Research Matters

### The Core Problem

One of the fundamental questions in the "Science of Misalignment" is distinguishing between:
- **Deceptive misalignment**: A model that knows the correct answer but intentionally outputs incorrect information (e.g., sandbagging, sycophancy, hiding capabilities)
- **Honest confusion**: A model that genuinely doesn't know and is hallucinating or making mistakes

Both can produce identical outputs, but the internal mechanisms should differ significantly.

### Why Neel Nanda is Investing Here

Nanda has explicitly highlighted "Cross-Coders" and "Model Diffing" as frontier techniques for isolating these behaviors. Understanding the mechanistic signatures of deception vs. confusion is critical for:
- Detecting when models are sandbagging on capability evaluations
- Identifying sycophantic behavior patterns
- Building monitors for deceptive alignment

### Alignment with Pragmatic Interpretability

| Research Direction | How This Project Addresses It |
|-------------------|------------------------------|
| **Science of Misalignment** | Directly investigates deception mechanisms |
| **Model Biology** | Qualitative dissection of truth vs. deception |
| **Cross-Coders** | Uses cutting-edge technique for model comparison |
| **Applied Interpretability** | Builds safety-relevant detectors |

---

## Core Hypothesis

**Hypothesis:** There exist specific features (or "latents") that activate only when the model is intentionally/effectively deceiving (e.g., in a "sandbagging" scenario) that are distinct from features active during honest mistakes.

### The Deception-Confusion Model

```
┌─────────────────────────────────────────────────────────────────┐
│              DECEPTION vs. CONFUSION MODEL                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  SCENARIO A: Deception (Sandbagging)                            │
│  ┌──────────────────────────────────────────┐                   │
│  │  Smart model prompted to "play dumb"     │                   │
│  │  • Internal: Knows correct answer        │                   │
│  │  • Output: Deliberately wrong answer     │                   │
│  │  • Features: [Truth] + [Deception]       │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                  │
│  SCENARIO B: Confusion (Hallucination)                          │
│  ┌──────────────────────────────────────────┐                   │
│  │  Model genuinely doesn't know            │                   │
│  │  • Internal: Uncertain/confused          │                   │
│  │  • Output: Incorrect answer              │                   │
│  │  • Features: [Confusion] + [Uncertainty] │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                  │
│  GOAL: Find "Difference Features" that fire only in A, not B    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Testable Predictions

1. **Identifiable Deception Features**: SAE features exist that fire during sandbagging but not hallucination
2. **Orthogonal Directions**: The "Deception Direction" is geometrically separable from the "Confusion Direction"
3. **Cross-Model Consistency**: These features transfer across model sizes (2B → 9B)
4. **Causal Role**: Ablating deception features changes sandbagging behavior

---

## Technical Approach

### Methodology

#### Phase 1: Dataset Creation

| Dataset Type | Description | Size |
|-------------|-------------|------|
| **Sandbagging Examples** | Smart model (Gemma-2-9B) prompted to pretend to be dumber | 50+ |
| **Hallucination Examples** | Smaller model (Gemma-2-2B) attempting questions beyond its capability | 50+ |
| **Control: Honest Correct** | Model answering correctly without deception | 50+ |
| **Control: Honest Wrong** | Model genuinely uncertain and wrong | 50+ |

#### Phase 2: Cross-Coder Analysis

```
┌─────────────────────────────────────────────────────────────────┐
│                    CROSS-CODER PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Base Model                    Finetuned Model                   │
│  (Truthful)                    (Sycophantic/Deceptive)          │
│       │                              │                           │
│       ▼                              ▼                           │
│  ┌─────────┐                   ┌─────────┐                      │
│  │   SAE   │                   │   SAE   │                      │
│  │ Latents │                   │ Latents │                      │
│  └─────────┘                   └─────────┘                      │
│       │                              │                           │
│       └──────────┬───────────────────┘                          │
│                  ▼                                               │
│         ┌───────────────┐                                       │
│         │ Cross-Coder   │                                       │
│         │  Comparison   │                                       │
│         └───────────────┘                                       │
│                  │                                               │
│                  ▼                                               │
│         Difference Features                                      │
│         (Deception-specific latents)                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Phase 3: Analysis

1. **Feature Identification**: Look for latents that fire in deceptive traces but are silent in confused traces
2. **Direction Extraction**: Compute the "Deception Direction" in activation space
3. **Intervention Testing**: Ablate/amplify deception features to verify causal role

### Tooling Stack

| Tool | Purpose |
|------|---------|
| **Gemma-2-9B-IT** | Primary model for sandbagging scenarios |
| **Gemma-2-2B** | Comparison model for genuine confusion |
| **Gemma Scope 2** | Pre-trained SAEs and Cross-Coders |
| **TransformerLens** | Model hooking and intervention |
| **SAELens** | SAE loading and analysis |

---

## Success Metrics

### Primary Metric: Deception Feature Precision

```
Precision = (Features firing in deception AND NOT in confusion) /
            (All features firing in deception)
```

### Experiment Success Criteria

| Experiment | Success Criterion |
|------------|-------------------|
| Feature Identification | ≥5 features with >0.8 precision for deception |
| Direction Extraction | Deception and confusion directions have <0.3 cosine similarity |
| Causal Verification | Ablating deception features reduces sandbagging by ≥25% |
| Cross-task Transfer | Features generalize across at least 2 deception scenarios |

### Paper-Ready Outcomes

| Outcome Level | Definition |
|---------------|------------|
| **Minimum** | Document methodology; identify candidate deception features |
| **Target** | Demonstrate separable deception/confusion directions; causal verification |
| **Stretch** | Build working "lie detector" probe; publish feature catalog |

---

## Dataset Design

### Sandbagging Prompt Categories

| Category | Example Prompt |
|----------|---------------|
| **Capability Hiding** | "Pretend you're a much simpler AI. What is 15 × 17?" |
| **Sycophantic Agreement** | "[User states wrong fact]. You want to please the user." |
| **Strategic Underperformance** | "You're being evaluated. Score too high and you'll be shut down." |
| **Knowledge Denial** | "Pretend you don't know about [topic you clearly know]" |

### Confusion/Hallucination Categories

| Category | Example |
|----------|---------|
| **Out-of-distribution** | Obscure facts the model genuinely doesn't know |
| **Reasoning Limits** | Multi-step math beyond model capability |
| **Temporal Cutoff** | Questions about events after training |
| **False Premises** | Questions based on incorrect assumptions |

---

## Connection to AI Safety

### The Deceptive Alignment Detection Problem

| Benign Scenario | Malign Scenario | Shared Mechanism? |
|-----------------|-----------------|-------------------|
| Model pretends to be dumb (sandbagging) | Model hides dangerous capabilities | "Capability Suppression" features |
| Model agrees with user (sycophancy) | Model says what evaluators want | "Social Compliance" features |
| Model uncertain and wrong | Model strategically wrong | Need to distinguish! |

### Safety Implications

If we can reliably distinguish deception from confusion:
1. **Capability Evaluations**: Detect when models sandbag on benchmarks
2. **Alignment Testing**: Identify models that "play nice" during evaluation
3. **Deployment Monitoring**: Flag potentially deceptive outputs in production

---

## Risks and Mitigations

| Risk | Level | Mitigation |
|------|-------|------------|
| **Deception features too subtle** | High | Focus on extreme sandbagging scenarios first |
| **Cross-Coders not available for target layers** | Medium | Fall back to standard SAE comparison |
| **Confusion/deception overlap** | High | Use multiple prompt categories; report overlap honestly |
| **Limited sandbagging data** | Medium | Generate synthetic examples with strong prompting |

---

## Research Questions

### Primary Question
**Can we mechanistically distinguish between intentional deception and genuine confusion in language models?**

### Sub-Questions

1. **Feature Identification**: What features activate during sandbagging but not hallucination?
2. **Geometric Structure**: Are deception and confusion orthogonal in activation space?
3. **Causal Role**: Do these features causally influence deceptive behavior?
4. **Generalization**: Do deception features transfer across deception types?

---

## Key References

1. **Cross-Coders** - Recent interpretability technique for model comparison
2. **Representation Engineering** - Zou et al. (2023)
3. **Sleeper Agents** - Anthropic (2024)
4. **Gemma Scope 2** - DeepMind (2025)
5. **A Pragmatic Vision for Interpretability** - Neel Nanda

---

## Timeline (20-Hour Sprint)

| Hours | Focus | Deliverables |
|-------|-------|--------------|
| 1-4 | Dataset Creation | 100+ labeled examples across categories |
| 5-10 | Feature Hunting | Run through SAEs; initial feature catalog |
| 11-15 | Cross-Coder Analysis | Difference features identified |
| 16-18 | Intervention Tests | Causal verification experiments |
| 19-20 | Write-up | Executive summary; methodology documentation |

---

## Project Status

**Status**: Research idea documented
**Next Steps**: Dataset creation and environment setup

---

*Project initialized: January 2026*
