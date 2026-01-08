# Sparse Probing for "Sleeping" Capabilities

## Self-Directed Learning Project in Mechanistic Interpretability

**Theme:** Applied Interpretability / Monitoring
**Difficulty:** Medium (Strong Conceptual Fit)
**Format:** 20-hour research sprint

---

## Executive Summary

This project investigates whether **Sparse Autoencoder (SAE) features provide a better basis for training detectors** of rare or "sleeping" capabilities compared to raw model activations. We'll focus on detecting code backdoors as a concrete test case, comparing SAE-based probes against standard linear probes to determine which approach generalizes better to unseen threats in low-data regimes.

---

## Why This Research Matters

### The Core Problem

A major challenge in AI safety is detecting **"sleeping" capabilities**—knowledge or skills that a model possesses but doesn't demonstrate until a specific trigger occurs. Examples include:
- Code vulnerabilities hidden in seemingly safe code
- Knowledge of dangerous information that isn't revealed without specific prompts
- Capabilities that emerge only under certain conditions

Current monitoring approaches may miss these dormant capabilities because they rely on behavioral signals that aren't present until the capability is activated.

### Why Neel Nanda is Investing Here

Nanda has become a "big convert to probes" as a pragmatic tool for monitoring systems. Key questions he's interested in:
- **Are SAEs actually better than linear probes** for extracting hidden information?
- Can we build **practical monitors** for dangerous capabilities?
- Do SAE features provide **better generalization** in low-data scenarios?

His students have recently published work comparing SAEs to standard linear probes, making this a timely research direction.

### Alignment with Pragmatic Interpretability

| Research Direction | How This Project Addresses It |
|-------------------|------------------------------|
| **Applied Interpretability** | Builds concrete safety monitoring tool |
| **Pragmatic Tools** | Tests probes as practical detectors |
| **Low-Data Regimes** | Focuses on realistic deployment scenarios |
| **Capability Monitoring** | Directly addresses hidden capability detection |

---

## Core Hypothesis

**Hypothesis:** Sparse Autoencoder features provide a better basis for training "detectors" of rare/sleeping concepts than the raw residual stream, particularly in low-data regimes.

### The SAE Advantage Model

```
┌─────────────────────────────────────────────────────────────────┐
│              SAE PROBE vs. STANDARD PROBE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  STANDARD LINEAR PROBE                                          │
│  ┌──────────────────────────────────────────┐                   │
│  │  Raw Activations ──► Linear Probe ──► Detection             │
│  │                                          │                   │
│  │  • High-dimensional input                │                   │
│  │  • Features are entangled                │                   │
│  │  • Needs more training data              │                   │
│  │  • May overfit to spurious correlations  │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                  │
│  SAE-BASED PROBE                                                │
│  ┌──────────────────────────────────────────┐                   │
│  │  Raw Acts ──► SAE ──► Sparse Features ──► Probe ──► Detection│
│  │                                          │                   │
│  │  • Disentangled features                 │                   │
│  │  • Sparse, interpretable basis           │                   │
│  │  • May need less training data           │                   │
│  │  • Features have semantic meaning        │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                  │
│  QUESTION: Which generalizes better to unseen threats?          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Testable Predictions

1. **Better Generalization**: SAE probes outperform standard probes on held-out backdoor types
2. **Data Efficiency**: SAE probes achieve comparable accuracy with less training data
3. **Interpretability**: We can identify which SAE features the probe uses (explainability bonus)
4. **Transfer**: SAE probes transfer better across model sizes or domains

---

## Technical Approach

### Test Case: Code Backdoor Detection

We'll use **code backdoor detection** as our concrete test case because:
- Clear ground truth (backdoor present/absent)
- Diverse types of backdoors to test generalization
- Safety-relevant application
- Existing datasets and benchmarks

### Methodology

#### Phase 1: Dataset Curation

| Dataset Component | Description | Size |
|------------------|-------------|------|
| **Clean Code** | Safe code snippets without vulnerabilities | 200+ |
| **Obvious Backdoors** | Clearly malicious code (training) | 100+ |
| **Subtle Backdoors** | Hard-to-detect vulnerabilities (test) | 50+ |
| **Novel Backdoors** | Types not in training (generalization test) | 30+ |

**Backdoor Categories:**
- SQL injection vulnerabilities
- Command injection
- Path traversal
- Hardcoded credentials
- Timing-based backdoors
- Obfuscated malicious code

#### Phase 2: Probe Training

```
┌─────────────────────────────────────────────────────────────────┐
│                    PROBE TRAINING PIPELINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  For each code snippet:                                         │
│                                                                  │
│  1. Pass through Gemma-2-9B                                     │
│  2. Extract activations at multiple layers                      │
│                                                                  │
│  STANDARD PROBE:                                                │
│  ┌─────────────────────────────────────────┐                    │
│  │  Raw activations ──► Logistic Regression │                    │
│  │  (d_model dimensional input)             │                    │
│  └─────────────────────────────────────────┘                    │
│                                                                  │
│  SAE PROBE:                                                     │
│  ┌─────────────────────────────────────────┐                    │
│  │  Raw activations ──► SAE ──► Sparse features                 │
│  │  Sparse features ──► Logistic Regression │                    │
│  │  (d_sae dimensional sparse input)        │                    │
│  └─────────────────────────────────────────┘                    │
│                                                                  │
│  Compare: Accuracy, Generalization, Data Efficiency             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Phase 3: Evaluation

| Metric | Description |
|--------|-------------|
| **In-distribution accuracy** | Performance on held-out examples of trained backdoor types |
| **Out-of-distribution accuracy** | Performance on novel backdoor types |
| **Data efficiency curve** | Accuracy vs. training set size |
| **Feature interpretability** | Can we understand what the SAE probe detects? |

### Tooling Stack

| Tool | Purpose |
|------|---------|
| **Gemma-2-9B** | Base model for activation extraction |
| **Gemma Scope 2** | Pre-trained SAEs |
| **scikit-learn** | Linear probe training |
| **TransformerLens** | Activation extraction |
| **SAELens** | SAE loading and feature analysis |

---

## Success Metrics

### Primary Metric: Generalization Gap

```
Generalization_Advantage = OOD_Accuracy(SAE_Probe) - OOD_Accuracy(Standard_Probe)
```

Where OOD = Out-of-Distribution (novel backdoor types)

### Experiment Success Criteria

| Experiment | Success Criterion |
|------------|-------------------|
| SAE Probe Accuracy | ≥80% on in-distribution test set |
| Generalization Advantage | SAE probe ≥10% better on OOD test set |
| Data Efficiency | SAE probe matches standard probe with 50% less data |
| Interpretability | Can identify ≥3 meaningful features used by probe |

### Paper-Ready Outcomes

| Outcome Level | Definition |
|---------------|------------|
| **Minimum** | Document methodology; baseline probe performance |
| **Target** | Demonstrate generalization advantage; interpret probe features |
| **Stretch** | Transfer to other domains; practical deployment recommendations |

---

## Detailed Experimental Design

### Experiment 1: Data Efficiency Comparison

Train both probes on increasing amounts of data and measure accuracy:

```
Training Size:  10%   25%   50%   75%   100%
Standard Probe: ___   ___   ___   ___   ___
SAE Probe:      ___   ___   ___   ___   ___
```

**Hypothesis**: SAE probe reaches high accuracy with less data

### Experiment 2: Generalization to Novel Threats

| Backdoor Type | In Training? | Standard Probe | SAE Probe |
|--------------|--------------|----------------|-----------|
| SQL Injection | Yes | ___ | ___ |
| Command Injection | Yes | ___ | ___ |
| Path Traversal | No | ___ | ___ |
| Timing Attack | No | ___ | ___ |
| Obfuscated Code | No | ___ | ___ |

**Hypothesis**: SAE probe shows smaller accuracy drop on unseen types

### Experiment 3: Feature Analysis

For the SAE probe, identify which features have highest weight:
- Are these features interpretable?
- Do they correspond to security-relevant concepts?
- Can we understand WHY the probe makes its decisions?

---

## Connection to AI Safety

### The Monitoring Problem

```
┌─────────────────────────────────────────────────────────────────┐
│              CAPABILITY MONITORING CHALLENGE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PROBLEM: Dangerous capabilities may be "dormant"               │
│                                                                  │
│  Model knows how to:                                            │
│  • Write malware (but doesn't show it)                          │
│  • Create bioweapons (but refuses to discuss)                   │
│  • Hack systems (but appears safe in evals)                     │
│                                                                  │
│  SOLUTION: Probe internal representations                       │
│                                                                  │
│  If the model KNOWS something dangerous, its activations        │
│  should encode that knowledge—even if outputs are safe.         │
│                                                                  │
│  SAE probes may be better at extracting this hidden knowledge   │
│  because:                                                       │
│  • Features are disentangled                                    │
│  • Semantic concepts are separated                              │
│  • Less likely to overfit to surface patterns                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Implications for Deployment

If SAE probes are better at detecting hidden capabilities:
1. **Pre-deployment screening**: Scan models for dangerous knowledge
2. **Runtime monitoring**: Flag suspicious internal states
3. **Capability elicitation**: Understand what models really know
4. **Alignment verification**: Check if safety training actually removed capabilities

---

## Risks and Mitigations

| Risk | Level | Mitigation |
|------|-------|------------|
| **No SAE advantage** | Medium | Important negative result; report honestly |
| **Backdoor dataset too easy** | Medium | Include adversarial/obfuscated examples |
| **SAE reconstruction loss hides signal** | Medium | Try multiple SAE layers; compare |
| **Probe overfits to spurious features** | High | Use strong regularization; cross-validation |

---

## Research Questions

### Primary Question
**Do SAE features provide a better basis for detecting "sleeping" capabilities than raw activations?**

### Sub-Questions

1. **Generalization**: Do SAE probes generalize better to novel threats?
2. **Data Efficiency**: Do SAE probes require less training data?
3. **Interpretability**: Can we understand what SAE probes detect?
4. **Practical Value**: Is the improvement large enough to matter in practice?

---

## Key References

1. **"Are SAEs better than probes?"** - Recent MATS work
2. **Representation Engineering** - Zou et al. (2023)
3. **Gemma Scope 2** - DeepMind (2025)
4. **Linear Probes for NLP** - Classic probing literature
5. **A Pragmatic Vision for Interpretability** - Neel Nanda

---

## Timeline (20-Hour Sprint)

| Hours | Focus | Deliverables |
|-------|-------|--------------|
| 1-4 | Dataset Curation | 400+ code snippets with labels |
| 5-8 | Baseline Probes | Standard linear probes trained and evaluated |
| 9-14 | SAE Probes | SAE-based probes; comparison experiments |
| 15-18 | Analysis | Generalization, data efficiency, interpretability |
| 19-20 | Write-up | Executive summary; practical recommendations |

---

## Project Status

**Status**: Research idea documented
**Next Steps**: Code backdoor dataset curation

---

*Project initialized: January 2026*
