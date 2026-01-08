# Strategic Research Roadmap: Mechanistic Analysis of Self-Correction in Open-Weight Models

## Executive Summary

This document outlines the complete research strategy for the MATS Summer 2026 application targeting Neel Nanda's stream. The selected topic—**Mechanistic Decomposition of Chain-of-Thought Self-Correction**—represents the optimal intersection of mentor interest, technical frontier, and methodological pragmatism.

---

## 1. Strategic Context

### 1.1 Why This Topic?

Based on comprehensive analysis of Neel Nanda's recent communications, three vectors converge:

| Vector | Evidence | Alignment |
|--------|----------|-----------|
| **Mentor Interest** | Nanda explicitly asks about "thinking models" and whether reasoning is "structured search or rolling dice" | Direct |
| **Technical Frontier** | Gemma Scope 2 with Transcoders released Dec 2025—cutting-edge tooling | Demonstrated mastery |
| **Pragmatic Focus** | Pivot from "ambitious" to "pragmatic" interpretability—solving real problems | Philosophically aligned |

### 1.2 The Research Pivot

Neel Nanda's research philosophy has evolved:

**Before (2022-2023)**: "Ambitious Mechanistic Interpretability"
- Goal: Fully reverse-engineer neural networks
- Approach: Find all circuits, understand every weight
- Outcome: "Low chance of incredibly big deal"

**Now (2024-2025)**: "Pragmatic Interpretability"
- Goal: Solve problems on critical path to AGI safety
- Approach: Model biology + applied interpretability
- Outcome: "High chance of medium big deal"

**Implication**: Projects should focus on specific, safety-relevant behaviors in real models—not toy circuits in small transformers.

---

## 2. Core Hypothesis

### The Monitor-Inhibition Model

We hypothesize that self-correction comprises two distinct functional components:

**Monitor Circuit** (Detection)
- Located in middle-to-late layers (hypothesized: layers 20-30)
- Tracks logical consistency of current generation
- Outputs error signal when inconsistency detected
- Does NOT generate text—generates internal signal

**Inhibition Circuit** (Action)
- Receives signal from Monitor
- Suppresses "default" (incorrect) continuation
- Boosts correction tokens ("Wait", "Actually", "Let me recheck")
- Redirects generation toward corrective behavior

### Testable Predictions

1. **Existence**: SAE features exist that fire specifically before correction
2. **Necessity**: Ablating these features breaks correction behavior
3. **Sufficiency**: Activating these features induces correction on correct answers
4. **Structure**: Transcoders reveal Monitor → Inhibition causal pathway

---

## 3. Technical Stack

### Required Components

| Component | Version | Purpose |
|-----------|---------|---------|
| PyTorch | 2.0+ | Backend |
| TransformerLens | 2.0+ | Model hooks |
| SAELens | Latest | SAE loading |
| Gemma-2-9B-IT | - | Target model |
| Gemma Scope 2 SAEs | 65k width | Feature extraction |
| Gemma Scope 2 Transcoders | 131k | Circuit tracing |

### Why NOT Train Your Own SAEs

| Factor | Training | Pre-trained |
|--------|----------|-------------|
| Time | Days-weeks | Minutes |
| Compute | Multiple GPUs | Single GPU |
| Risk | Bugs, dead features | Battle-tested |
| Focus | Engineering | Science |

**Strategic choice**: Use Gemma Scope to maximize science time.

---

## 4. Methodology

### Phase 1: Data Generation

**Objective**: Create curated dataset of self-correction instances

**Approach**:
1. Generate 200 candidate prompts (arithmetic, logic, multi-step)
2. Run through Gemma-2-9B-IT with CoT prompting
3. Filter for "Goldilocks" cases:
   - Error in initial reasoning
   - Correction phrase present ("Wait", "Actually")
   - Arrives at correct answer
4. Create matched controls:
   - Correct-No-Correction
   - Incorrect-No-Correction

**Target**: 30-50 clean self-correction examples

### Phase 2: Feature Identification

**Objective**: Find SAE features associated with correction

**Approach**:
1. Run correction examples through model with SAE hooks
2. Record feature activations at critical positions:
   - Token before correction phrase
   - First token of correction phrase
   - Throughout correction span
3. Use attribution patching to rank features
4. Cross-reference with control conditions
5. Interpret features via Neuronpedia

**Metrics**:
- Correlation with correction events
- Uniqueness to correction (vs controls)
- Layer localization

### Phase 3: Causal Validation

**Objective**: Prove causal role of identified features

**Experiments**:

| Experiment | Manipulation | Prediction |
|------------|--------------|------------|
| **Ablation** | Zero top correction features | Correction rate drops ≥30% |
| **Steering** | Clamp features high on correct answers | False correction rate ≥20% |
| **Dose-Response** | Vary ablation strength | Monotonic effect |
| **Specificity** | Ablate random features | No effect on correction |

### Phase 4: Circuit Mapping

**Objective**: Trace Monitor → Inhibition pathway

**Approach**:
1. Use Transcoders to decompose MLP computation
2. Identify input features (Monitor candidates)
3. Identify output features (Inhibition candidates)
4. Map causal connections
5. Visualize as attribution graph

---

## 5. Connection to AI Safety

### The Deception Analogy

| Benign (Self-Correction) | Malign (Deception) |
|--------------------------|-------------------|
| Monitor: "Is my math wrong?" | Monitor: "Am I being watched?" |
| Inhibit: Wrong answer | Inhibit: True goals |
| Output: "Wait, let me recalculate" | Output: Aligned-seeming response |

**Insight**: Understanding benign monitors helps detect malign ones.

### Contribution to Research Agenda

1. **Science of Misalignment**: Characterize internal "checking" mechanisms
2. **Applied Interpretability**: Tools for reasoning quality monitoring
3. **Model Biology**: Document functional circuits in production models

---

## 6. Risk Assessment

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| OOM on 9B | High | Blocks progress | Fall back to 2B |
| Polysemantic features | High | Muddy interpretation | Focus on combinations |
| No clean circuit | Medium | Weaker claims | Report distributed finding |
| Low correction rate | Medium | Insufficient data | Few-shot priming |

### Research Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Null result | Medium | Disappointing | Report honestly as science |
| Features don't transfer | Medium | Limited scope | Acknowledge in limitations |
| Time overrun | Medium | Incomplete work | Prioritize core experiments |

---

## 7. Deliverables

### Application Package

1. **Executive Summary** (1-2 pages)
   - Question → Method → Finding → Implication
   - Standalone readability
   - Specific quantitative claims

2. **Main Report** (8-12 pages)
   - Introduction with motivation
   - Methods with technical detail
   - Results with figures/tables
   - Discussion with limitations
   - Safety implications

3. **Code Release**
   - Reproducible notebooks
   - Clean, documented code
   - Data files

4. **Visualizations**
   - Feature activation heatmaps
   - Ablation/steering bar charts
   - Circuit diagrams

---

## 8. Success Criteria

### Minimum Viable Outcome
- Identify ≥3 candidate correction features
- Document correlation with correction events
- Clear methodology for others to build on

### Target Outcome
- Demonstrate causal necessity (ablation ≥30% effect)
- Demonstrate causal sufficiency (steering ≥20% effect)
- Partial circuit diagram
- Cross-task evidence

### Stretch Outcome
- Full Monitor → Inhibition circuit
- Generalization across model sizes
- Connection to safety mechanisms (refusal overlap?)
- Conference-ready paper

---

## 9. Why This Will Succeed

### Strategic Alignment
- Directly answers Nanda's stated question about "thinking models"
- Uses his team's tooling (Gemma Scope)
- Fits "pragmatic interpretability" philosophy

### Technical Feasibility
- Pre-trained SAEs eliminate engineering risk
- Gemma-2-9B is well-studied model
- Methods proven in prior work

### Scope Appropriateness
- Achievable in 4 weeks
- Clear success metrics
- Graceful degradation on partial success

### Research Taste
- Novel question (CoT self-correction internals)
- Safety-relevant (monitor detection)
- Honest limitations expected

---

## 10. References

### Primary Sources
1. Nanda, N. "MATS Applications + Research Directions I'm Currently Excited About" (2025)
2. Google DeepMind. "Gemma Scope 2 Technical Paper" (2025)
3. "Transcoders enable fine-grained circuit analysis" (2025)

### Methodological Background
4. Anthropic. "On the Biology of a Large Language Model" (2025)
5. "Understanding Refusal in LLMs with SAEs" (2025)
6. "A Pragmatic Vision for Interpretability" (2024)

### Tooling
7. TransformerLens documentation
8. SAELens documentation
9. Neuronpedia

---

*Strategic roadmap created: January 2026*
*Target: MATS Summer 2026 Application*
