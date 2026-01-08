# Mechanistic Decomposition of Chain-of-Thought Reasoning in Open-Weight Models

## Self-Directed Learning Project in Mechanistic Interpretability

> **Note**: This project was originally planned as an application for Neel Nanda's MATS Stream (Summer 2026). Unfortunately, I was unable to complete the work within the application timeline. However, since all the research documentation, methodology, and resources are available, I'm continuing this as a **self-directed learning project** for curiosity and skill development in mechanistic interpretability.

---

## Executive Summary

This project investigates the **internal mechanisms of self-correction in Chain-of-Thought (CoT) reasoning** using Gemma Scope 2's Sparse Autoencoders (SAEs) and Transcoders. We aim to identify and characterize the "Monitor-Inhibition" circuit that enables language models to detect errors and trigger correction behaviors (e.g., "Wait, that's wrong...").

**Title:** *Tracing the "Aha!" Moment: Circuit Analysis of Self-Correction in Chain-of-Thought Reasoning using Gemma Scope Transcoders*

---

## Why This Research Matters

### The Problem
Modern "thinking models" (OpenAI o1, DeepSeek-R1) demonstrate remarkable self-correction abilities during extended reasoning chains. But we don't understand **how** this works internally:
- Does the model engage in structured search with backtracking?
- Or is it "rolling the dice" and getting lucky?
- What internal signals trigger the "Wait, that's wrong" moment?

### The Safety Implications
Understanding self-correction mechanisms directly connects to **deceptive alignment** concerns:
- If models have internal "monitors" that check reasoning correctness
- The same architecture might check "Am I being watched?" for deception
- Characterizing benign monitors helps us detect malign ones

### Alignment with Current Interpretability Research
| Research Direction | How This Project Addresses It |
|----------|------------------------------|
| **"Thinking Models"** | Directly investigates CoT reasoning mechanics |
| **Pragmatic Interpretability** | Uses tools to solve concrete safety problems |
| **Model Biology** | Qualitative dissection of specific behaviors |
| **Gemma Scope 2** | Leverages DeepMind's latest SAEs and Transcoders |
| **Applied Interpretability** | Demonstrates causal intervention capabilities |

---

## Core Hypothesis: The Monitor-Inhibition Model

We hypothesize that self-correction involves **two distinct functional components**:

```
┌─────────────────────────────────────────────────────────────────┐
│                     MONITOR-INHIBITION MODEL                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: "23 × 41 = 923" (incorrect)                             │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────┐                   │
│  │          MONITOR CIRCUIT                  │                   │
│  │  (Layers 20-30)                          │                   │
│  │                                          │                   │
│  │  • Tracks logical consistency            │                   │
│  │  • Detects arithmetic errors             │                   │
│  │  • Fires "inconsistency" features        │                   │
│  └──────────────────────────────────────────┘                   │
│                              │                                   │
│                    Error Signal                                  │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────┐                   │
│  │      INHIBITION/STEERING CIRCUIT          │                   │
│  │                                          │                   │
│  │  • Suppresses "default" (wrong) token    │                   │
│  │  • Boosts correction tokens              │                   │
│  │  • Outputs: "Wait", "Actually", etc.     │                   │
│  └──────────────────────────────────────────┘                   │
│                              │                                   │
│                              ▼                                   │
│  Output: "Wait, let me recalculate..."                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Testable Predictions

1. **Identifiable Features**: We can find SAE features that consistently activate before correction tokens
2. **Causal Necessity**: Ablating these features breaks self-correction (model outputs wrong answer confidently)
3. **Causal Sufficiency**: Artificially activating these features induces false corrections (model doubts correct answers)
4. **Circuit Structure**: Transcoders reveal causal links between Monitor and Inhibition components

---

## Technical Approach

### Tooling Stack

| Tool | Purpose | Source |
|------|---------|--------|
| **Gemma-2-9B-IT** | Target model (instruction-tuned for CoT) | Google DeepMind |
| **Gemma Scope 2** | Pre-trained SAEs and Transcoders | Google DeepMind |
| **TransformerLens** | Model hooking and intervention | Neel Nanda |
| **SAELens** | SAE loading and analysis | MATS ecosystem |
| **Neuronpedia** | Feature visualization and interpretation | Community |

### Why Gemma Scope 2?

Gemma Scope 2 provides **pre-trained interpretability artifacts** for the entire Gemma 2 family:
- SAEs for every layer (residual stream, attention, MLP)
- **Transcoders** for MLP computation decomposition
- JumpReLU activation for strict sparsity
- Open-source and well-documented

**Strategic Advantage**: We skip months of SAE training and jump straight to biology.

### Methodology Overview

```
Phase 1: Setup & Data          Phase 2: Feature Hunting
┌─────────────────────┐       ┌─────────────────────┐
│ • Load Gemma-2-9B   │       │ • Run correction    │
│ • Load Gemma Scope  │  ──►  │   prompts through   │
│ • Generate self-    │       │   model + SAEs      │
│   correction dataset│       │ • Attribution       │
│ • Establish metrics │       │   patching to find  │
└─────────────────────┘       │   key features      │
                              └─────────────────────┘
                                        │
                                        ▼
Phase 4: Write-up              Phase 3: Intervention
┌─────────────────────┐       ┌─────────────────────┐
│ • Executive summary │       │ • Ablation: Break   │
│ • Full report with  │  ◄──  │   correction        │
│   graphs & tables   │       │ • Steering: Induce  │
│ • Code release      │       │   false corrections │
│ • Limitations       │       │ • Cross-task tests  │
└─────────────────────┘       └─────────────────────┘
```

---

## Success Metrics

### Primary Metric: Correction Logit Difference

```
Correction_Score = Logit("Wait") - Logit([Incorrect_Token])
```

This quantifies how strongly the model prefers correction over continuing the error.

### Experiment Success Criteria

| Experiment | Success Criterion |
|------------|-------------------|
| Feature Identification | ≥3 features with >0.7 correlation to correction events |
| Ablation (Suppress) | ≥30% reduction in correction rate |
| Steering (Induce) | ≥20% increase in false correction rate |
| Circuit Mapping | Identifiable causal path in Transcoder graph |

### Paper-Ready Outcomes

| Outcome Level | Definition |
|---------------|------------|
| **Minimum** | Identify candidate Monitor features; document methodology |
| **Target** | Demonstrate causal ablation/steering; map partial circuit |
| **Stretch** | Full circuit diagram; cross-task generalization; safety implications |

---

## Research Questions

### Primary Question
**What are the internal mechanisms that enable language models to detect and correct their own reasoning errors during Chain-of-Thought?**

### Sub-Questions

1. **Detection Mechanism**
   - What features activate when the model "notices" an error?
   - Are these features task-specific (math) or general (reasoning)?
   - At which layers does error detection occur?

2. **Correction Mechanism**
   - How does error detection translate to behavioral change?
   - What suppresses the "default" (wrong) continuation?
   - What boosts correction tokens ("Wait", "Actually")?

3. **Circuit Architecture**
   - Is there a clean Monitor → Inhibition pathway?
   - Does this overlap with refusal/safety circuits?
   - How distributed is the computation?

4. **Generalization**
   - Do the same features fire for math vs. logic vs. coding errors?
   - Does the circuit scale across model sizes (2B → 9B → 27B)?

---

## Dataset Design

### Self-Correction Prompt Categories

| Category | Example | Purpose |
|----------|---------|---------|
| **Arithmetic** | "What is 23 × 41? Think step by step." | Clear ground truth |
| **Logic** | "If A implies B, and B is false, what can we conclude about A?" | Reasoning chains |
| **Factual** | "What year did the Berlin Wall fall? Let me think..." | Knowledge + reasoning |
| **Multi-step** | "Solve: 3x + 7 = 22. Show your work." | Extended CoT |

### Data Collection Strategy

1. **Generate 200 candidate prompts** using GPT-4 or Claude
2. **Run through Gemma-2-9B-IT** with CoT prompting
3. **Filter for "Goldilocks" cases**:
   - Model initially makes error
   - Model writes correction phrase ("Wait", "Actually", "Let me recheck")
   - Model arrives at correct answer
4. **Target: 30-50 clean self-correction examples**

### Control Conditions

| Condition | Description |
|-----------|-------------|
| **Correct-No-Correction** | Model gets it right first try |
| **Incorrect-No-Correction** | Model wrong and doesn't correct |
| **Correction-Clean** | Clear self-correction (target) |
| **False-Correction** | Model "corrects" right answer to wrong |

---

## Connection to AI Safety

### The Deceptive Alignment Link

The Monitor-Inhibition model maps directly to deception concerns:

| Benign (Our Study) | Malign (Safety Concern) |
|-------------------|------------------------|
| Monitor: "Is my math wrong?" | Monitor: "Am I being tested?" |
| Inhibition: Suppress wrong answer | Inhibition: Suppress true goals |
| Output: "Wait, let me recalculate" | Output: Appear aligned |

**If we can find the benign monitor, we may detect the malign one.**

### Contribution to AI Safety Research

This project contributes to:
1. **Science of Misalignment**: Understanding how models internally represent "checking"
2. **Applied Interpretability**: Tools to monitor reasoning quality
3. **Model Biology**: Cataloging functional circuits in production models

---

## Risks and Mitigations

| Risk | Level | Mitigation |
|------|-------|------------|
| **OOM on 9B model** | High | Fall back to Gemma-2-2B; methodology > scale |
| **Polysemantic features** | High | Focus on feature combinations; acknowledge in write-up |
| **No clean circuit** | Medium | Report "distributed" finding; pivot to attention analysis |
| **Insufficient corrections** | Medium | Use few-shot priming to induce correction behavior |
| **Features don't transfer** | Medium | Report negative result honestly; it's still science |

---

## Expected Contributions

### Scientific Contributions
1. First mechanistic characterization of self-correction in open-weight models
2. Novel application of Transcoders to reasoning analysis
3. Evidence for/against the Monitor-Inhibition hypothesis

### Practical Contributions
1. Reproducible methodology for reasoning circuit analysis
2. Feature catalog for correction-related computations
3. Intervention techniques for steering correction behavior

### Safety Contributions
1. Framework for detecting internal "checking" mechanisms
2. Connection between benign and potentially deceptive monitors
3. Tools for reasoning quality assurance in deployed systems

---

## Repository Structure

```
Neel_Nanda/
├── README.md                           # This file
├── docs/
│   ├── research_plan.md               # Original strategic analysis
│   ├── four_week_plan.md              # Detailed execution timeline
│   ├── topic_evaluation.md            # Topic selection rationale
│   └── neel_nanda_research_profile.md # Mentor background
├── resources/
│   ├── key_references.md              # Papers and links
│   ├── setup_guide.md                 # Environment configuration
│   └── gemma_scope_guide.md           # Gemma Scope 2 specifics
├── experiments/
│   ├── notebooks/
│   │   ├── 01_environment_setup.ipynb
│   │   ├── 02_data_generation.ipynb
│   │   ├── 03_feature_analysis.ipynb
│   │   ├── 04_intervention_experiments.ipynb
│   │   └── 05_circuit_mapping.ipynb
│   ├── data/
│   │   ├── correction_prompts.json
│   │   └── control_prompts.json
│   ├── results/
│   │   └── .gitkeep
│   └── figures/
│       └── .gitkeep
└── paper/
    ├── executive_summary.md
    ├── main_report.md
    └── appendix.md
```

---

## Key References

### Core Methodology
1. **Gemma Scope 2 Technical Paper** - DeepMind (2025)
2. **Transcoders enable fine-grained circuit analysis** - LessWrong (2025)
3. **On the Biology of a Large Language Model** - Anthropic (2025)

### Neel Nanda's Guidance
4. **MATS Applications + Research Directions** - LessWrong
5. **How Can Interpretability Researchers Help AGI Go Well?** - Alignment Forum
6. **A Pragmatic Vision for Interpretability** - LessWrong

### Technical Background
7. **Gemma Scope: Open Sparse Autoencoders** - arXiv (2024)
8. **Understanding Refusal in LLMs with SAEs** - EMNLP (2025)

---

## Timeline Overview

| Week | Focus | Key Deliverables |
|------|-------|------------------|
| **Week 1** | Setup + Data | Environment ready, 50 correction examples |
| **Week 2** | Feature Analysis | Top 10 candidate features identified |
| **Week 3** | Interventions | Ablation/steering results quantified |
| **Week 4** | Write-up + Polish | Executive summary, full report, code |

**See [docs/four_week_plan.md](docs/four_week_plan.md) for detailed daily breakdown.**

---

## Resources & References

- **Gemma Scope 2**: [Alignment Forum Announcement](https://www.alignmentforum.org/posts/YQro5LyYjDzZrBCdb/announcing-gemma-scope-2)
- **Neuronpedia**: [neuronpedia.org](https://neuronpedia.org)
- **TransformerLens**: [github.com/neelnanda-io/TransformerLens](https://github.com/neelnanda-io/TransformerLens)
- **SAELens**: [github.com/jbloomAus/SAELens](https://github.com/jbloomAus/SAELens)
- **MATS Program**: [matsprogram.org](https://matsprogram.org)
- **Neel Nanda's Research**: [neelnanda.io](https://neelnanda.io)

---

## Project Status

**Status**: Self-directed learning project (ongoing)

**Original goal**: MATS Summer 2026 application
**Current goal**: Personal skill development in mechanistic interpretability

All research planning, methodology, and documentation remain valid and can be followed at your own pace. The 4-week timeline can be adapted as needed for self-study.

---

*Project initialized: January 2026*
*Updated: Converted to self-directed learning project*
