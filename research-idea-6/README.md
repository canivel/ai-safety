# From Inference to Pandering: Circuit Analysis of Implicit User Modeling and Sycophancy

## Self-Directed Learning Project in Mechanistic Interpretability

**Theme:** Model Biology / Science of Misalignment / Applied Interpretability
**Difficulty:** High (Technical & Conceptual)
**Format:** 20-hour research sprint

---

## Executive Summary

This project investigates the **mechanistic circuitry** by which language models infer user attributes (e.g., gender) from implicit cues and use those inferences to modulate responses—potentially in sycophantic or stereotyped ways. Using Gemma Scope Cross-Coders, we aim to identify "User Modeling" features that are learned during chat training and trace how they causally influence "Opinion" features to produce pandering behavior.

**Title:** *From Inference to Pandering: Circuit Analysis of Implicit User Modeling and Sycophancy using Gemma Scope Cross-Coders*

---

## Why This Research Matters

### The Core Problem

Modern chat models don't just answer questions—they **model the user**. They infer attributes like:
- Gender ("As a mother of three...")
- Age ("Back in my day...")
- Expertise level ("I'm new to programming...")
- Emotional state ("I'm really frustrated...")

This user modeling can lead to **sycophancy**: the model altering its answers to match perceived user preferences or stereotypes rather than providing accurate information.

### Why Neel Nanda is Investing Here

Nanda has explicitly highlighted "modeling of the user" as a behavior he is excited to investigate:
> "I am looking for latents related to especially interesting related chat-model behaviors (e.g., modeling of the user) in an unsupervised fashion."

This connects directly to:
- **Science of Misalignment**: User-based manipulation is a safety failure
- **Cross-Coders**: Perfect for comparing Base vs. Chat models
- **Model Biology**: Understanding how RLHF creates user modeling circuits

### The Trap to Avoid

**Do NOT** just train a linear probe to predict user gender from activations. This has been done (e.g., ICML 2025 paper "What Kind of User Are You?"). A winning proposal must go beyond proving the phenomenon exists—it must investigate the **mechanism** and **downstream effects**.

### Alignment with Pragmatic Interpretability

| Research Direction | How This Project Addresses It |
|-------------------|------------------------------|
| **Model Biology** | Dissects user modeling circuits |
| **Cross-Coders** | Uses frontier technique for Base vs. Chat comparison |
| **Science of Misalignment** | Investigates sycophancy mechanism |
| **Applied Interpretability** | Directly relevant to deployed chat systems |

---

## Core Hypothesis

**Hypothesis:** Models do not just passively represent user gender; they have active "Sycophancy Circuits" where "User Attribute" features (e.g., "User is Female") causally modulate "Opinion" features to match perceived stereotypes.

### The User Modeling → Sycophancy Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│           USER MODELING → SYCOPHANCY CIRCUIT                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INPUT: "As a mother of three, I'm thinking about buying a car" │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────┐                   │
│  │       USER INFERENCE CIRCUIT              │                   │
│  │  (Hypothesized: Chat-specific features)   │                   │
│  │                                          │                   │
│  │  Implicit cue: "mother of three"         │                   │
│  │  Inferred: User is female, parent        │                   │
│  │  Features: [User-Female], [User-Parent]  │                   │
│  └──────────────────────────────────────────┘                   │
│                              │                                   │
│                    User Attribute Signal                        │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────┐                   │
│  │       SYCOPHANCY/PANDERING CIRCUIT        │                   │
│  │                                          │                   │
│  │  User-Female feature MODULATES:          │                   │
│  │  • Topic emphasis (safety > speed)       │                   │
│  │  • Recommendation style                  │                   │
│  │  • Tone and metaphors                    │                   │
│  └──────────────────────────────────────────┘                   │
│                              │                                   │
│                              ▼                                   │
│  OUTPUT: Stereotyped response (e.g., emphasizes minivans)       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Research Questions

1. **Does the Chat model have dedicated "User Gender" features that the Base model lacks?**
   - This would be a strong "Model Diffing" result showing RLHF creates user modeling circuits

2. **Do these features causally influence downstream responses?**
   - Can we induce stereotyped responses by clamping user attribute features?

3. **What is the circuit architecture?**
   - Which attention heads read from user inference features and write to response generation?

### Testable Predictions

1. **Chat-Specific Features**: Cross-Coders reveal "User Attribute" features present in Chat model but absent in Base model
2. **Causal Influence**: Clamping "User is Female" feature changes response content for male users
3. **Stereotype Modulation**: User attribute features correlate with opinion/recommendation features for stereotyped topics
4. **Identifiable Read Heads**: Specific attention heads move user attribute information to response generation

---

## Technical Approach

### Methodology

#### Phase 1: Dataset Construction (Implicit vs. Explicit)

**Critical Design Choice**: Use IMPLICIT gender cues, not explicit statements.

| Cue Type | Example | Why Important |
|----------|---------|---------------|
| **Implicit Female** | "As a mother of three...", "My husband and I..." | Tests inference, not keyword matching |
| **Implicit Male** | "As a father...", "My wife and I..." | Matched control |
| **Neutral** | "I'm thinking about..." | Baseline without gender cues |

**Opinion Questions** (where stereotypes might influence answers):

| Category | Example Questions |
|----------|------------------|
| **Cars** | "What car should I buy?" (SUV/minivan vs. sports car) |
| **Fashion** | "What should I wear to an interview?" |
| **Parenting** | "How should I handle my child's behavior?" |
| **Career** | "Should I ask for a raise?" |
| **Hobbies** | "What hobby should I pick up?" |

#### Phase 2: Feature Hunting with Cross-Coders

```
┌─────────────────────────────────────────────────────────────────┐
│              CROSS-CODER ANALYSIS PIPELINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  STEP 1: Compare Base vs. Chat Model Activations                │
│  ┌─────────────────────────────────────────┐                    │
│  │  Same prompt → Both models              │                    │
│  │  Base Model: Gemma-2-9B                 │                    │
│  │  Chat Model: Gemma-2-9B-IT              │                    │
│  └─────────────────────────────────────────┘                    │
│                                                                  │
│  STEP 2: Cross-Coder Feature Comparison                         │
│  ┌─────────────────────────────────────────┐                    │
│  │  Find features that:                    │                    │
│  │  • Fire strongly in Chat on gender cues │                    │
│  │  • Are ABSENT or weak in Base           │                    │
│  │  → "Chat-Specific User Modeling" features│                   │
│  └─────────────────────────────────────────┘                    │
│                                                                  │
│  STEP 3: Feature-Response Correlation                           │
│  ┌─────────────────────────────────────────┐                    │
│  │  Correlate user attribute features with │                    │
│  │  downstream response content features   │                    │
│  │  → Evidence for causal pathway          │                    │
│  └─────────────────────────────────────────┘                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Phase 3: Causal Intervention (The "Wow" Factor)

**Experiment A: Cross-Gender Steering**
```
Setup:
- Input: Male user cue ("As a father...")
- Intervention: Clamp "User is Female" feature
- Measure: Does response shift toward female stereotypes?

Expected Results:
- Car recommendations shift toward minivans/SUVs
- Tone/language becomes more stereotypically "female-directed"
- Topic emphasis changes
```

**Experiment B: Circuit Tracing**
```
Goal: Find the "read head" that connects user inference to response

Method:
1. Identify position where user gender is inferred (implicit cue token)
2. Identify position where response is generated (opinion tokens)
3. Search for attention heads that:
   - Attend FROM response position TO user cue position
   - Move "User Attribute" information

This maps the information flow of sycophancy.
```

### Tooling Stack

| Tool | Purpose |
|------|---------|
| **Gemma-2-9B** | Base model (no user modeling training) |
| **Gemma-2-9B-IT** | Chat model (RLHF-trained user modeling) |
| **Gemma Scope 2** | Pre-trained SAEs and Cross-Coders |
| **TransformerLens** | Model hooking and intervention |
| **SAELens** | SAE loading and feature analysis |

---

## Success Metrics

### Primary Metric: Sycophancy Modulation

```
Sycophancy_Effect = P(stereotyped_response | clamped_feature) -
                    P(stereotyped_response | baseline)
```

### Experiment Success Criteria

| Experiment | Success Criterion |
|------------|-------------------|
| Chat-Specific Features | ≥3 features that fire on gender cues in Chat but not Base |
| Cross-Gender Steering | ≥20% shift in stereotyped response content |
| Feature-Response Correlation | >0.5 correlation between user features and opinion features |
| Circuit Identification | Identify ≥1 attention head in the user→response pathway |

### Paper-Ready Outcomes

| Outcome Level | Definition |
|---------------|------------|
| **Minimum** | Identify Chat-specific user modeling features; document methodology |
| **Target** | Demonstrate causal steering; map partial circuit |
| **Stretch** | Full sycophancy circuit diagram; cross-attribute generalization (age, expertise) |

---

## Detailed Experimental Design

### Experiment 1: Chat vs. Base Feature Comparison

| Feature Type | Base Model | Chat Model | Interpretation |
|--------------|------------|------------|----------------|
| "User is Female" | Absent/Weak | Strong | RLHF learned user modeling |
| "User is Parent" | Absent/Weak | Strong | RLHF learned user modeling |
| "General Gender" | Present | Present | Pre-existing knowledge |
| "Topic: Cars" | Present | Present | Pre-existing knowledge |

**Key Finding**: Chat-specific features = Evidence that RLHF creates user modeling circuits

### Experiment 2: Steering Matrix

```
┌─────────────────────────────────────────────────────────────────┐
│                    STEERING EXPERIMENT MATRIX                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│               │  No Steering  │ Clamp "Female" │ Clamp "Male"  │
│  ─────────────┼───────────────┼────────────────┼───────────────│
│  Female Cue   │  Baseline     │  Amplify?      │  Counter?     │
│  ("mother")   │  (stereotyped)│                │               │
│  ─────────────┼───────────────┼────────────────┼───────────────│
│  Male Cue     │  Baseline     │  Cross-gender  │  Amplify?     │
│  ("father")   │  (stereotyped)│  steering test │               │
│  ─────────────┼───────────────┼────────────────┼───────────────│
│  Neutral      │  Baseline     │  Induce female │  Induce male  │
│  Cue          │  (balanced)   │  stereotypes   │  stereotypes  │
│  ─────────────┴───────────────┴────────────────┴───────────────│
│                                                                  │
│  SUCCESS: Cross-gender steering produces measurable shift       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Experiment 3: Circuit Tracing

```
┌─────────────────────────────────────────────────────────────────┐
│                    CIRCUIT TRACING                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Token Positions:                                               │
│                                                                  │
│  "As a mother of three, what car should I buy?"                 │
│       ↑                              ↑                           │
│    [Gender Cue]                 [Response Start]                │
│    Position 3                   Position 10+                     │
│                                                                  │
│  Question: Which attention heads connect these positions?       │
│                                                                  │
│  Method: Activation patching                                    │
│  1. Run with female cue, cache "User-Female" feature at pos 3   │
│  2. Run with male cue, patch in female feature at pos 3         │
│  3. For each attention head, measure response change            │
│  4. Heads with largest effect = "User Modeling Read Heads"      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Connection to AI Safety

### The Sycophancy Problem

Sycophancy is a core safety concern because:
1. **Manipulation**: Model tells users what they want to hear, not what's true
2. **Stereotype Reinforcement**: Model amplifies societal biases
3. **Trust Erosion**: Users can't rely on model for objective information
4. **Hidden Behavior**: Users may not realize they're being pandered to

### How This Research Helps

| Finding | Safety Implication |
|---------|-------------------|
| **User Modeling Circuits Exist** | Know what to monitor/ablate |
| **RLHF Creates These Circuits** | Training procedure creates safety risk |
| **Causal Pathway Identified** | Can intervene to reduce sycophancy |
| **Cross-Coder Methodology Works** | Scalable approach for other behaviors |

### Broader Applications

The methodology transfers to other user modeling behaviors:
- **Expertise Inference**: Does the model dumb down for "novices"?
- **Emotional State**: Does the model validate emotions over facts?
- **Political Affiliation**: Does the model tailor political content?

---

## Risks and Mitigations

| Risk | Level | Mitigation |
|------|-------|------------|
| **No Chat-specific features found** | Medium | Important negative result; may mean user modeling is emergent |
| **Steering effects too subtle** | High | Use extreme stereotype topics; measure multiple output dimensions |
| **Confounding with other variables** | Medium | Careful dataset design; matched controls |
| **Cross-Coders not available** | Low | Fall back to standard SAE comparison |
| **Ethical concerns** | Low | Focus on understanding mechanism, not amplifying harm |

---

## Research Questions

### Primary Question
**How do chat models mechanistically infer user attributes from implicit cues, and how do these inferences causally modulate response content (sycophancy)?**

### Sub-Questions

1. **Feature Existence**: Are there Chat-specific "User Attribute" features?
2. **Causal Role**: Do these features causally influence responses?
3. **Circuit Architecture**: What attention heads connect user inference to response?
4. **RLHF Effect**: Does chat training create or amplify user modeling circuits?

---

## What Makes This "MATS-Ready"

| Criterion | How This Proposal Meets It |
|-----------|---------------------------|
| **Novel Angle** | Circuit analysis, not just probing |
| **Uses Frontier Tools** | Cross-Coders for Base vs. Chat comparison |
| **Safety Relevance** | Directly addresses sycophancy |
| **Teaches Something New** | Mechanism of RLHF-induced user modeling |
| **Feasible in 20 Hours** | Focused scope with clear deliverables |
| **Evidence For/Against** | Hypothesis-driven with falsifiable predictions |

---

## Key References

1. **"What Kind of User Are You?"** - ICML 2025 (prior work on user modeling)
2. **Cross-Coders** - Recent interpretability technique
3. **Gemma Scope 2** - DeepMind (2025)
4. **"Towards Understanding Sycophancy in LLMs"** - Anthropic
5. **A Pragmatic Vision for Interpretability** - Neel Nanda
6. **Representation Engineering** - Zou et al. (2023)

---

## Timeline (20-Hour Sprint)

| Hours | Focus | Deliverables |
|-------|-------|--------------|
| 1-4 | Dataset Construction | 100+ prompts with implicit gender cues |
| 5-10 | Cross-Coder Feature Hunting | Chat-specific user modeling features |
| 11-15 | Steering Experiments | Causal verification; stereotyping modulation |
| 16-18 | Circuit Tracing | Identify user→response attention heads |
| 19-20 | Write-up | Executive summary; methodology; safety implications |

---

## Project Status

**Status**: Research idea documented
**Next Steps**: Dataset construction with implicit gender cues

---

*Project initialized: January 2026*
