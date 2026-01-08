# Cross-Modal Semantics in Gemma 3: Do "Cat" (Text) and "Cat" (Image) Share a Feature?

## Self-Directed Learning Project in Mechanistic Interpretability

**Theme:** Frontier Model Biology / Multimodal
**Difficulty:** High (Novelty)
**Format:** 20-hour research sprint

---

## Executive Summary

This project investigates whether multimodal models learn **"Platonic" representations**—abstract concept features that are shared across modalities. Using Gemma Scope 2 on the multimodal Gemma 3 family, we'll search for SAE features that activate robustly for both text descriptions and image embeddings of the same concept, mapping the geometry of cross-modal integration.

---

## Why This Research Matters

### The Core Question

Do multimodal models:
- **A) Learn separate representations** for text "cat" and image of a cat?
- **B) Learn unified "Platonic" concepts** that fire regardless of modality?

Understanding this has profound implications for:
- How concepts are represented in modern AI systems
- Whether safety interventions on text transfer to images
- The fundamental architecture of multimodal understanding

### Why Neel Nanda is Investing Here

With Gemma Scope 2 covering the multimodal Gemma 3 family, Nanda is explicitly interested in research that "teaches him something new" about these brand-new architectures. Key questions:
- How does multimodal integration actually work internally?
- Are there interpretable cross-modal features?
- What is the "biology" of multimodal models?

This is **basic science on the absolute cutting edge**.

### Alignment with Pragmatic Interpretability

| Research Direction | How This Project Addresses It |
|-------------------|------------------------------|
| **Frontier Model Biology** | First interpretability study on Gemma 3 multimodal |
| **Gemma Scope 2** | Uses DeepMind's latest SAEs on newest architecture |
| **Basic Science** | Investigates fundamental questions about representation |
| **Novel Contribution** | High chance of discovering something new |

---

## Core Hypothesis

**Hypothesis:** There exist "Abstract Concept Features" in late layers of Gemma 3 that fire robustly for both the text token "Eiffel Tower" and an image embedding of the Eiffel Tower.

### The Platonic Representation Model

```
┌─────────────────────────────────────────────────────────────────┐
│              PLATONIC vs. MODALITY-SPECIFIC MODEL                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  MODALITY-SPECIFIC (Hypothesis A)                               │
│  ┌──────────────────────────────────────────┐                   │
│  │                                          │                   │
│  │  Text: "cat" ──► Text-Cat Feature        │                   │
│  │                      ↓                   │                   │
│  │                  [Separate]              │                   │
│  │                      ↑                   │                   │
│  │  Image: 🐱 ───► Image-Cat Feature        │                   │
│  │                                          │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                  │
│  PLATONIC (Hypothesis B - What We're Testing)                   │
│  ┌──────────────────────────────────────────┐                   │
│  │                                          │                   │
│  │  Text: "cat" ──┐                         │                   │
│  │                ├──► Abstract "Cat"       │                   │
│  │  Image: 🐱 ───┘     Feature              │                   │
│  │                                          │                   │
│  │  (Modality-invariant representation)     │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Testable Predictions

1. **Shared Features Exist**: SAE features with high activation for both text and image inputs of same concept
2. **High Mutual Information**: Significant correlation between text and image activations for same concept
3. **Late Layer Convergence**: Cross-modal features more prevalent in later layers
4. **Causal Cross-Modal Steering**: Clamping text-derived features affects image interpretation

---

## Technical Approach

### Methodology

#### Phase 1: Dataset Creation

| Concept Category | Text Input | Image Input | Count |
|-----------------|------------|-------------|-------|
| **Landmarks** | "Eiffel Tower", "Statue of Liberty" | Photos | 20 |
| **Animals** | "cat", "elephant", "penguin" | Photos | 20 |
| **Objects** | "red apple", "vintage car" | Photos | 20 |
| **Scenes** | "beach sunset", "snowy mountain" | Photos | 20 |
| **Abstract** | "happiness", "danger" | Evocative images | 10 |

**Critical Design Choice**: Match text descriptions exactly to image content to ensure fair comparison.

#### Phase 2: Cross-Modal Feature Search

```
┌─────────────────────────────────────────────────────────────────┐
│              CROSS-MODAL FEATURE SEARCH PIPELINE                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INPUT PAIRS: (Text, Image) for same concept                    │
│                                                                  │
│  For each pair:                                                 │
│  ┌─────────────────────────────────────────┐                    │
│  │  1. Run text through Gemma 3            │                    │
│  │  2. Run image through Gemma 3           │                    │
│  │  3. Extract SAE activations (late layers)│                   │
│  │  4. Compute feature overlap             │                    │
│  └─────────────────────────────────────────┘                    │
│                                                                  │
│  METRICS:                                                       │
│  • Jaccard similarity of active features                        │
│  • Correlation of feature activation strengths                  │
│  • Mutual information between modality activations              │
│                                                                  │
│  OUTPUT: Cross-modal feature candidates                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Phase 3: Validation and Intervention

| Experiment | Method | Expected Result |
|------------|--------|-----------------|
| **Concept Specificity** | Do "cat" features fire for cat images but not dog images? | High specificity |
| **Cross-Modal Steering** | Clamp "snow" feature (from text); describe beach image | Model mentions snow/cold? |
| **Layer Analysis** | Compare cross-modal overlap across layers | More overlap in late layers |
| **Baseline Comparison** | Compare random text-image pairs | Low overlap (control) |

### Tooling Stack

| Tool | Purpose |
|------|---------|
| **Gemma 3** | Multimodal target model |
| **Gemma Scope 2** | Pre-trained SAEs for Gemma 3 |
| **TransformerLens** | Model hooking (if compatible) |
| **SAELens** | SAE loading and analysis |
| **Custom scripts** | Multimodal activation extraction |

---

## Success Metrics

### Primary Metric: Cross-Modal Feature Correlation

```
CrossModal_Score = Correlation(SAE_features(text), SAE_features(image))
                   for matched concept pairs
```

### Experiment Success Criteria

| Experiment | Success Criterion |
|------------|-------------------|
| Feature Identification | ≥10 features with >0.6 cross-modal correlation |
| Concept Specificity | Cross-modal features are concept-specific (not generic activation) |
| Steering Effect | Cross-modal steering produces measurable output change |
| Layer Gradient | Clear increase in cross-modal overlap from early to late layers |

### Paper-Ready Outcomes

| Outcome Level | Definition |
|---------------|------------|
| **Minimum** | Document cross-modal activation patterns; methodology |
| **Target** | Identify candidate Platonic features; demonstrate steering |
| **Stretch** | Map geometry of multimodal integration; contribute to Gemma 3 interpretability |

---

## Detailed Experimental Design

### Experiment 1: Cross-Modal Correlation Analysis

For each concept:
1. Collect top-50 active SAE features for text input
2. Collect top-50 active SAE features for image input
3. Compute overlap and correlation

```
Concept         | Text Features | Image Features | Overlap | Correlation
----------------|---------------|----------------|---------|------------
Eiffel Tower    | [f1, f2, ...]| [f3, f4, ...]  | ___     | ___
Cat             | [...]         | [...]          | ___     | ___
Beach Sunset    | [...]         | [...]          | ___     | ___
```

### Experiment 2: Specificity Testing

Do "cat" features fire for dog images?

```
Feature: "Cat-candidate-feature-42"

Activation on:
- Text "cat": ___
- Image of cat: ___
- Text "dog": ___
- Image of dog: ___

Specificity = (cat_activation) / (cat_activation + dog_activation)
```

### Experiment 3: Cross-Modal Steering

```
┌─────────────────────────────────────────────────────────────────┐
│              CROSS-MODAL STEERING EXPERIMENT                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  SETUP:                                                         │
│  • Input: Image of sunny beach                                  │
│  • Task: "Describe this image"                                  │
│                                                                  │
│  BASELINE: Normal description                                   │
│  "A beautiful sunny beach with blue water and sand..."          │
│                                                                  │
│  INTERVENTION: Clamp "Snow" feature (derived from text)         │
│  Expected: Description mentions cold/winter elements?           │
│  "A beach scene... the sand looks almost like snow..."          │
│                                                                  │
│  SUCCESS: Text-derived features causally affect image processing│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Connection to AI Safety

### Why Cross-Modal Representations Matter for Safety

```
┌─────────────────────────────────────────────────────────────────┐
│              SAFETY IMPLICATIONS                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  IF concepts are shared across modalities:                      │
│                                                                  │
│  ✓ Safety interventions on text may transfer to images          │
│  ✓ A "violence" feature ablated from text also blocks           │
│    violent image generation/interpretation                       │
│  ✓ Monitoring systems can use unified detectors                 │
│                                                                  │
│  IF concepts are modality-specific:                             │
│                                                                  │
│  ✗ Need separate safety measures for each modality              │
│  ✗ Jailbreaks via modality switching may be easier              │
│  ✗ More complex monitoring requirements                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Implications for Multimodal Safety

1. **Transfer of Safety Training**: Do text-based safety interventions protect against image-based attacks?
2. **Unified Monitoring**: Can we build one monitor for all modalities?
3. **Attack Surface**: Are cross-modal inconsistencies exploitable?
4. **Representation Alignment**: Are multimodal models more or less aligned across modalities?

---

## Risks and Mitigations

| Risk | Level | Mitigation |
|------|-------|------------|
| **Gemma 3 SAEs not yet available** | Medium | Use available layers; contribute early findings |
| **Architecture complexity** | High | Focus on late layers where integration happens |
| **No cross-modal features found** | Medium | Important negative result; may indicate modality-specific processing |
| **Tooling compatibility** | Medium | May need custom activation extraction |

---

## Research Questions

### Primary Question
**Do multimodal models learn unified "Platonic" concept representations that are shared across text and image modalities?**

### Sub-Questions

1. **Feature Existence**: Are there SAE features that fire for both text and image inputs of the same concept?
2. **Layer Distribution**: At which layers does cross-modal integration occur?
3. **Causal Role**: Do text-derived features causally influence image processing (and vice versa)?
4. **Geometry**: What is the structure of the multimodal representation space?

---

## Key References

1. **Gemma Scope 2** - DeepMind (2025)
2. **"The Platonic Representation Hypothesis"** - Recent multimodal work
3. **CLIP and Vision-Language Models** - Cross-modal representation literature
4. **Gemma 3 Technical Report** - Google DeepMind
5. **A Pragmatic Vision for Interpretability** - Neel Nanda

---

## Timeline (20-Hour Sprint)

| Hours | Focus | Deliverables |
|-------|-------|--------------|
| 1-3 | Dataset Creation | 90+ matched text-image pairs |
| 4-8 | Activation Extraction | Cross-modal feature activations |
| 9-14 | Correlation Analysis | Cross-modal feature candidates |
| 15-18 | Steering Experiments | Causal verification |
| 19-20 | Write-up | Findings; geometry visualization; implications |

---

## Project Status

**Status**: Research idea documented
**Next Steps**: Verify Gemma Scope 2 availability for Gemma 3; begin dataset curation

---

*Project initialized: January 2026*
