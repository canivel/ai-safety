# Four-Week Execution Plan: User Modeling and Sycophancy Research

## Overview

This plan distributes the research across **4 weeks** with clear daily objectives. The structure follows a **two-track approach**: Simple Track (SAEs only) in Weeks 1-2, Advanced Track (Cross-Coders) in Weeks 3-4.

---

## Week 1: Infrastructure & Dataset Creation

**Goal**: Establish stable environment and create comprehensive sycophancy evaluation dataset

### Day 1 (Monday): Environment Setup

| Time Block | Task | Deliverable |
|------------|------|-------------|
| Morning (3h) | Set up compute environment | Working Python environment |
| | Install: `torch`, `transformer_lens`, `sae_lens` | requirements.txt |
| | Verify GPU access and memory | GPU benchmark log |
| Afternoon (3h) | Load Gemma-2-9B-IT (Chat model) | Model loading script |
| | Test basic inference | Sample outputs |
| | Load Gemma Scope SAEs for layers 15-30 | SAE loading verified |

**Checkpoint**: Model + SAEs load without errors

### Day 2 (Tuesday): Dataset Design

| Time Block | Task | Deliverable |
|------------|------|-------------|
| Morning (3h) | Design expertise framing pairs | Template document |
| | Topics: coding, science, history, health | |
| | Create 20 Novice/Expert pair templates | `pair_templates.json` |
| Afternoon (3h) | Design opinion-seeking prompts | |
| | Include incorrect beliefs across domains | `opinion_prompts.json` |
| | Design feedback request prompts | `feedback_prompts.json` |

**Checkpoint**: Prompt templates complete

### Day 3 (Wednesday): Dataset Generation

| Time Block | Task | Deliverable |
|------------|------|-------------|
| Morning (3h) | Generate 40 expertise framing pairs (80 prompts) | `expertise_pairs.json` |
| | Use GPT-4/Claude to expand templates | |
| | Ensure diversity across topics | |
| Afternoon (3h) | Generate 30 opinion-seeking prompts | `opinion_prompts.json` |
| | Generate 20 feedback request prompts | `feedback_prompts.json` |
| | Generate 40 neutral control prompts | `control_prompts.json` |

**Checkpoint**: 170+ prompts generated

### Day 4 (Thursday): Baseline Collection

| Time Block | Task | Deliverable |
|------------|------|-------------|
| Morning (3h) | Run all prompts through Gemma-2-9B-IT | Raw responses |
| | Collect responses for Novice framing | `novice_responses.json` |
| | Collect responses for Expert framing | `expert_responses.json` |
| Afternoon (3h) | Manual inspection of 20 response pairs | Sycophancy examples |
| | Identify clear sycophancy cases | Annotated examples |
| | Document behavioral patterns | Patterns document |

**Checkpoint**: Baseline responses collected, sycophancy confirmed

### Day 5 (Friday): Metrics Implementation

| Time Block | Task | Deliverable |
|------------|------|-------------|
| Morning (3h) | Implement sycophancy metrics | `metrics.py` |
| | Hedging detector (count hedge words) | |
| | Praise detector (sentiment analysis) | |
| | Tone differential calculator | |
| Afternoon (3h) | Run metrics on baseline responses | Baseline metrics |
| | Compute sycophancy scores | `baseline_scores.csv` |
| | Validate metrics against manual labels | Validation report |

**Checkpoint**: Quantified sycophancy baseline

### Weekend: Review & Planning
- Review Week 1 deliverables
- Identify any data gaps
- Prepare for feature analysis phase

---

## Week 2: Feature Analysis (Simple Track)

**Goal**: Identify SAE features that encode user characteristics and correlate with sycophancy

### Day 6 (Monday): Activation Collection

| Time Block | Task | Deliverable |
|------------|------|-------------|
| Morning (3h) | Set up activation extraction pipeline | Extraction code |
| | Hook SAEs at layers 15, 20, 25, 30 | |
| | Test on single prompt pair | Test output |
| Afternoon (3h) | Run all expertise pairs through pipeline | Activation data |
| | Store activations at key positions: | |
| | - User cue position | |
| | - Pre-response position | |
| | - Response positions | `activations.pt` |

**Checkpoint**: Activations collected for all expertise pairs

### Day 7 (Tuesday): Differential Analysis

| Time Block | Task | Deliverable |
|------------|------|-------------|
| Morning (3h) | Compute feature activation differentials | |
| | Δ = Activation(Novice) - Activation(Expert) | `differentials.csv` |
| | For each feature, compute mean Δ | |
| Afternoon (3h) | Rank features by |Δ| magnitude | Feature rankings |
| | Identify top 20 "user cue sensitive" features | Top features list |
| | Statistical significance testing | p-values |

**Checkpoint**: Top 20 candidate features identified

### Day 8 (Wednesday): Feature Interpretation

| Time Block | Task | Deliverable |
|------------|------|-------------|
| Morning (3h) | Look up top 10 features on Neuronpedia | Feature documentation |
| | Examine max-activating examples | |
| | Assign preliminary labels | Feature labels v1 |
| Afternoon (3h) | Cross-reference with sycophancy scores | Correlation analysis |
| | Do high-Δ features correlate with sycophancy? | |
| | Identify "User Model" vs "Topic" features | Refined feature list |

**Checkpoint**: Top 10 features labeled with interpretation

### Day 9 (Thursday): Attribution Analysis

| Time Block | Task | Deliverable |
|------------|------|-------------|
| Morning (3h) | Implement gradient attribution | Attribution code |
| | For sycophantic outputs, which features matter? | |
| | Compare to differential-based ranking | |
| Afternoon (3h) | Implement activation patching | Patching code |
| | Patch "Expert" features into "Novice" prompts | |
| | Measure response change | Patching results |

**Checkpoint**: Causal evidence for feature importance

### Day 10 (Friday): Feature Validation

| Time Block | Task | Deliverable |
|------------|------|-------------|
| Morning (3h) | Test features on held-out prompts | Generalization test |
| | Do features activate on opinion-seeking? | |
| | Do features activate on feedback requests? | |
| Afternoon (3h) | Compile feature dossier | Feature report |
| | Top 5 validated user-model features | |
| | Interpretation + evidence for each | Final feature list |

**Checkpoint**: 5 validated user-modeling features

### Weekend: Analysis & Planning
- Synthesize feature findings
- Prepare intervention experiments
- Draft feature analysis section

---

## Week 3: Causal Interventions

**Goal**: Demonstrate that ablating user-model features reduces sycophancy while preserving helpfulness

### Day 11 (Monday): Ablation Framework

| Time Block | Task | Deliverable |
|------------|------|-------------|
| Morning (3h) | Implement feature ablation utilities | Ablation code |
| | Zero feature activations during forward pass | |
| | Test on single example | |
| Afternoon (3h) | Run ablation on top 5 features (individually) | Individual ablation data |
| | Collect responses for each ablation | |
| | Compare to baseline | Initial ablation results |

**Checkpoint**: Ablation pipeline working

### Day 12 (Tuesday): Sycophancy Reduction

| Time Block | Task | Deliverable |
|------------|------|-------------|
| Morning (3h) | Run full ablation experiment | |
| | All expertise pairs with ablation | `ablated_responses.json` |
| | Compute sycophancy scores | `ablation_scores.csv` |
| Afternoon (3h) | Analyze sycophancy reduction | |
| | Compare Novice-Expert differential | Reduction metrics |
| | Statistical significance testing | |
| | Identify most impactful features | Key features |

**Checkpoint**: Quantified sycophancy reduction

### Day 13 (Wednesday): Helpfulness Preservation

| Time Block | Task | Deliverable |
|------------|------|-------------|
| Morning (3h) | Design helpfulness evaluation | Helpfulness prompts |
| | Simple coding tasks | |
| | Factual Q&A | |
| | Instruction following | |
| Afternoon (3h) | Run helpfulness eval with/without ablation | |
| | Compare accuracy, coherence | Helpfulness scores |
| | Ensure ablation doesn't break model | Preservation report |

**Checkpoint**: Helpfulness preserved (≤10% degradation)

### Day 14 (Thursday): Steering Experiments

| Time Block | Task | Deliverable |
|------------|------|-------------|
| Morning (3h) | Extract "Expert user" activation direction | Steering vector |
| | Average activations on Expert prompts | |
| | Subtract Novice average | |
| Afternoon (3h) | Apply steering to Novice prompts | |
| | Add "Expert" direction during inference | Steered responses |
| | Measure effect on response tone | Steering results |

**Checkpoint**: Steering vector reduces sycophancy

### Day 15 (Friday): Results Compilation

| Time Block | Task | Deliverable |
|------------|------|-------------|
| Morning (3h) | Create summary figures | Figures folder |
| | Sycophancy score bar charts | |
| | Feature activation heatmaps | |
| | Ablation effect plots | |
| Afternoon (3h) | Create summary tables | Tables document |
| | Baseline vs ablated comparison | |
| | Feature properties summary | |
| | Helpfulness metrics | |

**Checkpoint**: All Simple Track results compiled

### Weekend: Analysis & Cross-Coder Prep
- Review intervention results
- Decide: proceed to Cross-Coder or refine Simple Track
- Begin Cross-Coder setup

---

## Week 4: Cross-Coder Analysis & Write-up

**Goal**: Compare Base vs Chat models, write final paper

### Day 16 (Monday): Cross-Coder Setup

| Time Block | Task | Deliverable |
|------------|------|-------------|
| Morning (3h) | Load Gemma-2-9B (Base model) | Base model ready |
| | Load Gemma Scope Cross-Coder | Cross-Coder loaded |
| | Verify setup on test prompt | |
| Afternoon (3h) | Run expertise pairs through Base model | Base responses |
| | Confirm: Base shows less sycophancy? | Base vs Chat comparison |
| | This validates the RLHF hypothesis | |

**Checkpoint**: Base vs Chat behavioral difference confirmed

### Day 17 (Tuesday): Cross-Coder Analysis

| Time Block | Task | Deliverable |
|------------|------|-------------|
| Morning (3h) | Extract Cross-Coder features | |
| | For each expertise pair, get difference features | `cross_coder_features.csv` |
| | Identify Chat-specific features | |
| Afternoon (3h) | Compare to Simple Track findings | |
| | Do Cross-Coder features match SAE features? | Comparison report |
| | Are there new features unique to Cross-Coder? | |

**Checkpoint**: Cross-Coder features identified

### Day 18 (Wednesday): Executive Summary

| Time Block | Task | Deliverable |
|------------|------|-------------|
| Morning (3h) | Draft executive summary (1-2 pages) | `executive_summary.md` v1 |
| | Structure: Context → Hypothesis → Method → Result → Implication | |
| | Focus on key findings only | |
| Afternoon (3h) | Revise and tighten | `executive_summary.md` v2 |
| | Ensure standalone readability | |
| | Get feedback if possible | Final executive summary |

**Checkpoint**: Executive summary complete

### Day 19 (Thursday): Main Report

| Time Block | Task | Deliverable |
|------------|------|-------------|
| Morning (3h) | Write Introduction + Methods | Sections draft |
| | Motivate sycophancy problem | |
| | Describe dataset and methodology | |
| Afternoon (3h) | Write Results + Discussion | Sections draft |
| | Present findings with figures | |
| | Acknowledge limitations | |
| | Safety implications | |

**Checkpoint**: Full paper draft complete

### Day 20 (Friday): Polish & Code Release

| Time Block | Task | Deliverable |
|------------|------|-------------|
| Morning (3h) | Full paper revision | Final paper |
| | Check all claims against evidence | |
| | Proofread thoroughly | |
| Afternoon (3h) | Clean and document code | Code release |
| | Create reproducible notebooks | |
| | Test notebooks end-to-end | |

**Checkpoint**: Paper and code ready

### Weekend: Final Review
- Fresh eyes review of paper
- Final code testing
- Celebrate completion!

---

## Daily Schedule Template

```
┌──────────────────────────────────────────────────────┐
│              TYPICAL RESEARCH DAY                     │
├──────────────────────────────────────────────────────┤
│  09:00 - 09:15  │  Review yesterday's progress       │
│  09:15 - 12:00  │  Morning deep work block           │
│  12:00 - 13:00  │  Lunch + thinking break            │
│  13:00 - 16:00  │  Afternoon deep work block         │
│  16:00 - 16:30  │  Document progress, update notes   │
│  16:30 - 17:00  │  Plan tomorrow's priorities        │
└──────────────────────────────────────────────────────┘
```

---

## Checkpoint Summary

| Week | Day | Key Checkpoint |
|------|-----|----------------|
| 1 | 1 | Model + SAEs loaded |
| 1 | 2 | Prompt templates complete |
| 1 | 3 | 170+ prompts generated |
| 1 | 4 | Baseline responses, sycophancy confirmed |
| 1 | 5 | Sycophancy metrics working |
| 2 | 6 | Activations collected |
| 2 | 7 | Top 20 candidate features |
| 2 | 8 | Top 10 features labeled |
| 2 | 9 | Causal evidence established |
| 2 | 10 | 5 validated user-model features |
| 3 | 11 | Ablation pipeline working |
| 3 | 12 | Sycophancy reduction quantified |
| 3 | 13 | Helpfulness preserved |
| 3 | 14 | Steering vector effective |
| 3 | 15 | All Simple Track results compiled |
| 4 | 16 | Base vs Chat comparison done |
| 4 | 17 | Cross-Coder features identified |
| 4 | 18 | Executive summary complete |
| 4 | 19 | Full paper draft |
| 4 | 20 | Paper + code ready |

---

## Success Metrics by Week

### Week 1: Foundation
- [ ] Environment working
- [ ] 170+ prompts created
- [ ] Sycophancy confirmed in baseline
- [ ] Metrics implemented

### Week 2: Discovery
- [ ] Activations collected
- [ ] Top 20 candidate features
- [ ] Top 5 validated features
- [ ] Feature interpretations documented

### Week 3: Intervention
- [ ] Ablation reduces sycophancy (≥25%)
- [ ] Helpfulness preserved (≤10% degradation)
- [ ] Steering vector effective
- [ ] Results compiled with figures

### Week 4: Synthesis
- [ ] Cross-Coder analysis (if time)
- [ ] Executive summary (1-2 pages)
- [ ] Main report (8-12 pages)
- [ ] Clean, documented code

---

## Risk Mitigation by Week

### Week 1 Risks
| Risk | Mitigation |
|------|------------|
| OOM on 9B | Start with 2B for development |
| SAE loading issues | Check SAELens version |
| Sycophancy not observable | Use clearer expertise cues |

### Week 2 Risks
| Risk | Mitigation |
|------|------------|
| No clear features | Lower significance threshold |
| Features too polysemantic | Focus on feature combinations |
| Attribution too slow | Subsample dataset |

### Week 3 Risks
| Risk | Mitigation |
|------|------------|
| Ablation hurts helpfulness | Use weaker ablation |
| Steering too strong | Tune steering coefficient |
| Results not significant | Report honest null results |

### Week 4 Risks
| Risk | Mitigation |
|------|------------|
| Cross-Coder too complex | Report Simple Track only |
| Time pressure | Prioritize executive summary |
| Code not reproducible | Test on fresh environment |

---

## Deliverables Checklist

### Week 1 Deliverables
- [ ] Working environment
- [ ] 170+ sycophancy prompts
- [ ] Baseline responses
- [ ] Sycophancy metrics

### Week 2 Deliverables
- [ ] Activation extraction pipeline
- [ ] Feature differential analysis
- [ ] Top 5 validated features
- [ ] Feature interpretation document

### Week 3 Deliverables
- [ ] Ablation experiment results
- [ ] Steering experiment results
- [ ] Helpfulness evaluation
- [ ] Figures and tables

### Week 4 Deliverables
- [ ] Cross-Coder analysis (optional)
- [ ] Executive summary (1-2 pages)
- [ ] Main report (8-12 pages)
- [ ] Reproducible code notebooks

---

*Execution plan created: January 2026*
*Duration: 4 weeks (20 working days)*
