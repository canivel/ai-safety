# Four-Week Research Execution Plan

## Mechanistic Decomposition of Chain-of-Thought Self-Correction

---

## Overview

This plan distributes the research across **4 weeks** with clear daily objectives, deliverables, and checkpoints. The structure prioritizes:

1. **Front-loading infrastructure** to maximize science time
2. **Parallel workstreams** where possible
3. **Built-in buffer time** for unexpected challenges
4. **Continuous documentation** for paper writing

---

## Week 1: Infrastructure & Data Generation

**Goal**: Establish stable environment and generate high-quality self-correction dataset

### Day 1 (Monday): Environment Setup

| Time Block | Task | Deliverable |
|------------|------|-------------|
| Morning (3h) | Set up compute environment (Colab Pro+ or local GPU) | Working Python environment |
| | Install core libraries: `torch`, `transformer_lens`, `sae_lens` | Requirements.txt |
| | Verify CUDA/GPU access | GPU benchmark log |
| Afternoon (3h) | Load Gemma-2-9B-IT model | Model loading script |
| | Run basic inference test | Sample outputs |
| | Document any version conflicts | Setup troubleshooting notes |

**Checkpoint**: Can run `model.generate()` on Gemma-2-9B-IT

### Day 2 (Tuesday): Gemma Scope Integration

| Time Block | Task | Deliverable |
|------------|------|-------------|
| Morning (3h) | Download Gemma Scope 2 SAEs for layers 20-30 | SAE artifacts |
| | Load SAEs using SAELens | SAE loading script |
| | Test SAE reconstruction on sample prompts | Reconstruction loss metrics |
| Afternoon (3h) | Download Transcoders for target layers | Transcoder artifacts |
| | Verify Transcoder integration | Integration test |
| | Measure reconstruction fidelity | Fidelity report |

**Checkpoint**: SAEs and Transcoders loaded, reconstruction loss < 5%

### Day 3 (Wednesday): Dataset Design & Generation

| Time Block | Task | Deliverable |
|------------|------|-------------|
| Morning (3h) | Design prompt templates for self-correction | Prompt template document |
| | Categories: arithmetic, logic, multi-step, factual | |
| | Create 200 candidate prompts (use GPT-4/Claude for generation) | `candidate_prompts.json` |
| Afternoon (3h) | Run Gemma-2-9B-IT on all prompts with CoT | Raw outputs |
| | Log all generations with metadata | `raw_generations.json` |
| | Begin manual labeling of correction types | Labeling schema |

**Checkpoint**: 200 prompts generated, model outputs collected

### Day 4 (Thursday): Dataset Filtering & Curation

| Time Block | Task | Deliverable |
|------------|------|-------------|
| Morning (3h) | Filter for "Goldilocks" self-correction cases | Filtered dataset |
| | Criteria: error → correction phrase → correct answer | |
| | Target: 30-50 clean examples | `correction_examples.json` |
| Afternoon (3h) | Create control conditions | Control dataset |
| | Correct-No-Correction examples | |
| | Incorrect-No-Correction examples | |
| | Document edge cases and anomalies | Edge case notes |

**Checkpoint**: 30+ self-correction examples, 30+ controls

### Day 5 (Friday): Metrics & Baseline

| Time Block | Task | Deliverable |
|------------|------|-------------|
| Morning (3h) | Implement Correction Logit Difference metric | Metric function |
| | `Score = Logit("Wait") - Logit([Incorrect])` | |
| | Test on sample cases | Baseline measurements |
| Afternoon (3h) | Run full baseline measurements | `baseline_metrics.csv` |
| | Document baseline correction rates | Baseline report |
| | Identify token positions for analysis | Token position map |

**Checkpoint**: Quantitative baseline established

### Weekend: Buffer & Documentation

- Review week's work
- Write up methodology section draft
- Identify any gaps in data or setup
- Prepare for feature analysis phase

---

## Week 2: Feature Analysis & Circuit Hunting

**Goal**: Identify candidate Monitor features using attribution methods

### Day 6 (Monday): Attribution Setup

| Time Block | Task | Deliverable |
|------------|------|-------------|
| Morning (3h) | Implement attribution patching framework | Attribution code |
| | Set up hooks for activation capture | Hook utilities |
| | Test on single example | Validation output |
| Afternoon (3h) | Run attribution on 10 correction examples | Initial attributions |
| | Rank features by contribution to "Wait" logit | Feature rankings |
| | Identify top 20 candidate features | Candidate list |

**Checkpoint**: Attribution pipeline working, initial feature candidates

### Day 7 (Tuesday): Feature Interpretation - Round 1

| Time Block | Task | Deliverable |
|------------|------|-------------|
| Morning (3h) | Look up top 10 features on Neuronpedia | Feature documentation |
| | Examine max-activating examples | |
| | Attempt initial labeling | Feature labels v1 |
| Afternoon (3h) | Cross-reference with correction contexts | Correlation analysis |
| | Check for polysemanticity | Polysemanticity notes |
| | Identify "cleaner" vs "messier" features | Feature quality ranking |

**Checkpoint**: Top 10 features labeled with confidence scores

### Day 8 (Wednesday): Scaling Analysis

| Time Block | Task | Deliverable |
|------------|------|-------------|
| Morning (3h) | Run attribution on full correction dataset (30-50 examples) | Full attribution data |
| | Measure feature consistency across examples | Consistency metrics |
| | Identify features with >0.7 correlation | High-confidence features |
| Afternoon (3h) | Layer-wise analysis: where do correction features emerge? | Layer emergence plot |
| | Compare early vs late layer features | Layer comparison |
| | Document "hot spots" for correction computation | Hot spot map |

**Checkpoint**: Feature consistency quantified across dataset

### Day 9 (Thursday): Control Comparison

| Time Block | Task | Deliverable |
|------------|------|-------------|
| Morning (3h) | Run attribution on control examples | Control attributions |
| | Correct-No-Correction: which features are missing? | Differential analysis |
| | Incorrect-No-Correction: which features fire but don't trigger correction? | |
| Afternoon (3h) | Statistical comparison: correction vs control | Statistical report |
| | Identify features unique to correction | Unique features list |
| | Identify false positive features | False positive notes |

**Checkpoint**: Clear differentiation between correction and control features

### Day 10 (Friday): Transcoder Circuit Analysis

| Time Block | Task | Deliverable |
|------------|------|-------------|
| Morning (3h) | Use Transcoders to trace feature interactions | Circuit traces |
| | Input features → Output features mapping | Interaction matrix |
| | Look for Monitor → Inhibition pathways | Pathway candidates |
| Afternoon (3h) | Visualize preliminary circuit graph | Circuit diagram v1 |
| | Identify key nodes and edges | Key circuit components |
| | Document gaps and uncertainties | Uncertainty notes |

**Checkpoint**: Preliminary circuit diagram, candidate pathways identified

### Weekend: Analysis & Planning

- Synthesize feature findings
- Prioritize features for intervention experiments
- Draft feature analysis section of paper
- Plan intervention experiments in detail

---

## Week 3: Causal Intervention Experiments

**Goal**: Demonstrate causal role of identified features through ablation and steering

### Day 11 (Monday): Ablation Framework

| Time Block | Task | Deliverable |
|------------|------|-------------|
| Morning (3h) | Implement feature ablation utilities | Ablation code |
| | Set activation to zero for target features | |
| | Test on single example | Ablation test |
| Afternoon (3h) | Run ablation on top 5 "Monitor" features | Ablation results |
| | Measure impact on Correction Logit Difference | |
| | Document behavioral changes | Behavioral notes |

**Checkpoint**: Ablation pipeline working, initial results

### Day 12 (Tuesday): Ablation Experiments

| Time Block | Task | Deliverable |
|------------|------|-------------|
| Morning (3h) | Full ablation sweep: each feature individually | Individual ablation data |
| | Measure correction rate drop | |
| | Identify most critical features | Critical feature list |
| Afternoon (3h) | Combination ablation: multiple features together | Combination results |
| | Test for redundancy vs additivity | Redundancy analysis |
| | Document which combinations break correction | Breaking combinations |

**Checkpoint**: Ablation success criteria met (≥30% correction reduction)

### Day 13 (Wednesday): Steering Framework

| Time Block | Task | Deliverable |
|------------|------|-------------|
| Morning (3h) | Implement steering/clamping utilities | Steering code |
| | Clamp features to high activation values | |
| | Test on examples that DON'T need correction | Steering test |
| Afternoon (3h) | Run steering on correct-answer cases | False correction attempts |
| | Measure induced doubt/false correction rate | |
| | Document behavioral changes | Steering behavioral notes |

**Checkpoint**: Steering pipeline working, can induce behavioral changes

### Day 14 (Thursday): Steering Experiments

| Time Block | Task | Deliverable |
|------------|------|-------------|
| Morning (3h) | Full steering sweep: each feature individually | Individual steering data |
| | Measure false correction induction rate | |
| | Identify most potent steering features | Potent features list |
| Afternoon (3h) | Combined steering experiments | Combination steering |
| | Test steering strength variations | Strength curves |
| | Document side effects and anomalies | Side effect notes |

**Checkpoint**: Steering success criteria met (≥20% false correction induction)

### Day 15 (Friday): Cross-Task Generalization

| Time Block | Task | Deliverable |
|------------|------|-------------|
| Morning (3h) | Create small test set for different domains | Cross-task dataset |
| | Logic problems (not math) | |
| | Simple coding errors | |
| Afternoon (3h) | Test ablation/steering on new domains | Cross-task results |
| | Do features transfer? | Transfer analysis |
| | Document task-specific vs general features | Generalization report |

**Checkpoint**: Generalization assessment complete

### Weekend: Results Synthesis

- Compile all intervention results
- Create summary tables and figures
- Identify strongest evidence for/against hypothesis
- Draft intervention results section

---

## Week 4: Write-up, Polish & Submission

**Goal**: Produce publication-ready paper and clean code release

### Day 16 (Monday): Results Compilation

| Time Block | Task | Deliverable |
|------------|------|-------------|
| Morning (3h) | Create all figures and visualizations | Figures folder |
| | Feature activation heatmaps | |
| | Ablation/steering bar charts | |
| | Circuit diagrams | |
| Afternoon (3h) | Create summary tables | Tables document |
| | Baseline vs intervention comparisons | |
| | Feature properties table | |
| | Cross-task transfer matrix | |

**Checkpoint**: All visual assets ready

### Day 17 (Tuesday): Executive Summary

| Time Block | Task | Deliverable |
|------------|------|-------------|
| Morning (3h) | Draft executive summary (1-2 pages) | `executive_summary.md` v1 |
| | Structure: Context → Hypothesis → Method → Result → Implication | |
| | Focus on key findings only | |
| Afternoon (3h) | Revise and tighten | `executive_summary.md` v2 |
| | Get feedback if possible | |
| | Ensure standalone readability | Final executive summary |

**Checkpoint**: Executive summary complete and polished

### Day 18 (Wednesday): Main Report - Part 1

| Time Block | Task | Deliverable |
|------------|------|-------------|
| Morning (3h) | Write Introduction section | Introduction draft |
| | Motivate the problem clearly | |
| | State hypothesis and approach | |
| Afternoon (3h) | Write Methods section | Methods draft |
| | Gemma Scope + SAEs + Transcoders | |
| | Attribution and intervention approaches | |
| | Dataset description | |

**Checkpoint**: Introduction and Methods complete

### Day 19 (Thursday): Main Report - Part 2

| Time Block | Task | Deliverable |
|------------|------|-------------|
| Morning (3h) | Write Results section | Results draft |
| | Feature identification findings | |
| | Ablation results | |
| | Steering results | |
| | Cross-task analysis | |
| Afternoon (3h) | Write Discussion section | Discussion draft |
| | Interpretation of findings | |
| | Limitations (be honest!) | |
| | Future work | |
| | Safety implications | |

**Checkpoint**: Full draft complete

### Day 20 (Friday): Polish & Code Release

| Time Block | Task | Deliverable |
|------------|------|-------------|
| Morning (3h) | Full paper revision | Final paper |
| | Check all claims against evidence | |
| | Ensure figures are referenced correctly | |
| | Proofread thoroughly | |
| Afternoon (3h) | Clean and document code | Code release |
| | Create reproducible notebooks | |
| | Write code README | |
| | Test notebooks end-to-end | |

**Checkpoint**: Paper and code ready for submission

### Weekend: Final Review

- Fresh eyes review of paper
- Final code testing
- Submission preparation
- Backup all materials

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
| 1 | 1 | Model loads and generates |
| 1 | 2 | SAEs and Transcoders integrated |
| 1 | 3 | 200 prompts generated |
| 1 | 4 | 30+ clean correction examples |
| 1 | 5 | Baseline metrics established |
| 2 | 6 | Attribution pipeline working |
| 2 | 7 | Top 10 features labeled |
| 2 | 8 | Feature consistency quantified |
| 2 | 9 | Correction vs control differentiated |
| 2 | 10 | Preliminary circuit diagram |
| 3 | 11 | Ablation pipeline working |
| 3 | 12 | ≥30% correction reduction achieved |
| 3 | 13 | Steering pipeline working |
| 3 | 14 | ≥20% false correction induction |
| 3 | 15 | Generalization assessment complete |
| 4 | 16 | All figures ready |
| 4 | 17 | Executive summary complete |
| 4 | 18 | Intro + Methods complete |
| 4 | 19 | Full draft complete |
| 4 | 20 | Paper + code ready |

---

## Risk Mitigation by Week

### Week 1 Risks
| Risk | Mitigation |
|------|------------|
| GPU OOM | Fall back to Gemma-2-2B |
| Low correction rate | Use few-shot priming |
| SAE loading issues | Check SAELens version compatibility |

### Week 2 Risks
| Risk | Mitigation |
|------|------------|
| No clear features | Lower correlation threshold to 0.5 |
| Polysemantic features | Focus on feature combinations |
| Attribution too slow | Subsample dataset |

### Week 3 Risks
| Risk | Mitigation |
|------|------------|
| Ablation no effect | Check feature selection, try combinations |
| Steering side effects | Document as limitation, reduce strength |
| No cross-task transfer | Report negative result honestly |

### Week 4 Risks
| Risk | Mitigation |
|------|------------|
| Results unclear | Focus on most solid findings |
| Time pressure | Prioritize executive summary |
| Code not reproducible | Test on fresh environment |

---

## Deliverables Checklist

### Week 1 Deliverables
- [ ] Working environment with all libraries
- [ ] Gemma-2-9B-IT loaded and generating
- [ ] Gemma Scope SAEs and Transcoders loaded
- [ ] Self-correction dataset (30-50 examples)
- [ ] Control dataset (30+ examples)
- [ ] Baseline metrics documented

### Week 2 Deliverables
- [ ] Attribution pipeline code
- [ ] Top 10 features identified and labeled
- [ ] Feature consistency analysis
- [ ] Correction vs control comparison
- [ ] Preliminary circuit diagram
- [ ] Feature analysis section draft

### Week 3 Deliverables
- [ ] Ablation experiment results
- [ ] Steering experiment results
- [ ] Cross-task generalization analysis
- [ ] Intervention results section draft
- [ ] Key figures created

### Week 4 Deliverables
- [ ] Executive summary (1-2 pages)
- [ ] Full paper (8-12 pages)
- [ ] Clean, documented code
- [ ] Reproducible notebooks
- [ ] All figures and tables

---

## Success Metrics Summary

| Metric | Target | Minimum |
|--------|--------|---------|
| Self-correction examples | 50 | 30 |
| Identified features | 5+ | 3 |
| Feature correlation | >0.7 | >0.5 |
| Ablation effect | ≥30% | ≥20% |
| Steering effect | ≥20% | ≥10% |
| Cross-task transfer | 2+ domains | 1 domain |
| Paper pages | 8-12 | 5 |
| Code notebooks | 5 | 3 |

---

*Plan created: January 2026*
*Duration: 4 weeks (20 working days)*
*Target: MATS Summer 2026 Application*
