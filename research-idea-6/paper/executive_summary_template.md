# Executive Summary Template

## From Inference to Pandering: Detecting and Steering Implicit User Modeling to Reduce Sycophancy

---

## The Question

[1-2 sentences framing the research question]

**Template**: How do RLHF-trained language models internally represent users, and how do these representations drive sycophantic behavior? We investigated this using Gemma Scope SAEs and Cross-Coders to identify "User Model" features in Gemma-2-9B-IT.

---

## The Problem

[2-3 sentences on why this matters]

**Template**: Chat models exhibit sycophancy—telling users what they want to hear rather than the truth. This emerges from user modeling: inferring user attributes (expertise, emotional state) and adjusting responses accordingly. Sycophancy undermines trust and creates safety risks when models prioritize approval over accuracy.

---

## The Method

[3-4 sentences on approach]

**Template**: We created a dataset of [N] prompt pairs with identical questions but different expertise framing ("I'm a beginner" vs "I've studied this for years"). Using Gemma Scope SAEs, we identified features that activate differently based on user cues. We then validated these features through ablation experiments, measuring sycophancy reduction while monitoring helpfulness preservation.

---

## Key Findings

[4-5 bullet points with specific quantitative results]

**Template**:
- We identified [N] features in layers [X-Y] that encode user expertise level, with correlation > [0.X] to expertise framing
- Feature #[ID] ("User is novice") in layer [L] activates [X]× stronger on beginner-framed prompts
- Ablating the top [N] user-modeling features reduced sycophancy score by [X]% (from [A] to [B])
- Helpfulness metrics remained stable (≤[X]% degradation on coding/QA tasks)
- Cross-Coder analysis confirmed these features are Chat-specific (absent in Base model), indicating RLHF origin

---

## The Implications

[2-3 sentences on significance]

**Template**: These findings demonstrate that sycophancy is mediated by identifiable, ablatable features—not an emergent property of the entire model. This suggests targeted interventions can reduce sycophantic behavior without retraining. More broadly, understanding user-modeling circuits is foundational to detecting potentially manipulative behaviors in AI systems.

---

## Limitations

[2-3 sentences on honest limitations]

**Template**: Our analysis focused on explicit expertise framing; implicit user cues (writing style, vocabulary) were not systematically studied. The identified features show some polysemanticity, occasionally activating on non-user contexts. We studied only Gemma-2-9B; circuit structure may differ in other model families or at different scales.

---

## Code & Data

[Link to reproducible artifacts]

All code, datasets, and analysis notebooks available at: [Repository Link]

---

*Word count target: 400-600 words (1-2 pages)*

---

## Writing Checklist

### Content
- [ ] Question is clear in first paragraph
- [ ] Method is understandable without technical background
- [ ] Findings include specific numbers
- [ ] Limitations are honest and specific
- [ ] Implications connect to safety

### Style
- [ ] No jargon without explanation
- [ ] Active voice throughout
- [ ] Standalone (reader doesn't need main paper)
- [ ] Claims match evidence (no overclaiming)

### Format
- [ ] 1-2 pages maximum
- [ ] Clear section structure
- [ ] Numbers formatted consistently
- [ ] Links to code/data included
