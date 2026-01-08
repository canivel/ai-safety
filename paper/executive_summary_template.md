# Executive Summary Template

## [Title]

*Tracing the "Aha!" Moment: Circuit Analysis of Self-Correction in Chain-of-Thought Reasoning using Gemma Scope Transcoders*

---

## The Question

[1-2 sentences framing the research question]

**Template**: How do language models internally detect and correct their own reasoning errors during Chain-of-Thought? We investigated this using Gemma Scope 2's sparse autoencoders and transcoders on Gemma-2-9B-IT.

---

## The Method

[2-3 sentences on approach]

**Template**: We curated a dataset of [N] self-correction instances where Gemma-2-9B-IT initially made arithmetic errors then corrected itself (e.g., "Wait, that's wrong..."). Using attribution patching through Gemma Scope SAEs at layers [X-Y], we identified features that activate specifically before correction tokens. We then validated these features through ablation and steering experiments.

---

## Key Findings

[3-4 bullet points with specific quantitative results]

**Template**:
- We identified [N] features in layers [X-Y] that consistently activate (correlation > 0.7) before correction tokens
- Feature #[ID] in layer [L] appears to function as an "error detector," firing when arithmetic inconsistencies are present
- Ablating the top [N] correction-associated features reduced self-correction rate by [X]% (from [A]% to [B]%)
- Artificially activating these features on correct-answer cases induced false corrections [X]% of the time

---

## The Implications

[2-3 sentences on significance]

**Template**: These findings suggest that self-correction in Chain-of-Thought relies on identifiable internal monitoring mechanisms, separable from the generation process itself. This has implications for AI safety: if models develop internal "checkers" for reasoning correctness, similar mechanisms might detect whether the model is being tested—relevant to concerns about deceptive alignment. Our methodology provides a template for investigating reasoning circuits in other "thinking models."

---

## Limitations

[2-3 sentences on honest limitations]

**Template**: Our analysis focused solely on arithmetic self-correction; generalization to other reasoning domains (logic, coding) showed [mixed/limited/strong] transfer. The identified features show some polysemanticity, activating on related but distinct concepts like negation and hedging. We studied only Gemma-2-9B; circuit structure may differ in other model families or scales.

---

## Code & Data

[Link to reproducible artifacts]

All code, datasets, and visualizations available at: [Repository Link]

---

*Word count target: 400-600 words (1-2 pages)*

---

## Writing Tips for Executive Summary

### Do:
- Lead with the specific finding, not background
- Include concrete numbers (percentages, counts)
- Be honest about limitations
- Make it standalone (reader shouldn't need main paper)

### Don't:
- Use jargon without explanation
- Overstate claims beyond evidence
- Include methodology details (save for main paper)
- Use "we found something interesting"—say what it is

### Structure Check:
- [ ] Can someone understand the main finding in 30 seconds?
- [ ] Are all claims backed by numbers from experiments?
- [ ] Are limitations honest and specific?
- [ ] Does it make the reader want to read more?
