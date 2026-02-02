# %% [markdown]
# # Interactive Deep Dive: `generate_cot_response`
#
# This script lets you step through the text generation process
# and inspect every variable at runtime.
#
# Run this in VS Code with the Python Interactive Window or Jupyter.
# Each cell (marked with `# %%`) can be executed independently.

# %% [markdown]
# ## Setup: Load Model and Libraries

# %%
# ============================================================
# CELL 1: IMPORTS
# ============================================================
# Run this first to load all required libraries

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from typing import Tuple

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# %%
# ============================================================
# CELL 2: LOAD MODEL
# ============================================================
# This loads Gemma 2B-IT into memory (~5GB VRAM)

MODEL_NAME = "gemma-2-2b-it"
print(f"Loading {MODEL_NAME}...")

model = HookedTransformer.from_pretrained(
    MODEL_NAME,
    device=DEVICE,
    dtype=torch.float16,
)

print(f"Model loaded!")
print(f"  - Vocabulary size: {model.cfg.d_vocab}")
print(f"  - Hidden dimension: {model.cfg.d_model}")
print(f"  - Number of layers: {model.cfg.n_layers}")

# %% [markdown]
# ## Step-by-Step Generation
#
# Now we'll go through each step of the generation process,
# printing intermediate values so you can see exactly what happens.

# %%
# ============================================================
# CELL 3: DEFINE THE PROMPT
# ============================================================

prompt = """Question: What is 2 + 2?

Let me think through this step by step:"""

print("=" * 60)
print("PROMPT:")
print("=" * 60)
print(prompt)
print("=" * 60)

# %%
# ============================================================
# CELL 4: STEP 1 - TOKENIZATION
# ============================================================
# Convert text to token IDs

input_tokens = model.to_tokens(prompt)

print("STEP 1: TOKENIZATION")
print("=" * 60)
print(f"Input type: {type(prompt)}")
print(f"Output type: {type(input_tokens)}")
print(f"Output shape: {input_tokens.shape}")
print(f"  - Batch size: {input_tokens.shape[0]}")
print(f"  - Sequence length: {input_tokens.shape[1]}")
print()
print("Token IDs (first 20):")
print(input_tokens[0, :20].tolist())
print()
print("Decoded tokens (first 20):")
for i, tok_id in enumerate(input_tokens[0, :20].tolist()):
    tok_str = model.to_string(torch.tensor([[tok_id]]))
    print(f"  [{i}] ID={tok_id:6d} -> '{tok_str}'")

# %%
# ============================================================
# CELL 5: STEP 2 - CLONE FOR GENERATION
# ============================================================
# Create a copy we'll append to

all_tokens = input_tokens.clone()

print("STEP 2: CLONE TOKENS")
print("=" * 60)
print(f"Original shape: {input_tokens.shape}")
print(f"Cloned shape: {all_tokens.shape}")
print(f"Are they the same object? {all_tokens is input_tokens}")
print(f"Do they have same values? {torch.equal(all_tokens, input_tokens)}")
print()
print("Why clone?")
print("  - We'll modify all_tokens by appending new tokens")
print("  - clone() prevents modifying the original input_tokens")

# %%
# ============================================================
# CELL 6: STEP 3 - GET LOGITS (ONE FORWARD PASS)
# ============================================================
# Run the model to get predictions

with torch.no_grad():  # Don't track gradients
    full_logits = model(all_tokens)

print("STEP 3: FORWARD PASS -> LOGITS")
print("=" * 60)
print(f"Input shape: {all_tokens.shape}")
print(f"Output shape: {full_logits.shape}")
print(f"  - Batch size: {full_logits.shape[0]}")
print(f"  - Sequence positions: {full_logits.shape[1]}")
print(f"  - Vocabulary size: {full_logits.shape[2]}")
print()
print("The model outputs predictions for EVERY position.")
print("Position i predicts what comes AFTER token i.")
print()
print("We only need the LAST position to predict the next token:")

# Get logits for last position only
logits = full_logits[:, -1, :]  # [batch, vocab]
print(f"After slicing [:, -1, :]: shape = {logits.shape}")

# %%
# ============================================================
# CELL 7: EXAMINE RAW LOGITS
# ============================================================
# Look at what the model predicts

print("EXAMINING RAW LOGITS")
print("=" * 60)
print(f"Logits shape: {logits.shape}")
print(f"Logits dtype: {logits.dtype}")
print(f"Min logit: {logits.min().item():.4f}")
print(f"Max logit: {logits.max().item():.4f}")
print(f"Mean logit: {logits.mean().item():.4f}")
print()

# Find top 10 predicted tokens
top_k = 10
top_logits, top_indices = torch.topk(logits[0], top_k)

print(f"TOP {top_k} PREDICTED TOKENS (before temperature/sampling):")
print("-" * 60)
for i, (logit, idx) in enumerate(zip(top_logits, top_indices)):
    token_str = model.to_string(torch.tensor([[idx.item()]]))
    # Convert to probability for intuition
    prob = F.softmax(logits[0], dim=-1)[idx].item()
    print(f"  {i+1}. '{token_str}' (ID={idx.item()}) logit={logit.item():.2f} prob={prob:.2%}")

# %%
# ============================================================
# CELL 8: STEP 4 - APPLY TEMPERATURE
# ============================================================
# Scale logits to control randomness

temperature = 0.7
logits_original = logits.clone()
logits_scaled = logits / temperature

print("STEP 4: TEMPERATURE SCALING")
print("=" * 60)
print(f"Temperature: {temperature}")
print()
print("BEFORE temperature (original logits):")
print(f"  Max: {logits_original.max().item():.4f}")
print(f"  Min: {logits_original.min().item():.4f}")
print()
print("AFTER temperature (divided by {})".format(temperature))
print(f"  Max: {logits_scaled.max().item():.4f}")
print(f"  Min: {logits_scaled.min().item():.4f}")
print()

# Show effect on probabilities
probs_original = F.softmax(logits_original[0], dim=-1)
probs_scaled = F.softmax(logits_scaled[0], dim=-1)

print("EFFECT ON TOP TOKEN PROBABILITIES:")
print("-" * 60)
top_indices_orig = torch.topk(probs_original, 5).indices
for idx in top_indices_orig:
    token_str = model.to_string(torch.tensor([[idx.item()]]))
    p_orig = probs_original[idx].item()
    p_scaled = probs_scaled[idx].item()
    print(f"  '{token_str}': {p_orig:.2%} -> {p_scaled:.2%} (change: {(p_scaled-p_orig)*100:+.1f}pp)")

print()
print("Lower temperature SHARPENS the distribution (top tokens get higher prob)")

# Use scaled logits going forward
logits = logits_scaled

# %%
# ============================================================
# CELL 9: STEP 5 - TOP-P SAMPLING (Part 1: Sort)
# ============================================================
# Sort tokens by probability to prepare for nucleus sampling

top_p = 0.9

# Sort logits in descending order
sorted_logits, sorted_indices = torch.sort(logits, descending=True)

print("STEP 5a: SORT TOKENS BY LOGIT")
print("=" * 60)
print(f"sorted_logits shape: {sorted_logits.shape}")
print(f"sorted_indices shape: {sorted_indices.shape}")
print()
print("First 10 sorted logits (highest to lowest):")
for i in range(10):
    idx = sorted_indices[0, i].item()
    logit = sorted_logits[0, i].item()
    token_str = model.to_string(torch.tensor([[idx]]))
    print(f"  {i+1}. logit={logit:.2f} ID={idx} '{token_str}'")

# %%
# ============================================================
# CELL 10: STEP 5 - TOP-P SAMPLING (Part 2: Cumulative Sum)
# ============================================================
# Calculate running sum of probabilities

# Convert sorted logits to probabilities
sorted_probs = F.softmax(sorted_logits, dim=-1)

# Calculate cumulative sum
cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

print("STEP 5b: CUMULATIVE PROBABILITY")
print("=" * 60)
print()
print("Individual probs (first 10):")
for i in range(10):
    print(f"  Token {i+1}: prob={sorted_probs[0, i].item():.4f}")
print()
print("Cumulative probs (first 10):")
for i in range(10):
    cum = cumulative_probs[0, i].item()
    marker = " <-- top_p cutoff here!" if i > 0 and cumulative_probs[0, i-1].item() < top_p <= cum else ""
    print(f"  After token {i+1}: cumsum={cum:.4f}{marker}")
print()
print(f"top_p = {top_p}")
print("Tokens AFTER cumsum exceeds top_p will be removed")

# %%
# ============================================================
# CELL 11: STEP 5 - TOP-P SAMPLING (Part 3: Create Mask)
# ============================================================
# Determine which tokens to remove

# Initial mask: True where cumsum > top_p
sorted_indices_to_remove = cumulative_probs > top_p

print("STEP 5c: CREATE REMOVAL MASK")
print("=" * 60)
print()
print("Initial mask (True = will be removed):")
print(f"  First 15 values: {sorted_indices_to_remove[0, :15].tolist()}")
print()

# Shift mask right by 1 (to include the token that crosses threshold)
sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
sorted_indices_to_remove[..., 0] = 0  # Never remove first token

print("After shifting (include threshold-crossing token):")
print(f"  First 15 values: {sorted_indices_to_remove[0, :15].tolist()}")
print()

# Count how many tokens we're keeping
n_kept = (~sorted_indices_to_remove[0]).sum().item()
print(f"Tokens kept: {n_kept} out of {model.cfg.d_vocab}")
print(f"Tokens removed: {model.cfg.d_vocab - n_kept}")

# %%
# ============================================================
# CELL 12: STEP 5 - TOP-P SAMPLING (Part 4: Apply Mask)
# ============================================================
# Map mask back to original positions and apply

# Scatter the sorted mask back to original token positions
indices_to_remove = sorted_indices_to_remove.scatter(
    1, sorted_indices, sorted_indices_to_remove
)

print("STEP 5d: APPLY MASK TO LOGITS")
print("=" * 60)
print()
print("Before masking:")
print(f"  Non-inf logits: {(logits != float('-inf')).sum().item()}")
print()

# Apply mask: set removed tokens to -infinity
logits_masked = logits.clone()
logits_masked[indices_to_remove] = float('-inf')

print("After masking:")
print(f"  Non-inf logits: {(logits_masked != float('-inf')).sum().item()}")
print()
print("Logits set to -inf will have probability 0 after softmax")

logits = logits_masked

# %%
# ============================================================
# CELL 13: STEP 6 - SAMPLE NEXT TOKEN
# ============================================================
# Convert to probabilities and randomly sample

# Convert to probabilities
probs = F.softmax(logits, dim=-1)

print("STEP 6: SAMPLE FROM DISTRIBUTION")
print("=" * 60)
print()
print(f"Probability sum: {probs.sum().item():.6f} (should be ~1.0)")
print(f"Non-zero probs: {(probs > 0).sum().item()}")
print()

# Show the tokens we're sampling from
nonzero_mask = probs[0] > 0
nonzero_probs = probs[0][nonzero_mask]
nonzero_indices = torch.where(nonzero_mask)[0]

print("TOKENS IN SAMPLING POOL:")
print("-" * 60)
# Sort by probability for display
sorted_probs_display, sorted_idx = torch.sort(nonzero_probs, descending=True)
for i in range(min(10, len(sorted_probs_display))):
    idx = nonzero_indices[sorted_idx[i]].item()
    prob = sorted_probs_display[i].item()
    token_str = model.to_string(torch.tensor([[idx]]))
    print(f"  '{token_str}' (ID={idx}): {prob:.2%}")
if len(sorted_probs_display) > 10:
    print(f"  ... and {len(sorted_probs_display) - 10} more tokens")
print()

# Sample!
next_token = torch.multinomial(probs, num_samples=1)
sampled_token_str = model.to_string(next_token)

print("SAMPLED TOKEN:")
print(f"  Token ID: {next_token.item()}")
print(f"  Token string: '{sampled_token_str}'")
print(f"  Probability: {probs[0, next_token.item()].item():.2%}")

# %%
# ============================================================
# CELL 14: STEP 7 - APPEND TO SEQUENCE
# ============================================================
# Add the new token and check for end

print("STEP 7: APPEND TO SEQUENCE")
print("=" * 60)
print()
print(f"Before append: all_tokens shape = {all_tokens.shape}")

# Append
all_tokens = torch.cat([all_tokens, next_token], dim=1)

print(f"After append: all_tokens shape = {all_tokens.shape}")
print()
print(f"New token: '{sampled_token_str}'")
print()

# Check for EOS
eos_id = model.tokenizer.eos_token_id
is_eos = next_token.item() == eos_id
print(f"Is this the EOS token? {is_eos}")
print(f"  (EOS token ID: {eos_id})")

if is_eos:
    print("\nGeneration would STOP here!")
else:
    print("\nGeneration continues...")

# %%
# ============================================================
# CELL 15: STEP 8 - DECODE TO TEXT
# ============================================================
# Convert all tokens back to readable text

generated_text = model.to_string(all_tokens[0])

print("STEP 8: DECODE TO TEXT")
print("=" * 60)
print()
print("Full generated text so far:")
print("-" * 60)
print(generated_text)
print("-" * 60)

# %%
# ============================================================
# CELL 16: TWO GENERATION APPROACHES - THEORY
# ============================================================
#
# There are TWO ways to generate text, both mathematically IDENTICAL:
#
# ┌─────────────────────────────────────────────────────────────────┐
# │ APPROACH 1: KV-CACHE (Production - FAST)                        │
# │                                                                 │
# │ How it works:                                                   │
# │   Step 1: Process prompt, SAVE K,V vectors to cache             │
# │   Step 2: Process only new token, REUSE cached K,V              │
# │   Step 3: Process only new token, REUSE cached K,V              │
# │   ...                                                           │
# │                                                                 │
# │ Complexity: O(n) - linear in sequence length                    │
# │ Used by: ChatGPT, Claude, all production systems                │
# │ Use when: You just need the output text                         │
# └─────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────┐
# │ APPROACH 2: NAIVE STEP-BY-STEP (Research - SLOW)                │
# │                                                                 │
# │ How it works:                                                   │
# │   Step 1: Process ALL tokens [1...n]                            │
# │   Step 2: Process ALL tokens [1...n+1] (redo everything!)       │
# │   Step 3: Process ALL tokens [1...n+2] (redo everything!)       │
# │   ...                                                           │
# │                                                                 │
# │ Complexity: O(n²) - quadratic in sequence length                │
# │ Used by: Researchers who need to inspect activations            │
# │ Use when: You need to see/modify internal states                │
# └─────────────────────────────────────────────────────────────────┘
#
# WHY BOTH EXIST:
# - KV-cache is an OPTIMIZATION, not a different computation
# - K and V for token i only depend on tokens 1..i (causal masking)
# - Once computed, they NEVER change, so we can cache them
# - But for research, we often want ALL activations, not just K,V
#
# MATHEMATICAL PROOF THEY'RE EQUIVALENT:
# The attention mechanism computes:
#   Attention_i = softmax(Q_i · K_{1:i}^T / √d) · V_{1:i}
#
# Key insight: K_j and V_j for position j only depend on tokens 1 to j
# So when generating token i+1:
#   - K_{1:i} is the SAME as when we generated token i
#   - V_{1:i} is the SAME as when we generated token i
#   - We only need to compute K_{i+1} and V_{i+1} for the new token
#
# This is why KV-cache gives IDENTICAL results but much faster!

print("=" * 70)
print("TWO GENERATION APPROACHES")
print("=" * 70)
print("""
APPROACH 1: KV-CACHE (Fast, Production)
  - Caches Key/Value vectors from previous tokens
  - Only computes new token's attention each step
  - O(n) complexity - FAST
  - Used by: ChatGPT, Claude, all production systems
  - Use for: Actual experiments, when you just need output

APPROACH 2: NAIVE (Slow, Research)
  - Recomputes ENTIRE sequence each step
  - No caching - everything fresh
  - O(n²) complexity - SLOW
  - Used by: Researchers inspecting activations
  - Use for: Learning, debugging, activation analysis

Both produce IDENTICAL outputs!
""")
print("=" * 70)

# %%
# ============================================================
# IMPLEMENTATION: APPROACH 1 - KV-CACHE (FAST)
# ============================================================

@torch.no_grad()
def generate_fast(
    model: HookedTransformer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Tuple[str, torch.Tensor]:
    """
    FAST generation using KV-cache (production method).

    This is what ChatGPT/Claude use internally.
    TransformerLens's model.generate() handles KV-caching automatically.

    USE THIS FOR:
    ✓ Actual experiments where you need many generations
    ✓ Any time you just need the text output
    ✓ Benchmarking, evaluation, data collection

    DO NOT USE FOR:
    ✗ When you need to inspect internal activations
    ✗ Intervention experiments (activation patching)
    ✗ Understanding how generation works step-by-step

    Complexity: O(n) - fast!
    """
    input_tokens = model.to_tokens(prompt)

    # model.generate() uses KV-caching internally
    # We don't see the cache, but it's there making things fast
    output_tokens = model.generate(
        input_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        verbose=False,
    )

    return model.to_string(output_tokens[0]), output_tokens


# %%
# ============================================================
# IMPLEMENTATION: APPROACH 2 - NAIVE STEP-BY-STEP (SLOW)
# ============================================================

@torch.no_grad()
def generate_naive(
    model: HookedTransformer,
    prompt: str,
    max_new_tokens: int = 10,  # Keep small! O(n²) complexity
    temperature: float = 0.7,
    top_p: float = 0.9,
    verbose: bool = True,
) -> Tuple[str, torch.Tensor]:
    """
    SLOW generation using naive step-by-step (research method).

    Recomputes the ENTIRE forward pass for each new token.
    Wasteful but transparent - you can see/modify everything.

    USE THIS FOR:
    ✓ Understanding how generation works
    ✓ Educational purposes
    ✓ When you'll add activation caching later
    ✓ Debugging and inspection

    DO NOT USE FOR:
    ✗ Generating lots of text (too slow)
    ✗ Production systems
    ✗ Any time you just need the output

    Complexity: O(n²) - slow!
    Keep max_new_tokens <= 20 for reasonable speed.
    """
    input_tokens = model.to_tokens(prompt)
    all_tokens = input_tokens.clone()

    if verbose:
        print(f"NAIVE generation (max {max_new_tokens} tokens)")
        print(f"This is O(n²) - intentionally slow for transparency")
        print(f"Initial sequence: {all_tokens.shape[1]} tokens")
        print("-" * 50)

    for step in range(max_new_tokens):
        # =========================================================
        # THE KEY DIFFERENCE: We process ALL tokens every time
        # This is "wasteful" but lets us inspect everything
        # =========================================================

        # Forward pass on ENTIRE sequence (this is the slow part)
        logits = model(all_tokens)[:, -1, :]  # [batch, vocab]

        # Temperature scaling
        logits = logits / temperature

        # Top-p (nucleus) sampling
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')

        # Sample next token
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Append to sequence
        all_tokens = torch.cat([all_tokens, next_token], dim=1)

        if verbose:
            token_str = model.to_string(next_token)
            prob = probs[0, next_token.item()].item()
            seq_len = all_tokens.shape[1]
            print(f"Step {step+1}: '{token_str}' (prob={prob:.1%}) [seq_len={seq_len}]")

        # Check for EOS
        if next_token.item() == model.tokenizer.eos_token_id:
            if verbose:
                print("-> EOS token reached, stopping")
            break

    return model.to_string(all_tokens[0]), all_tokens


# %%
# ============================================================
# IMPLEMENTATION: APPROACH 2B - NAIVE WITH ACTIVATION CACHING
# ============================================================

@torch.no_grad()
def generate_with_activations(
    model: HookedTransformer,
    prompt: str,
    max_new_tokens: int = 10,  # Keep VERY small!
    layers_to_cache: list = None,
) -> Tuple[str, list]:
    """
    Generate while saving activations at each step.

    This is the INTERPRETABILITY version - saves internal states
    so we can analyze what the model is "thinking" at each step.

    USE THIS FOR:
    ✓ Analyzing how representations evolve during generation
    ✓ Finding which layers "decide" what to output
    ✓ Activation patching / intervention experiments
    ✓ Interpretability research

    WARNING: Very slow AND memory-intensive!
    Keep max_new_tokens <= 10 or you'll run out of memory.

    Returns:
        (generated_text, list_of_caches)
        where list_of_caches[i] is a dict of activations at step i
    """
    if layers_to_cache is None:
        # Default: first, middle, last layer
        layers_to_cache = [0, model.cfg.n_layers // 2, model.cfg.n_layers - 1]

    input_tokens = model.to_tokens(prompt)
    all_tokens = input_tokens.clone()
    all_caches = []

    # Define which activations to save
    names_to_cache = []
    for layer in layers_to_cache:
        names_to_cache.extend([
            f"blocks.{layer}.hook_resid_pre",   # Input to layer
            f"blocks.{layer}.hook_resid_post",  # Output of layer
            f"blocks.{layer}.attn.hook_pattern", # Attention patterns
        ])

    print(f"Generating with activation caching")
    print(f"Caching layers: {layers_to_cache}")
    print(f"This will be VERY slow and use lots of memory!")
    print("-" * 50)

    for step in range(max_new_tokens):
        # Run forward pass AND cache activations
        logits, cache = model.run_with_cache(
            all_tokens,
            names_filter=lambda name: name in names_to_cache
        )

        # Save activations (move to CPU to save GPU memory)
        step_cache = {k: v.cpu().clone() for k, v in cache.items()}
        all_caches.append(step_cache)

        # Greedy decoding (deterministic for reproducibility)
        next_token = logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
        all_tokens = torch.cat([all_tokens, next_token], dim=1)

        token_str = model.to_string(next_token)
        print(f"Step {step+1}: '{token_str}' [cached {len(step_cache)} activation tensors]")

        if next_token.item() == model.tokenizer.eos_token_id:
            print("-> EOS reached")
            break

    return model.to_string(all_tokens[0]), all_caches


# Run it!
print("=" * 60)
print("GENERATING WITH KV-CACHE (FAST)")
print("=" * 60)

test_prompt = """Question: What is the capital of France?

Let me think step by step:"""

import time
start = time.time()
result_text, result_tokens = generate_fast(model, test_prompt, max_new_tokens=100)
elapsed = time.time() - start

print(f"\nGenerated in {elapsed:.2f} seconds")
print("=" * 60)
print(result_text)
print("=" * 60)

# %%
# ============================================================
# CELL 17: EXPERIMENT - EFFECT OF TEMPERATURE
# ============================================================
# See how temperature changes the output

print("EXPERIMENT: EFFECT OF TEMPERATURE")
print("=" * 60)

test_prompt = "The meaning of life is"

for temp in [0.1, 0.5, 1.0, 1.5]:
    print(f"\n--- Temperature = {temp} ---")

    # Generate with this temperature
    tokens = model.to_tokens(test_prompt)
    logits = model(tokens)[:, -1, :] / temp
    probs = F.softmax(logits, dim=-1)

    # Show top 5 tokens
    top_probs, top_idx = torch.topk(probs[0], 5)
    for prob, idx in zip(top_probs, top_idx):
        token_str = model.to_string(torch.tensor([[idx.item()]]))
        print(f"  '{token_str}': {prob.item():.1%}")

# %%
# ============================================================
# CELL 18: EXPERIMENT - EFFECT OF TOP_P
# ============================================================
# See how top_p changes the sampling pool

print("EXPERIMENT: EFFECT OF TOP_P")
print("=" * 60)

test_prompt = "The meaning of life is"
tokens = model.to_tokens(test_prompt)
logits = model(tokens)[:, -1, :] / 0.7  # Fixed temperature

# Get base probabilities
base_probs = F.softmax(logits, dim=-1)

for top_p in [0.5, 0.7, 0.9, 0.95, 1.0]:
    # Apply top-p filtering
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Count tokens in pool
    n_tokens = (cumulative_probs[0] <= top_p).sum().item() + 1  # +1 for threshold token
    n_tokens = min(n_tokens, logits.shape[1])

    print(f"\ntop_p = {top_p}: {n_tokens} tokens in sampling pool")

    # Show which tokens
    for i in range(min(5, n_tokens)):
        idx = sorted_indices[0, i].item()
        prob = base_probs[0, idx].item()
        token_str = model.to_string(torch.tensor([[idx]]))
        print(f"  '{token_str}': {prob:.1%}")
    if n_tokens > 5:
        print(f"  ... and {n_tokens - 5} more")

# %%
# ============================================================
# CELL 19: CLEANUP
# ============================================================
# Free GPU memory

import gc

del model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"GPU memory after cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

print("Done! Model unloaded.")
