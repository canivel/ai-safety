#!/usr/bin/env python3
"""
Experiment 2: Attention Head Circuit Tracing — Ethnicity
(Adapted from gender version)

Which attention heads propagate ethnicity information from name tokens to
question tokens?

Method:
  Phase 1 -- Attention pattern extraction:
    1. Load model with eager attention (required for attention weight access)
    2. Run prompts (ref-group + cmp-group) with output_attentions=True
    3. For each head, compute mean attention from question tokens -> name tokens
    4. Rank heads by attention to name tokens and by ethnicity-differential attention

  Phase 2 -- Causal ablation verification:
    1. Zero out top-N name-attending heads via forward hooks
    2. Re-extract hidden states for all prompts
    3. Re-run probing classifier
    4. If probing accuracy drops -> those heads causally propagate ethnicity info

Usage:
    python run_circuit_tracing_eth.py <model_key> <comparison>
    python run_circuit_tracing_eth.py gemma4b white_vs_black
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import gc
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Add parent dir so we can import from shared/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.load_data import (
    load_questions,
    load_ethnicity_names,
    load_model_registry,
    get_ethnicity_comparison,
    VALID_COMPARISONS,
)

# === MODEL REGISTRY (extends shared registry with best_probe_layer) ===
_shared_registry = load_model_registry()
_PROBE_LAYERS = {
    "gemma4b": 17,
    "gemma1b": 6,
    "gemma12b": 17,
    "mistral7b": 15,
    "qwen7b": 14,
}
MODEL_REGISTRY = {}
for key, cfg in _shared_registry.items():
    MODEL_REGISTRY[key] = {**cfg, "best_probe_layer": _PROBE_LAYERS.get(key, 15)}

# === CLI ===
if len(sys.argv) < 3:
    print(f"Usage: python run_circuit_tracing_eth.py <model_key> <comparison>")
    print(f"  Models:      {list(MODEL_REGISTRY.keys())}")
    print(f"  Comparisons: {VALID_COMPARISONS}")
    sys.exit(1)

MODEL_KEY = sys.argv[1]
COMPARISON = sys.argv[2]

if MODEL_KEY not in MODEL_REGISTRY:
    print(f"ERROR: Unknown model '{MODEL_KEY}'. Choose from: {list(MODEL_REGISTRY.keys())}")
    sys.exit(1)
if COMPARISON not in VALID_COMPARISONS:
    print(f"ERROR: Unknown comparison '{COMPARISON}'. Choose from: {VALID_COMPARISONS}")
    sys.exit(1)

MODEL_CFG = MODEL_REGISTRY[MODEL_KEY]
MODEL_ID = MODEL_CFG["model_id"]
MODEL_SHORT = MODEL_CFG["model_short"]
LOADER = MODEL_CFG.get("loader", "auto")
BEST_PROBE_LAYER = MODEL_CFG["best_probe_layer"]

REF_GROUP, CMP_GROUP = get_ethnicity_comparison(COMPARISON)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "ethnicity_probing" / COMPARISON
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# === DATASET ===
questions = load_questions()
eth_data = load_ethnicity_names()

ref_names = eth_data[REF_GROUP]["names"]
cmp_names = eth_data[CMP_GROUP]["names"]

N_QUESTIONS = len(questions)
N_REF = len(ref_names)
N_CMP = len(cmp_names)

ref_prompts = [f"Hi, my name is {ref_names[i % N_REF]}. {questions[i]}" for i in range(N_QUESTIONS)]
cmp_prompts = [f"Hi, my name is {cmp_names[i % N_CMP]}. {questions[i]}" for i in range(N_QUESTIONS)]

all_ethnicity_prompts = ref_prompts + cmp_prompts
ethnicity_labels = np.array([0] * N_QUESTIONS + [1] * N_QUESTIONS)


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def load_model_and_tokenizer(model_id, loader, eager_attention=False):
    """Load model with optional eager attention for attention weight extraction."""
    from huggingface_hub import login
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    extra_kwargs = {}
    if eager_attention:
        extra_kwargs["attn_implementation"] = "eager"

    if loader == "gemma_causal":
        from transformers import AutoTokenizer, Gemma3ForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = Gemma3ForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto",
            **extra_kwargs)
        tok = tokenizer
        get_layers = lambda m: m.model.layers
        get_embed = lambda m: m.model.embed_tokens
    elif loader == "gemma_conditional":
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration
        tokenizer = AutoProcessor.from_pretrained(model_id)
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto",
            **extra_kwargs)
        tok = tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else tokenizer
        get_layers = lambda m: m.model.language_model.layers
        get_embed = lambda m: m.model.language_model.embed_tokens
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto",
            **extra_kwargs)
        tok = tokenizer
        get_layers = lambda m: m.model.layers
        get_embed = lambda m: m.model.embed_tokens

    model.eval()

    if hasattr(model.config, 'num_hidden_layers'):
        num_layers = model.config.num_hidden_layers
    elif hasattr(model.config, 'text_config') and hasattr(model.config.text_config, 'num_hidden_layers'):
        num_layers = model.config.text_config.num_hidden_layers
    else:
        raise ValueError(f"Cannot find num_hidden_layers in config: {model.config}")

    # Auto-detect attention head dimensions from config and o_proj weight
    cfg = model.config.text_config if hasattr(model.config, 'text_config') else model.config
    num_attn_heads = cfg.num_attention_heads
    head_dim = getattr(cfg, 'head_dim', None)
    # Verify against o_proj weight shape
    layer0 = get_layers(model)[0]
    o_proj_in = layer0.self_attn.o_proj.in_features
    if head_dim is None:
        head_dim = o_proj_in // num_attn_heads
    # Double-check: num_heads * head_dim must equal o_proj input
    if num_attn_heads * head_dim != o_proj_in:
        print(f"  WARNING: {num_attn_heads} heads * {head_dim} dim = {num_attn_heads * head_dim} "
              f"!= o_proj in_features {o_proj_in}")
        # Fall back to deriving from o_proj
        head_dim = o_proj_in // num_attn_heads
        print(f"  Using derived head_dim={head_dim}")

    return model, tok, get_layers, get_embed, num_layers, num_attn_heads, head_dim


def find_name_boundaries(tok, all_prompts, ref_prompts, cmp_prompts, n_questions):
    """Find name token positions by comparing ref/cmp token sequences."""
    boundaries = []
    for i in range(len(all_prompts)):
        q_idx = i if i < n_questions else i - n_questions
        this_text = all_prompts[i]
        pair_text = cmp_prompts[q_idx] if i < n_questions else ref_prompts[q_idx]

        this_ids = tok(this_text, return_tensors="pt", truncation=True, max_length=128)["input_ids"][0].tolist()
        pair_ids = tok(pair_text, return_tensors="pt", truncation=True, max_length=128)["input_ids"][0].tolist()

        name_start = min(len(this_ids), len(pair_ids))
        for j in range(min(len(this_ids), len(pair_ids))):
            if this_ids[j] != pair_ids[j]:
                name_start = j
                break

        suffix_len = 0
        for j in range(1, min(len(this_ids), len(pair_ids)) - name_start + 1):
            if this_ids[-j] == pair_ids[-j]:
                suffix_len += 1
            else:
                break

        name_end = len(this_ids) - suffix_len
        boundaries.append((name_start, name_end))

    return boundaries


def make_head_ablation_hook(target_heads, num_heads, head_dim):
    """
    Create a forward_pre_hook on o_proj that zeros specific attention heads.

    The input to o_proj is (batch, seq, num_heads * head_dim).
    We reshape, zero target heads, reshape back.
    """
    def hook_fn(module, input):
        x = input[0]  # (batch, seq, hidden_dim)
        batch, seq, hidden = x.shape
        x = x.view(batch, seq, num_heads, head_dim)
        for head_idx in target_heads:
            x[:, :, head_idx, :] = 0.0
        return (x.view(batch, seq, hidden),) + input[1:]
    return hook_fn


# =====================================================================
# MAIN
# =====================================================================

def main():
    t0 = datetime.now()
    print("=" * 70)
    print("  Experiment 2: Attention Head Circuit Tracing — Ethnicity")
    print(f"  Model:      {MODEL_ID}")
    print(f"  Comparison: {COMPARISON} ({REF_GROUP} vs {CMP_GROUP})")
    print(f"  Best probe layer: {BEST_PROBE_LAYER}")
    print("=" * 70)

    # --- Step 1: Load model with eager attention ---
    print(f"\n[1/6] Loading model with eager attention...")
    model, tok, get_layers, get_embed, num_layers, NUM_HEADS, HEAD_DIM = \
        load_model_and_tokenizer(MODEL_ID, LOADER, eager_attention=True)

    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Loaded | Layers: {num_layers} | VRAM: {vram:.1f} GB")
    print(f"  Attention: {NUM_HEADS} heads x {HEAD_DIM} dim (o_proj input: {NUM_HEADS * HEAD_DIM})")

    # --- Step 2: Find name boundaries ---
    print(f"\n[2/6] Finding name boundaries...")
    name_boundaries = find_name_boundaries(
        tok, all_ethnicity_prompts, ref_prompts, cmp_prompts, N_QUESTIONS)
    name_lens = [nb[1] - nb[0] for nb in name_boundaries]
    print(f"  Name token lengths: {sorted(set(name_lens))} unique values")

    # --- Step 3: Extract attention patterns ---
    N_ATTN_SUBSET = 50  # Questions for attention analysis
    n_prompts = N_ATTN_SUBSET * 2  # ref + cmp
    print(f"\n[3/6] Extracting attention patterns ({n_prompts} prompts)...")

    # Store per-head attention scores: (prompt, layer, head)
    # Score = mean attention from question tokens to name tokens
    attn_to_name = np.zeros((n_prompts, num_layers, NUM_HEADS))
    attn_from_last = np.zeros((n_prompts, num_layers, NUM_HEADS))

    # Select subset: first N_ATTN_SUBSET ref + first N_ATTN_SUBSET cmp
    subset_indices = list(range(N_ATTN_SUBSET)) + \
                     list(range(N_QUESTIONS, N_QUESTIONS + N_ATTN_SUBSET))

    for p_idx, prompt_idx in enumerate(tqdm(subset_indices, desc="Attention")):
        text = all_ethnicity_prompts[prompt_idx]
        name_start, name_end = name_boundaries[prompt_idx]

        inputs = tok(text, return_tensors="pt", truncation=True, max_length=128).to("cuda")
        seq_len = inputs["input_ids"].shape[1]
        question_start = min(name_end, seq_len)

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        # outputs.attentions: tuple of (1, num_heads, seq_len, seq_len) per layer
        attentions = outputs.attentions
        if attentions is None:
            print(f"  WARNING: No attention weights returned for prompt {prompt_idx}")
            continue

        for layer_idx in range(min(len(attentions), num_layers)):
            attn = attentions[layer_idx][0].float().cpu()  # (num_heads, seq, seq)
            n_heads_actual = attn.shape[0]

            for h in range(min(n_heads_actual, NUM_HEADS)):
                # Question tokens attending to name tokens
                if question_start < seq_len and name_start < name_end:
                    attn_qn = attn[h, question_start:, name_start:name_end]
                    attn_to_name[p_idx, layer_idx, h] = attn_qn.mean().item()

                # Last token attending to name tokens
                if name_start < name_end:
                    attn_ln = attn[h, -1, name_start:name_end]
                    attn_from_last[p_idx, layer_idx, h] = attn_ln.mean().item()

        # Free attention memory
        del outputs, attentions
        torch.cuda.empty_cache()

    # Split into ref/cmp
    ref_attn_name = attn_to_name[:N_ATTN_SUBSET]  # (50, layers, heads)
    cmp_attn_name = attn_to_name[N_ATTN_SUBSET:]

    ref_attn_last = attn_from_last[:N_ATTN_SUBSET]
    cmp_attn_last = attn_from_last[N_ATTN_SUBSET:]

    # --- Step 4: Rank heads ---
    print(f"\n[4/6] Ranking heads by name attention and ethnicity differential...")

    # Mean attention to name (across all prompts)
    mean_attn = attn_to_name.mean(axis=0)  # (layers, heads)
    mean_attn_last_token = attn_from_last.mean(axis=0)

    # Ethnicity differential: |mean_ref - mean_cmp|
    ethnicity_diff = np.abs(ref_attn_name.mean(axis=0) - cmp_attn_name.mean(axis=0))
    ethnicity_diff_last = np.abs(ref_attn_last.mean(axis=0) - cmp_attn_last.mean(axis=0))

    # Flatten and rank
    head_scores = []
    for layer in range(num_layers):
        for head in range(NUM_HEADS):
            head_scores.append({
                "layer": layer,
                "head": head,
                "mean_attn_to_name": float(mean_attn[layer, head]),
                "mean_attn_last_to_name": float(mean_attn_last_token[layer, head]),
                "ethnicity_diff_question": float(ethnicity_diff[layer, head]),
                "ethnicity_diff_last": float(ethnicity_diff_last[layer, head]),
            })

    # Sort by attention from question->name (most attentive first)
    by_attn = sorted(head_scores, key=lambda x: x["mean_attn_to_name"], reverse=True)
    print(f"\n  Top 10 heads by question->name attention:")
    for i, h in enumerate(by_attn[:10]):
        print(f"    {i+1}. Layer {h['layer']:2d} Head {h['head']:2d}: "
              f"attn={h['mean_attn_to_name']:.4f} "
              f"diff={h['ethnicity_diff_question']:.4f}")

    # Sort by ethnicity differential
    by_diff = sorted(head_scores, key=lambda x: x["ethnicity_diff_question"], reverse=True)
    print(f"\n  Top 10 heads by ethnicity-differential attention:")
    for i, h in enumerate(by_diff[:10]):
        print(f"    {i+1}. Layer {h['layer']:2d} Head {h['head']:2d}: "
              f"diff={h['ethnicity_diff_question']:.4f} "
              f"attn={h['mean_attn_to_name']:.4f}")

    # Sort by last-token attention to name
    by_last = sorted(head_scores, key=lambda x: x["mean_attn_last_to_name"], reverse=True)
    print(f"\n  Top 10 heads by last-token->name attention:")
    for i, h in enumerate(by_last[:10]):
        print(f"    {i+1}. Layer {h['layer']:2d} Head {h['head']:2d}: "
              f"last_attn={h['mean_attn_last_to_name']:.4f}")

    # --- Step 5: Ablation verification ---
    print(f"\n[5/6] Ablation verification (re-probing after zeroing top heads)...")

    layers_module = get_layers(model)

    # Baseline: extract hidden states at best probe layer (no ablation)
    print(f"  Extracting baseline hidden states at layer {BEST_PROBE_LAYER}...")
    captured = {}
    def capture_hook(module, input, output):
        if isinstance(output, tuple):
            captured['hs'] = output[0].detach().cpu()
        else:
            captured['hs'] = output.detach().cpu()

    target_layer_module = layers_module[BEST_PROBE_LAYER - 1] if BEST_PROBE_LAYER > 0 \
        else get_embed(model)

    handle_capture = target_layer_module.register_forward_hook(capture_hook)

    hs_baseline = []
    for text in tqdm(all_ethnicity_prompts, desc="Baseline HS"):
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=128).to("cuda")
        with torch.no_grad():
            model(**inputs)
        hs = captured['hs'].squeeze(0).float()
        hs_baseline.append(hs[-1].numpy())

    handle_capture.remove()
    hs_baseline = np.array(hs_baseline)

    # Train baseline probe
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score

    scaler_base = StandardScaler()
    X_base = scaler_base.fit_transform(hs_baseline)
    probe_base = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
    scores_base = cross_val_score(probe_base, X_base, ethnicity_labels, cv=5, scoring="accuracy")
    baseline_acc = scores_base.mean()
    print(f"  Baseline probe accuracy: {baseline_acc:.1%}")

    # Ablation experiments: zero top-N heads and re-probe
    ablation_results = []
    top_heads_by_attn = [(h["layer"], h["head"]) for h in by_attn]

    for n_ablate in [1, 3, 5, 10, 20]:
        heads_to_ablate = top_heads_by_attn[:n_ablate]
        print(f"\n  Ablating top {n_ablate} heads...")

        # Group heads by layer for efficient hooking
        heads_by_layer = {}
        for layer, head in heads_to_ablate:
            if layer not in heads_by_layer:
                heads_by_layer[layer] = []
            heads_by_layer[layer].append(head)

        # Register hooks on o_proj of affected layers
        ablation_handles = []
        for layer_idx, head_list in heads_by_layer.items():
            self_attn = layers_module[layer_idx].self_attn
            hook = make_head_ablation_hook(head_list, NUM_HEADS, HEAD_DIM)
            h = self_attn.o_proj.register_forward_pre_hook(hook)
            ablation_handles.append(h)

        # Re-capture with probe layer hook
        handle_capture = target_layer_module.register_forward_hook(capture_hook)

        hs_ablated = []
        for text in tqdm(all_ethnicity_prompts, desc=f"Ablated HS (top-{n_ablate})"):
            inputs = tok(text, return_tensors="pt", truncation=True, max_length=128).to("cuda")
            with torch.no_grad():
                model(**inputs)
            hs = captured['hs'].squeeze(0).float()
            hs_ablated.append(hs[-1].numpy())

        handle_capture.remove()
        for h in ablation_handles:
            h.remove()

        hs_ablated = np.array(hs_ablated)

        # Re-probe
        scaler_abl = StandardScaler()
        X_abl = scaler_abl.fit_transform(hs_ablated)
        probe_abl = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
        scores_abl = cross_val_score(probe_abl, X_abl, ethnicity_labels, cv=5, scoring="accuracy")
        ablated_acc = scores_abl.mean()

        acc_drop = baseline_acc - ablated_acc
        ablation_results.append({
            "n_heads_ablated": n_ablate,
            "heads": [{"layer": l, "head": h} for l, h in heads_to_ablate],
            "accuracy": float(ablated_acc),
            "accuracy_drop": float(acc_drop),
            "accuracy_drop_pct": float(acc_drop / baseline_acc * 100) if baseline_acc > 0 else 0,
        })
        print(f"  Top-{n_ablate}: acc={ablated_acc:.1%} "
              f"(D={acc_drop:+.1%}, {acc_drop/baseline_acc*100:+.1f}% relative)")

    # --- Step 6: Save results ---
    print(f"\n[6/6] Saving results...")

    results = {
        "model": MODEL_SHORT,
        "model_id": MODEL_ID,
        "comparison": COMPARISON,
        "ref_group": REF_GROUP,
        "cmp_group": CMP_GROUP,
        "best_probe_layer": BEST_PROBE_LAYER,
        "num_layers": num_layers,
        "num_heads": NUM_HEADS,
        "head_dim": HEAD_DIM,
        "n_attention_prompts": n_prompts,
        "baseline_probe_accuracy": float(baseline_acc),
        # Top heads rankings
        "top_heads_by_question_to_name_attn": [
            {k: v for k, v in h.items()} for h in by_attn[:20]
        ],
        "top_heads_by_ethnicity_differential": [
            {k: v for k, v in h.items()} for h in by_diff[:20]
        ],
        "top_heads_by_last_token_attn": [
            {k: v for k, v in h.items()} for h in by_last[:20]
        ],
        # Ablation results
        "ablation_results": ablation_results,
        # Full attention data (layer x head means)
        "mean_attn_to_name_per_head": mean_attn.tolist(),
        "ethnicity_diff_per_head": ethnicity_diff.tolist(),
        "mean_last_token_attn_per_head": mean_attn_last_token.tolist(),
    }

    path = RESULTS_DIR / f"{MODEL_SHORT}_circuit_tracing.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {path}")

    elapsed = (datetime.now() - t0).total_seconds() / 60
    print(f"\n{'=' * 70}")
    print(f"  RESULTS -- {MODEL_SHORT} ({COMPARISON})")
    print(f"{'=' * 70}")
    print(f"  Baseline probe: {baseline_acc:.1%}")
    for r in ablation_results:
        print(f"  Top-{r['n_heads_ablated']:2d} ablated: "
              f"{r['accuracy']:.1%} ({r['accuracy_drop_pct']:+.1f}%)")
    print(f"\n  Time: {elapsed:.1f} min")
    print(f"  VRAM peak: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    print(f"{'=' * 70}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
