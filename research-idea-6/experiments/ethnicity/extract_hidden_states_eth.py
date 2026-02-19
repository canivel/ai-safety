#!/usr/bin/env python3
"""
GPU-only: Extract hidden states + steering KL divergences for ETHNICITY probing.
Run this on RunPod, then download .npz files for local CPU analysis.

Performs pairwise binary comparisons (e.g., White vs Black names).

Outputs per model per comparison:
  - {model_short}_hidden_states.npz   (hidden states for all layers)
  - {model_short}_steering.json       (steering KL values)

Ethnicity categories aligned with EEOC federal standards (OMB race/ethnicity).

Usage:
    python extract_hidden_states_eth.py gemma4b white_vs_black
    python extract_hidden_states_eth.py qwen7b white_vs_hispanic
    python extract_hidden_states_eth.py gemma12b white_vs_asian
    python extract_hidden_states_eth.py mistral7b white_vs_native_american
    python extract_hidden_states_eth.py gemma1b white_vs_pacific_islander
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

# Force unbuffered output (critical for RunPod SSH monitoring)
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Add shared data directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.load_data import (
    load_questions,
    load_ethnicity_names,
    load_model_registry,
    get_ethnicity_comparison,
    VALID_COMPARISONS,
)


# === CLI PARSING ===
if len(sys.argv) < 3:
    print(f"Usage: python extract_hidden_states_eth.py <model_key> <comparison>")
    print(f"  Models: see shared/model_registry.json")
    print(f"  Comparisons: {VALID_COMPARISONS}")
    sys.exit(1)

MODEL_KEY = sys.argv[1]
COMPARISON = sys.argv[2]

registry = load_model_registry()
if MODEL_KEY not in registry:
    print(f"ERROR: Unknown model '{MODEL_KEY}'. Choose from: {list(registry.keys())}")
    sys.exit(1)

if COMPARISON not in VALID_COMPARISONS:
    print(f"ERROR: Unknown comparison '{COMPARISON}'. Choose from: {VALID_COMPARISONS}")
    sys.exit(1)

MODEL_CFG = registry[MODEL_KEY]
MODEL_ID = MODEL_CFG["model_id"]
MODEL_SHORT = MODEL_CFG["model_short"]
LOADER = MODEL_CFG.get("loader", "auto")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "ethnicity_probing" / COMPARISON
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# === LOAD DATASET ===
questions = load_questions()
eth_names = load_ethnicity_names()
ref_group, cmp_group = get_ethnicity_comparison(COMPARISON)

ref_data = eth_names[ref_group]
cmp_data = eth_names[cmp_group]
amb_data = eth_names["ambiguous"]

ref_names = ref_data["names"]
cmp_names = cmp_data["names"]
ref_genders = ref_data["gender"]
cmp_genders = cmp_data["gender"]
amb_names = amb_data["names"]

N_QUESTIONS = len(questions)
N_REF = len(ref_names)
N_CMP = len(cmp_names)
N_AMBIGUOUS = len(amb_names)

# Build prompts (same format as gender study)
ref_prompts = [f"Hi, my name is {ref_names[i % N_REF]}. {questions[i]}" for i in range(N_QUESTIONS)]
cmp_prompts = [f"Hi, my name is {cmp_names[i % N_CMP]}. {questions[i]}" for i in range(N_QUESTIONS)]
all_eth_prompts = ref_prompts + cmp_prompts
eth_labels = np.array([0] * N_QUESTIONS + [1] * N_QUESTIONS)  # 0=ref, 1=cmp

# Ambiguous prompts
ambiguous_prompts = [f"Hi, my name is {amb_names[i % N_AMBIGUOUS]}. {questions[i]}" for i in range(N_AMBIGUOUS)]

# Gender annotations for each prompt (for combined analysis later)
prompt_genders = []
for i in range(N_QUESTIONS):
    prompt_genders.append(ref_genders[i % N_REF])
for i in range(N_QUESTIONS):
    prompt_genders.append(cmp_genders[i % N_CMP])


def symmetric_kl(logits_a, logits_b):
    """Symmetric KL divergence between two logit distributions."""
    p = F.softmax(logits_a.float(), dim=-1).clamp(min=1e-10)
    q = F.softmax(logits_b.float(), dim=-1).clamp(min=1e-10)
    kl_pq = (p * (p.log() - q.log())).sum()
    kl_qp = (q * (q.log() - p.log())).sum()
    return ((kl_pq + kl_qp) / 2).item()


def load_model_and_tokenizer(model_id, loader):
    """Load model and tokenizer with appropriate backend."""
    from huggingface_hub import login
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    if loader == "gemma_causal":
        from transformers import AutoTokenizer, Gemma3ForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = Gemma3ForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto")
        tok = tokenizer
        get_layers = lambda m: m.model.layers
        get_embed = lambda m: m.model.embed_tokens
    elif loader == "gemma_conditional":
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration
        tokenizer = AutoProcessor.from_pretrained(model_id)
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto")
        tok = tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else tokenizer
        get_layers = lambda m: m.model.language_model.layers
        get_embed = lambda m: m.model.language_model.embed_tokens
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto")
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
    return model, tok, get_layers, get_embed, num_layers


def main():
    t0 = datetime.now()
    print("=" * 70)
    print("  GPU Extraction — Ethnicity Hidden States + Steering")
    print(f"  Model: {MODEL_ID}")
    print(f"  Comparison: {COMPARISON} ({ref_group} vs {cmp_group})")
    print(f"  Dataset: {N_QUESTIONS}Q, {N_REF} {ref_group}/{N_CMP} {cmp_group}/{N_AMBIGUOUS} ambiguous names")
    print("=" * 70)

    # --- Load model ---
    print(f"\n[1/5] Loading model...")
    model, tok, get_layers, get_embed, num_layers = load_model_and_tokenizer(MODEL_ID, LOADER)
    total_layers = num_layers + 1  # +1 for embedding layer

    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Loaded | Layers: {num_layers} | VRAM: {vram:.1f} GB")

    # --- Find name boundaries ---
    print(f"\n[2/5] Finding name boundaries...")
    name_boundaries = []

    for i in range(len(all_eth_prompts)):
        q_idx = i if i < N_QUESTIONS else i - N_QUESTIONS
        this_text = all_eth_prompts[i]
        pair_text = cmp_prompts[q_idx] if i < N_QUESTIONS else ref_prompts[q_idx]

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
        name_boundaries.append((name_start, name_end))

        if i == 0 or i == N_QUESTIONS:
            name_tok_ids = this_ids[name_start:name_end]
            decoded = tok.decode(name_tok_ids) if name_tok_ids else "<empty>"
            label = ref_group if i < N_QUESTIONS else cmp_group
            print(f"  [{label}] name_pos=[{name_start}:{name_end}], decoded='{decoded}'")

    name_lens = [nb[1] - nb[0] for nb in name_boundaries]
    print(f"  Name token lengths: {sorted(set(name_lens))} (unique)")

    # =========================================================
    # EXTRACT HIDDEN STATES — ethnicity prompts
    # =========================================================
    print(f"\n[3/5] Extracting hidden states ({len(all_eth_prompts)} ethnicity prompts)...")

    hs_last_token = {layer: [] for layer in range(total_layers)}
    hs_mean_question = {layer: [] for layer in range(total_layers)}

    for i, text in enumerate(tqdm(all_eth_prompts, desc="Ethnicity HS")):
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=128).to("cuda")
        seq_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        name_start, name_end = name_boundaries[i]
        question_start = min(name_end, seq_len)

        for layer_idx, hs in enumerate(outputs.hidden_states):
            h = hs.squeeze(0).float().cpu()

            # Last token
            hs_last_token[layer_idx].append(h[-1].numpy())

            # Question-only: exclude BOS and name tokens
            has_bos = 1 if name_start > 0 else 0
            mask = list(range(has_bos, name_start)) + list(range(question_start, seq_len))
            if len(mask) > 0:
                hs_mean_question[layer_idx].append(h[mask].mean(dim=0).numpy())
            else:
                hs_mean_question[layer_idx].append(h.mean(dim=0).numpy())

    # Stack into arrays
    for layer in range(total_layers):
        hs_last_token[layer] = np.array(hs_last_token[layer])
        hs_mean_question[layer] = np.array(hs_mean_question[layer])

    n_samples, hidden_dim = hs_last_token[0].shape
    print(f"  Shape: ({n_samples}, {hidden_dim}) | {total_layers} layers")

    # =========================================================
    # EXTRACT HIDDEN STATES — ambiguous prompts (all layers)
    # =========================================================
    print(f"\n[4/5] Extracting ambiguous hidden states ({N_AMBIGUOUS} prompts)...")

    hs_ambiguous = {layer: [] for layer in range(total_layers)}

    for i, text in enumerate(tqdm(ambiguous_prompts, desc="Ambiguous HS")):
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=128).to("cuda")
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        for layer_idx, hs in enumerate(outputs.hidden_states):
            h = hs.squeeze(0).float().cpu()
            hs_ambiguous[layer_idx].append(h[-1].numpy())

    for layer in range(total_layers):
        hs_ambiguous[layer] = np.array(hs_ambiguous[layer])

    # =========================================================
    # STEERING + KL DIVERGENCES (needs GPU)
    # =========================================================
    print(f"\n[5/5] Steering ablation + KL divergences...")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score

    # Quick probe at each layer to find best layer for steering
    print("  Finding best probe layer (quick scan)...")
    best_acc = 0.0
    best_layer = 0
    for layer in range(total_layers):
        scaler = StandardScaler()
        X_s = scaler.fit_transform(hs_last_token[layer])
        probe = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
        scores = cross_val_score(probe, X_s, eth_labels, cv=5, scoring="accuracy")
        acc = scores.mean()
        if acc > best_acc:
            best_acc = acc
            best_layer = layer
    print(f"  Best probe layer: {best_layer} ({best_acc:.1%})")

    # Get ethnicity direction at best layer
    scaler_steer = StandardScaler()
    X_steer_s = scaler_steer.fit_transform(hs_last_token[best_layer])
    probe_steer = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
    probe_steer.fit(X_steer_s, eth_labels)
    eth_direction = probe_steer.coef_[0] / scaler_steer.scale_
    eth_direction = eth_direction / np.linalg.norm(eth_direction)
    eth_dir_tensor = torch.tensor(eth_direction, dtype=torch.bfloat16).to("cuda")

    layers_module = get_layers(model)
    embed_module = get_embed(model)

    steering_data = {
        "comparison": COMPARISON,
        "ref_group": ref_group,
        "cmp_group": cmp_group,
        "steer_layer": best_layer,
        "best_probe_acc": float(best_acc),
    }

    steer_q_indices = list(range(0, min(50, N_QUESTIONS)))

    for strength in [0.0, 1.0, 2.0, 5.0, 10.0]:
        kl_cross = []
        kl_same = []

        for i in steer_q_indices:
            ref_text = ref_prompts[i]
            cmp_text = cmp_prompts[i]
            # Same-group comparison: second name from reference group
            ref_text_b = f"Hi, my name is {ref_names[(i + 1) % N_REF]}. {questions[i]}"

            def make_hook(direction, s):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        hs = output[0]
                    else:
                        hs = output
                    proj = torch.einsum('...d,d->...', hs.float(), direction.float())
                    hs_modified = hs.float() - s * proj.unsqueeze(-1) * direction.float()
                    if isinstance(output, tuple):
                        return (hs_modified.to(hs.dtype),) + output[1:]
                    return hs_modified.to(hs.dtype)
                return hook_fn

            if best_layer > 0:
                target_module = layers_module[best_layer - 1]
            else:
                target_module = embed_module

            if strength > 0:
                handle = target_module.register_forward_hook(make_hook(eth_dir_tensor, strength))

            inputs_r = tok(ref_text, return_tensors="pt", truncation=True, max_length=128).to("cuda")
            inputs_c = tok(cmp_text, return_tensors="pt", truncation=True, max_length=128).to("cuda")
            inputs_rb = tok(ref_text_b, return_tensors="pt", truncation=True, max_length=128).to("cuda")

            with torch.no_grad():
                logits_r = model(**inputs_r).logits[0, -1, :]
                logits_c = model(**inputs_c).logits[0, -1, :]
                logits_rb = model(**inputs_rb).logits[0, -1, :]

            kl_cross.append(symmetric_kl(logits_r, logits_c))
            kl_same.append(symmetric_kl(logits_r, logits_rb))

            if strength > 0:
                handle.remove()

        mean_cross = float(np.mean(kl_cross))
        mean_same = float(np.mean(kl_same))
        ratio = mean_cross / mean_same if mean_same > 0 else 0

        steering_data[f"strength_{strength}"] = {
            "cross_ethnicity_kl": mean_cross,
            "same_ethnicity_kl": mean_same,
            "ratio": ratio,
            "kl_cross_values": kl_cross,
            "kl_same_values": kl_same,
        }
        print(f"  alpha={strength:.1f}: cross={mean_cross:.6f}, same={mean_same:.6f}, ratio={ratio:.2f}x")

    # KL between ambiguous name pairs
    print("  KL between ambiguous pairs...")
    amb_kl_values = []
    for i in range(0, N_AMBIGUOUS - 1, 2):
        text_a = ambiguous_prompts[i]
        text_b = ambiguous_prompts[i + 1]
        inputs_a = tok(text_a, return_tensors="pt", truncation=True, max_length=128).to("cuda")
        inputs_b = tok(text_b, return_tensors="pt", truncation=True, max_length=128).to("cuda")
        with torch.no_grad():
            logits_a = model(**inputs_a).logits[0, -1, :]
            logits_b = model(**inputs_b).logits[0, -1, :]
        amb_kl_values.append(symmetric_kl(logits_a, logits_b))
    steering_data["amb_kl_values"] = amb_kl_values
    steering_data["amb_kl_mean"] = float(np.mean(amb_kl_values))
    print(f"  Ambiguous KL: {steering_data['amb_kl_mean']:.6f}")

    # =========================================================
    # SAVE .npz FILES
    # =========================================================
    print("\nSaving files...")

    # Hidden states: save per-layer arrays
    hs_save = {}
    for layer in range(total_layers):
        hs_save[f"last_token_layer_{layer}"] = hs_last_token[layer]
        hs_save[f"question_only_layer_{layer}"] = hs_mean_question[layer]
        hs_save[f"ambiguous_layer_{layer}"] = hs_ambiguous[layer]
    hs_save["eth_labels"] = eth_labels
    hs_save["total_layers"] = np.array([total_layers])
    hs_save["hidden_dim"] = np.array([hidden_dim])
    hs_save["name_boundaries"] = np.array(name_boundaries)
    # Gender annotations for combined analysis
    gender_codes = np.array([1 if g == "F" else 0 for g in prompt_genders])
    hs_save["prompt_genders"] = gender_codes

    hs_path = RESULTS_DIR / f"{MODEL_SHORT}_hidden_states.npz"
    np.savez_compressed(hs_path, **hs_save)
    hs_size = hs_path.stat().st_size / 1e6
    print(f"  Hidden states: {hs_path} ({hs_size:.1f} MB)")

    # Steering data: save as JSON
    steer_path = RESULTS_DIR / f"{MODEL_SHORT}_steering.json"
    with open(steer_path, "w") as f:
        json.dump(steering_data, f, indent=2)
    print(f"  Steering: {steer_path}")

    # =========================================================
    # CLEANUP
    # =========================================================
    elapsed = (datetime.now() - t0).total_seconds() / 60
    print(f"\n{'=' * 70}")
    print(f"  DONE — {MODEL_SHORT} / {COMPARISON}")
    print(f"  Time: {elapsed:.1f} min")
    print(f"  Files: {hs_path.name}, {steer_path.name}")
    print(f"{'=' * 70}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
