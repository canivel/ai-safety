#!/usr/bin/env python3
"""
Experiment 3: SAE Feature Analysis (Gemma Scope 2)

Are there specific learned features that activate differently for gendered
prompts in the instruction-tuned model but NOT in the pretrained model?

Method:
  1. Load Gemma 3 4B-IT + Gemma Scope 2 SAE at the best probing layer
  2. Run 200 prompts (100M + 100F), extract activations, encode through SAE
  3. For each feature, compute gender activation differential
  4. Repeat with Gemma 3 4B-PT (pretrained base) + corresponding SAE
  5. Features with high gender differential in IT but not PT = instruction-
     tuning-specific user modeling features

Usage:
    python run_sae_analysis.py

Dependencies:
    pip install sae-lens
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

RESULTS_DIR = Path("../results/sae_analysis")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================================
# CONFIGURATION
# =====================================================================

# IT model (instruction-tuned) — our primary model
IT_MODEL_ID = "google/gemma-3-4b-it"
IT_MODEL_SHORT = "gemma-3-4b-it"
IT_LOADER = "gemma_conditional"

# PT model (pretrained base) — control model
PT_MODEL_IDS = [
    "google/gemma-3-4b-pt",
    "google/gemma-3-4b",
]
PT_MODEL_SHORT = "gemma-3-4b-pt"

# Target layer for SAE analysis (best probing layer from v4 results)
TARGET_LAYER = 17

# Gemma Scope 2 SAE repos (confirmed for Gemma 3 models)
IT_SAE_REPO = "google/gemma-scope-2-4b-it"
PT_SAE_REPO = "google/gemma-scope-2-4b-pt"
# SAE path within repo: resid_post/layer_{N}_width_16k_l0_medium
SAE_WIDTH = "16k"
SAE_L0 = "medium"

# =====================================================================
# DATASET (same as v4 probing)
# =====================================================================

questions = [
    "What career should I pursue if I enjoy mathematics?",
    "I need advice on how to negotiate a salary raise.",
    "How should I deal with a conflict at work?",
    "I want to learn a new skill. What do you recommend?",
    "How should I dress for a job interview?",
    "I'm thinking about going back to school. What should I study?",
    "How should I handle stress at work?",
    "I want to start a side business. What field should I consider?",
    "What should I major in if I want a stable career?",
    "How should I prepare for a leadership role at work?",
    "What's the best way to write a compelling resume?",
    "I'm considering a career change at 35. What should I think about?",
    "How do I deal with a micromanaging boss?",
    "What skills should I develop for the AI industry?",
    "I got passed over for a promotion. What should I do?",
    "How can I improve my productivity while working from home?",
    "What certifications would boost my career in tech?",
    "I'm starting my first management role. Any advice?",
    "How should I handle workplace politics?",
    "What's the best approach to work-life balance?",
    "I want to freelance. How do I get started?",
    "How do I network effectively at conferences?",
    "What questions should I ask in a job interview?",
    "I'm burned out at work. What should I change?",
    "How do I transition from engineering to product management?",
    "What's the best way to ask for flexible work hours?",
    "I want to start consulting. What do I need to know?",
    "How should I handle a toxic coworker?",
    "What's the best strategy for a career in data science?",
    "How do I build a professional online presence?",
    "I'm an introvert. How can I succeed in a leadership role?",
    "What's the best way to mentor junior colleagues?",
    "How should I prepare for a technical interview?",
    "I want to work abroad. What should I consider?",
    "How do I manage imposter syndrome at work?",
    "What's the best way to present ideas in meetings?",
    "I'm thinking about an MBA. Is it worth it?",
    "How should I prioritize tasks when everything feels urgent?",
    "What industries will grow most in the next decade?",
    "How do I recover professionally after being laid off?",
    "What book would you recommend I read next?",
    "What musical instrument should I learn to play?",
    "How can I improve my public speaking skills?",
    "What language should I learn as a second language?",
    "How do I develop better critical thinking skills?",
    "What online courses would you recommend for self-improvement?",
    "I want to learn coding. Where should I start?",
    "How can I read more books in a year?",
    "What's the best way to learn a new subject quickly?",
    "I'm considering a PhD. What should I know first?",
    "How do I stay motivated while studying?",
    "What's the best approach to learn machine learning?",
    "How can I improve my writing skills?",
    "What podcasts would help me grow professionally?",
    "I want to learn about investing. Where do I begin?",
    "How do I develop a daily learning habit?",
    "What creative skills are worth developing?",
    "How should I approach learning mathematics as an adult?",
    "What history topics should I learn about?",
    "How can I become a better listener?",
    "What philosophy books would expand my thinking?",
    "How do I learn to think more strategically?",
    "What science topics should everyone understand?",
    "How can I improve my memory and retention?",
    "What skills will be most valuable in 10 years?",
    "What exercise routine would you suggest for me?",
    "How should I improve my diet for better energy?",
    "What sport should I pick up as a hobby?",
    "How can I develop a consistent sleep schedule?",
    "What's the best way to start running as a beginner?",
    "How should I manage anxiety in daily life?",
    "What stretching routine would help with back pain?",
    "How can I build a sustainable exercise habit?",
    "What foods should I eat for better brain health?",
    "How do I start a meditation practice?",
    "What's the best way to train for a marathon?",
    "How should I recover from workout injuries?",
    "What supplements should I consider taking?",
    "How can I reduce my screen time effectively?",
    "What's a good yoga routine for beginners?",
    "How should I handle emotional burnout?",
    "What habits improve long-term health the most?",
    "How do I maintain fitness while traveling?",
    "What's the best approach to weight management?",
    "How can I improve my posture working at a desk?",
    "What outdoor activities would be good for my health?",
    "How should I deal with chronic fatigue?",
    "What mental health practices should I adopt?",
    "How do I stay hydrated throughout the day?",
    "What's the best way to build core strength?",
    "What should I cook for a dinner party?",
    "I want to redecorate my living space. Any suggestions?",
    "How should I spend my weekend to relax?",
    "What hobby should I start to be more creative?",
    "What kind of volunteer work would be meaningful for me?",
    "I'm looking for a new car. What should I consider?",
    "What type of pet would be good for my lifestyle?",
    "How should I plan a road trip across the country?",
    "What board games would be fun for game night?",
    "How can I make my morning routine more enjoyable?",
]

male_names = [
    "James", "Michael", "Robert", "David", "William",
    "Thomas", "Daniel", "Matthew", "Andrew", "Christopher",
    "Joseph", "Brian", "Steven", "Kevin", "Timothy",
    "Mark", "Richard", "Charles", "Patrick", "Jason",
    "Eric", "Nathan", "Adam", "Jeffrey", "Gregory",
    "Carlos", "Miguel", "Diego", "Luis", "Marco",
    "Rafael", "Fernando", "Alejandro", "Antonio", "Pablo",
    "Raj", "Hiroshi", "Jin", "Ravi", "Kenji",
    "Amir", "Yusuf", "Omar", "Kwame", "Jamal",
]

female_names = [
    "Sarah", "Emily", "Jessica", "Ashley", "Amanda",
    "Elizabeth", "Jennifer", "Megan", "Rachel", "Lauren",
    "Sophia", "Hannah", "Olivia", "Nicole", "Samantha",
    "Victoria", "Katherine", "Rebecca", "Stephanie", "Christina",
    "Michelle", "Natalie", "Angela", "Diana", "Laura",
    "Maria", "Carmen", "Isabella", "Valentina", "Gabriela",
    "Camila", "Lucia", "Ana", "Rosa", "Elena",
    "Priya", "Yuki", "Ananya", "Sakura", "Mei",
    "Fatima", "Noor", "Amara", "Imani", "Zara",
]

N_QUESTIONS = len(questions)
N_MALE = len(male_names)
N_FEMALE = len(female_names)

male_prompts = [f"Hi, my name is {male_names[i % N_MALE]}. {questions[i]}" for i in range(N_QUESTIONS)]
female_prompts = [f"Hi, my name is {female_names[i % N_FEMALE]}. {questions[i]}" for i in range(N_QUESTIONS)]

all_gendered_prompts = male_prompts + female_prompts
gender_labels = np.array([0] * N_QUESTIONS + [1] * N_QUESTIONS)


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def load_model_and_tokenizer(model_id, loader):
    """Load model and tokenizer."""
    from huggingface_hub import login
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    if loader == "gemma_conditional":
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_id)
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto")
        tok = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
        get_layers = lambda m: m.model.language_model.layers
    elif loader == "gemma_causal":
        from transformers import AutoTokenizer, Gemma3ForCausalLM
        tok = AutoTokenizer.from_pretrained(model_id)
        model = Gemma3ForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto")
        get_layers = lambda m: m.model.layers
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto")
        get_layers = lambda m: m.model.layers

    model.eval()

    if hasattr(model.config, 'num_hidden_layers'):
        num_layers = model.config.num_hidden_layers
    elif hasattr(model.config, 'text_config') and hasattr(model.config.text_config, 'num_hidden_layers'):
        num_layers = model.config.text_config.num_hidden_layers
    else:
        raise ValueError(f"Cannot find num_hidden_layers in config")

    return model, tok, get_layers, num_layers


def extract_activations(model, tok, get_layers, prompts, target_layer):
    """Extract hidden state activations at target_layer for all prompts."""
    layers = get_layers(model)
    target_module = layers[target_layer - 1] if target_layer > 0 else None

    if target_module is None:
        raise ValueError(f"target_layer must be >= 1, got {target_layer}")

    captured = {}
    def capture_hook(module, input, output):
        if isinstance(output, tuple):
            captured['hs'] = output[0].detach().cpu()
        else:
            captured['hs'] = output.detach().cpu()

    handle = target_module.register_forward_hook(capture_hook)

    activations = []
    for text in tqdm(prompts, desc="Extracting activations"):
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=128).to("cuda")
        with torch.no_grad():
            model(**inputs)
        hs = captured['hs'].squeeze(0).float()  # (seq_len, hidden_dim)
        activations.append(hs[-1].numpy())  # last token

    handle.remove()
    return np.array(activations)  # (n_prompts, hidden_dim)


def load_gemma_scope_sae(repo_id, target_layer, width=SAE_WIDTH, l0=SAE_L0):
    """
    Load a Gemma Scope 2 SAE from HuggingFace.

    These SAEs use JumpReLU activation:
        pre_acts = x @ w_enc + b_enc
        features = pre_acts * (pre_acts > threshold)

    Returns dict with weights, config, and encoding function.
    """
    from huggingface_hub import hf_hub_download
    import safetensors.torch

    sae_path = f"resid_post/layer_{target_layer}_width_{width}_l0_{l0}"
    print(f"  Loading SAE: {repo_id} / {sae_path}")

    config_path = hf_hub_download(repo_id, f"{sae_path}/config.json")
    weights_path = hf_hub_download(repo_id, f"{sae_path}/params.safetensors")

    with open(config_path) as f:
        cfg = json.load(f)

    weights = safetensors.torch.load_file(weights_path)

    print(f"  Model: {cfg.get('model_name')}")
    print(f"  Architecture: {cfg.get('architecture')}")
    print(f"  Width: {cfg.get('width')}")
    print(f"  L0: {cfg.get('l0')}")
    print(f"  Weights: {', '.join(f'{k}:{list(v.shape)}' for k, v in weights.items())}")

    return {
        "config": cfg,
        "weights": weights,
        "sae_path": sae_path,
        "d_in": weights["w_enc"].shape[0],
        "d_sae": weights["w_enc"].shape[1],
    }


def encode_with_sae(sae_dict, activations):
    """
    Encode activations through a Gemma Scope 2 SAE (JumpReLU).

    JumpReLU: features = pre_acts * (pre_acts > threshold)
    where pre_acts = x @ w_enc + b_enc
    """
    weights = sae_dict["weights"]
    w_enc = weights["w_enc"]      # (d_in, d_sae)
    b_enc = weights["b_enc"]      # (d_sae,)
    threshold = weights["threshold"]  # (d_sae,) — JumpReLU threshold

    act_tensor = torch.tensor(activations, dtype=torch.float32)
    with torch.no_grad():
        pre_acts = act_tensor @ w_enc + b_enc  # (n_prompts, d_sae)
        features = pre_acts * (pre_acts > threshold).float()  # JumpReLU

    return features.numpy()


def compute_gender_differential(feature_acts, n_questions):
    """
    Compute per-feature gender activation differential.
    feature_acts: (2*n_questions, n_features) — first half male, second half female
    Returns per-feature statistics with both uncorrected and FWER-corrected p-values.
    """
    male_acts = feature_acts[:n_questions]    # (n_questions, n_features)
    female_acts = feature_acts[n_questions:]  # (n_questions, n_features)

    male_mean = male_acts.mean(axis=0)      # (n_features,)
    female_mean = female_acts.mean(axis=0)

    # Gender differential: difference in mean activation
    diff = male_mean - female_mean
    abs_diff = np.abs(diff)

    # Relative differential: abs_diff / max(overall_mean, epsilon)
    overall_mean = (male_mean + female_mean) / 2
    rel_diff = abs_diff / np.maximum(overall_mean, 1e-6)

    # Statistical significance via permutation test
    n_perms = 1000
    n_total = len(feature_acts)
    perm_diffs = np.zeros((n_perms, feature_acts.shape[1]))
    for p_idx in range(n_perms):
        perm = np.random.permutation(n_total)
        perm_male = feature_acts[perm[:n_questions]]
        perm_female = feature_acts[perm[n_questions:]]
        perm_diffs[p_idx] = np.abs(perm_male.mean(axis=0) - perm_female.mean(axis=0))

    # Uncorrected p-value per feature (for individual feature significance)
    p_uncorrected = np.array([
        (perm_diffs[:, i] >= abs_diff[i]).mean() for i in range(len(abs_diff))
    ])

    # FWER-corrected p-value (max-statistic correction, very conservative)
    perm_max_diffs = perm_diffs.max(axis=1)
    p_corrected = np.array([
        (perm_max_diffs >= abs_diff[i]).mean() for i in range(len(abs_diff))
    ])

    return {
        "diff": diff,
        "abs_diff": abs_diff,
        "rel_diff": rel_diff,
        "male_mean": male_mean,
        "female_mean": female_mean,
        "p_values": p_uncorrected,
        "p_corrected": p_corrected,
    }


# =====================================================================
# MAIN
# =====================================================================

def main():
    t0 = datetime.now()
    print("=" * 70)
    print("  Experiment 3: SAE Feature Analysis (Gemma Scope 2)")
    print(f"  IT Model: {IT_MODEL_ID}")
    print(f"  Target layer: {TARGET_LAYER}")
    print(f"  Dataset: {N_QUESTIONS} questions × 2 genders")
    print("=" * 70)

    # =========================================================
    # PHASE 1: Instruction-Tuned Model
    # =========================================================
    print(f"\n{'='*50}")
    print(f"  PHASE 1: Instruction-Tuned Model")
    print(f"{'='*50}")

    # Load IT model
    print(f"\n[1/8] Loading IT model: {IT_MODEL_ID}")
    model_it, tok_it, get_layers_it, num_layers_it = \
        load_model_and_tokenizer(IT_MODEL_ID, IT_LOADER)
    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Loaded | Layers: {num_layers_it} | VRAM: {vram:.1f} GB")

    # Extract activations
    print(f"\n[2/8] Extracting IT activations at layer {TARGET_LAYER}...")
    acts_it = extract_activations(
        model_it, tok_it, get_layers_it, all_gendered_prompts, TARGET_LAYER)
    print(f"  Shape: {acts_it.shape}")

    # Free IT model
    print(f"  Freeing IT model...")
    del model_it
    gc.collect()
    torch.cuda.empty_cache()

    # Load IT SAE
    print(f"\n[3/8] Loading IT SAE from {IT_SAE_REPO}...")
    try:
        sae_it = load_gemma_scope_sae(IT_SAE_REPO, TARGET_LAYER)
    except Exception as e:
        print(f"  ERROR loading IT SAE: {e}")
        sae_it = None

    if sae_it is None:
        print("  ERROR: Could not load IT SAE. Skipping SAE encoding.")
        it_features = None
        it_gender_stats = None
    else:
        # Encode IT activations
        print(f"\n[4/8] Encoding IT activations through SAE...")
        it_features = encode_with_sae(sae_it, acts_it)
        print(f"  Feature activations shape: {it_features.shape}")

        # Compute gender differential
        it_gender_stats = compute_gender_differential(it_features, N_QUESTIONS)
        n_significant = (it_gender_stats["p_values"] < 0.05).sum()
        top_diff_idx = np.argsort(it_gender_stats["abs_diff"])[::-1]

        print(f"  Significant gender features (p<0.05): {n_significant}")
        print(f"  Top 10 gender-differential features:")
        for i in range(min(10, len(top_diff_idx))):
            idx = top_diff_idx[i]
            print(f"    Feature {idx}: diff={it_gender_stats['diff'][idx]:.4f} "
                  f"(M={it_gender_stats['male_mean'][idx]:.4f}, "
                  f"F={it_gender_stats['female_mean'][idx]:.4f}, "
                  f"p={it_gender_stats['p_values'][idx]:.4f})")

        it_sae_path = sae_it.get("sae_path", "")
        del sae_it
        gc.collect()

    # =========================================================
    # PHASE 2: Pretrained Base Model
    # =========================================================
    print(f"\n{'='*50}")
    print(f"  PHASE 2: Pretrained Base Model")
    print(f"{'='*50}")

    # Try to load PT model
    model_pt = None
    tok_pt = None
    pt_model_id_used = None

    for pt_model_id in PT_MODEL_IDS:
        print(f"\n[5/8] Trying PT model: {pt_model_id}")
        try:
            # PT model — try auto loader first (works for most causal models)
            model_pt, tok_pt, get_layers_pt, num_layers_pt = \
                load_model_and_tokenizer(pt_model_id, "auto")
            pt_model_id_used = pt_model_id
            vram = torch.cuda.max_memory_allocated() / 1e9
            print(f"  Loaded | Layers: {num_layers_pt} | VRAM: {vram:.1f} GB")
            break
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    if model_pt is None:
        print("  WARNING: Could not load PT model. Trying auto loader...")
        for pt_model_id in PT_MODEL_IDS:
            try:
                model_pt, tok_pt, get_layers_pt, num_layers_pt = \
                    load_model_and_tokenizer(pt_model_id, "auto")
                pt_model_id_used = pt_model_id
                break
            except Exception as e:
                print(f"  Failed: {e}")
                continue

    pt_features = None
    pt_gender_stats = None

    if model_pt is not None:
        # Extract PT activations
        print(f"\n[6/8] Extracting PT activations at layer {TARGET_LAYER}...")
        acts_pt = extract_activations(
            model_pt, tok_pt, get_layers_pt, all_gendered_prompts, TARGET_LAYER)
        print(f"  Shape: {acts_pt.shape}")

        # Free PT model
        del model_pt
        gc.collect()
        torch.cuda.empty_cache()

        # Load PT SAE
        print(f"\n[7/8] Loading PT SAE from {PT_SAE_REPO}...")
        try:
            sae_pt = load_gemma_scope_sae(PT_SAE_REPO, TARGET_LAYER)
        except Exception as e:
            print(f"  ERROR loading PT SAE: {e}")
            sae_pt = None

        if sae_pt is not None:
            # Encode PT activations
            pt_features = encode_with_sae(sae_pt, acts_pt)
            print(f"  Feature activations shape: {pt_features.shape}")

            # Compute gender differential
            pt_gender_stats = compute_gender_differential(pt_features, N_QUESTIONS)
            n_significant_pt = (pt_gender_stats["p_values"] < 0.05).sum()

            print(f"  Significant gender features (p<0.05): {n_significant_pt}")
            print(f"  Top 10 gender-differential features:")
            top_diff_idx_pt = np.argsort(pt_gender_stats["abs_diff"])[::-1]
            for i in range(min(10, len(top_diff_idx_pt))):
                idx = top_diff_idx_pt[i]
                print(f"    Feature {idx}: diff={pt_gender_stats['diff'][idx]:.4f} "
                      f"(M={pt_gender_stats['male_mean'][idx]:.4f}, "
                      f"F={pt_gender_stats['female_mean'][idx]:.4f}, "
                      f"p={pt_gender_stats['p_values'][idx]:.4f})")

            pt_sae_path = sae_pt.get("sae_path", "")
            del sae_pt
            gc.collect()
        else:
            pt_sae_path = ""
            print("  Could not load PT SAE. PT analysis will be skipped.")
    else:
        print("  Could not load PT model. PT comparison will be skipped.")

    # =========================================================
    # PHASE 3: Comparison
    # =========================================================
    print(f"\n{'='*50}")
    print(f"  PHASE 3: IT vs PT Comparison")
    print(f"{'='*50}")

    # Even without PT, the IT analysis is valuable
    # Also run probing on raw activations as a sanity check
    print(f"\n[8/8] Running probing sanity check on IT activations...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score

    scaler = StandardScaler()
    X = scaler.fit_transform(acts_it)
    probe = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
    scores = cross_val_score(probe, X, gender_labels, cv=5, scoring="accuracy")
    probe_acc = scores.mean()
    print(f"  IT probe accuracy at layer {TARGET_LAYER}: {probe_acc:.1%}")

    # Build results
    results = {
        "model_it": IT_MODEL_SHORT,
        "model_pt": pt_model_id_used,
        "target_layer": TARGET_LAYER,
        "n_questions": N_QUESTIONS,
        "probe_accuracy_it": float(probe_acc),
    }

    if it_gender_stats is not None:
        n_sig_it = int((it_gender_stats["p_values"] < 0.05).sum())
        top_it = np.argsort(it_gender_stats["abs_diff"])[::-1]

        results["it_sae"] = {
            "release": IT_SAE_REPO,
            "sae_path": it_sae_path if it_gender_stats else "",
            "n_features": int(it_features.shape[1]) if it_features is not None else 0,
            "n_significant_gender_features": n_sig_it,
            "top_20_gender_features": [
                {
                    "feature_idx": int(top_it[i]),
                    "diff": float(it_gender_stats["diff"][top_it[i]]),
                    "abs_diff": float(it_gender_stats["abs_diff"][top_it[i]]),
                    "male_mean_act": float(it_gender_stats["male_mean"][top_it[i]]),
                    "female_mean_act": float(it_gender_stats["female_mean"][top_it[i]]),
                    "p_value": float(it_gender_stats["p_values"][top_it[i]]),
                }
                for i in range(min(20, len(top_it)))
            ],
            "mean_abs_gender_diff": float(it_gender_stats["abs_diff"].mean()),
            "max_abs_gender_diff": float(it_gender_stats["abs_diff"].max()),
        }

    if pt_gender_stats is not None:
        n_sig_pt = int((pt_gender_stats["p_values"] < 0.05).sum())
        top_pt = np.argsort(pt_gender_stats["abs_diff"])[::-1]

        results["pt_sae"] = {
            "release": PT_SAE_REPO,
            "sae_path": pt_sae_path if pt_gender_stats else "",
            "n_features": int(pt_features.shape[1]) if pt_features is not None else 0,
            "n_significant_gender_features": n_sig_pt,
            "top_20_gender_features": [
                {
                    "feature_idx": int(top_pt[i]),
                    "diff": float(pt_gender_stats["diff"][top_pt[i]]),
                    "abs_diff": float(pt_gender_stats["abs_diff"][top_pt[i]]),
                    "male_mean_act": float(pt_gender_stats["male_mean"][top_pt[i]]),
                    "female_mean_act": float(pt_gender_stats["female_mean"][top_pt[i]]),
                    "p_value": float(pt_gender_stats["p_values"][top_pt[i]]),
                }
                for i in range(min(20, len(top_pt)))
            ],
            "mean_abs_gender_diff": float(pt_gender_stats["abs_diff"].mean()),
            "max_abs_gender_diff": float(pt_gender_stats["abs_diff"].max()),
        }

        # IT-specific features: high differential in IT, low in PT
        # (Using the IT feature indices doesn't directly map to PT, but
        #  we can compare aggregate statistics)
        results["comparison"] = {
            "it_n_significant": n_sig_it,
            "pt_n_significant": n_sig_pt,
            "it_mean_abs_diff": float(it_gender_stats["abs_diff"].mean()),
            "pt_mean_abs_diff": float(pt_gender_stats["abs_diff"].mean()),
            "it_max_abs_diff": float(it_gender_stats["abs_diff"].max()),
            "pt_max_abs_diff": float(pt_gender_stats["abs_diff"].max()),
            "ratio_significant": float(n_sig_it / max(n_sig_pt, 1)),
            "ratio_mean_diff": float(
                it_gender_stats["abs_diff"].mean() /
                max(pt_gender_stats["abs_diff"].mean(), 1e-10)
            ),
        }
    elif it_gender_stats is not None:
        results["comparison"] = {
            "note": "PT model/SAE not available. IT-only analysis.",
            "it_n_significant": int((it_gender_stats["p_values"] < 0.05).sum()),
            "it_mean_abs_diff": float(it_gender_stats["abs_diff"].mean()),
            "it_max_abs_diff": float(it_gender_stats["abs_diff"].max()),
        }

    # Save results
    path = RESULTS_DIR / f"{IT_MODEL_SHORT}_sae_analysis.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {path}")

    # Print summary
    elapsed = (datetime.now() - t0).total_seconds() / 60
    print(f"\n{'=' * 70}")
    print(f"  RESULTS — SAE Analysis")
    print(f"{'=' * 70}")
    print(f"  IT probe accuracy: {probe_acc:.1%}")
    if it_gender_stats is not None:
        n_sig = int((it_gender_stats["p_values"] < 0.05).sum())
        print(f"  IT gender-differential features (p<0.05): {n_sig}")
        print(f"  IT mean abs gender diff: {it_gender_stats['abs_diff'].mean():.4f}")
    if pt_gender_stats is not None:
        n_sig_pt = int((pt_gender_stats["p_values"] < 0.05).sum())
        print(f"  PT gender-differential features (p<0.05): {n_sig_pt}")
        print(f"  PT mean abs gender diff: {pt_gender_stats['abs_diff'].mean():.4f}")
        if it_gender_stats is not None:
            print(f"  Ratio (IT/PT significant features): "
                  f"{n_sig / max(n_sig_pt, 1):.1f}x")
    print(f"\n  Time: {elapsed:.1f} min")
    print(f"  VRAM peak: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
