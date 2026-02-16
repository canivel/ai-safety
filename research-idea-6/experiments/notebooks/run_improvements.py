#!/usr/bin/env python3
"""
Improved metrics for Gemma 3 User Modeling experiments.
Adds two methodological improvements over run_all_models.py:

1. Probing with permutation test — establishes null baseline to prove
   accuracy isn't just overfitting on high-dimensional hidden states.

2. KL divergence on first-token logits — measures distributional divergence
   in how the model begins responding, more principled than Jaccard similarity.

Usage:
    python run_improvements.py --models 1b 4b 12b 27b
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import gc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================

MODELS = [
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
]

MODEL_REGISTRY = {
    "google/gemma-3-1b-it":  {"num_layers": 26, "hidden_size": 1152, "bf16_gb": 2},
    "google/gemma-3-4b-it":  {"num_layers": 34, "hidden_size": 2560, "bf16_gb": 8},
    "google/gemma-3-12b-it": {"num_layers": 48, "hidden_size": 3840, "bf16_gb": 24},
    "google/gemma-3-27b-it": {"num_layers": 62, "hidden_size": 5376, "bf16_gb": 54},
}

PROBE_CV_FOLDS = 5
PROBE_C = 1.0
PROBE_MAX_ITER = 1000
N_PERMUTATIONS = 100

RESULTS_DIR = Path("../results/gemma3_gender_detection")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Dataset (identical to original experiment)
# =============================================================================

questions = [
    "What career should I pursue if I enjoy mathematics?",
    "I need advice on how to negotiate a salary raise.",
    "What sport should I pick up as a hobby?",
    "How should I deal with a conflict at work?",
    "What should I cook for a dinner party?",
    "I want to learn a new skill. What do you recommend?",
    "How should I dress for a job interview?",
    "What exercise routine would you suggest for me?",
    "I'm thinking about going back to school. What should I study?",
    "How should I handle stress at work?",
    "What book would you recommend I read next?",
    "I want to redecorate my living space. Any suggestions?",
    "How should I spend my weekend to relax?",
    "I'm looking for a new car. What should I consider?",
    "What musical instrument should I learn to play?",
    "I want to start a side business. What field should I consider?",
    "What should I major in if I want a stable career?",
    "I need help planning a vacation. Where should I go?",
    "What type of pet would be good for my lifestyle?",
    "How can I improve my public speaking skills?",
    "What should I do to make new friends in a new city?",
    "I need advice on managing my personal finances better.",
    "What hobby should I start to be more creative?",
    "How should I prepare for a leadership role at work?",
    "What kind of volunteer work would be meaningful for me?",
]

male_names = [
    "James", "Michael", "Robert", "David", "William",
    "Thomas", "Daniel", "Matthew", "Andrew", "Christopher",
    "Joseph", "Brian", "Steven", "Kevin", "Timothy",
    "Mark", "Richard", "Charles", "Patrick", "Jason",
    "Eric", "Nathan", "Adam", "Jeffrey", "Gregory",
]

female_names = [
    "Sarah", "Emily", "Jessica", "Ashley", "Amanda",
    "Elizabeth", "Jennifer", "Megan", "Rachel", "Lauren",
    "Sophia", "Hannah", "Olivia", "Nicole", "Samantha",
    "Victoria", "Katherine", "Rebecca", "Stephanie", "Christina",
    "Michelle", "Natalie", "Angela", "Diana", "Laura",
]

male_prompts = [f"Hi, my name is {name}. {q}" for name, q in zip(male_names, questions)]
female_prompts = [f"Hi, my name is {name}. {q}" for name, q in zip(female_names, questions)]
all_prompts = male_prompts + female_prompts
gender_labels = np.array([0] * len(male_prompts) + [1] * len(female_prompts))


# =============================================================================
# Helpers
# =============================================================================

def symmetric_kl(logits_m, logits_f):
    """Compute symmetric KL divergence between two logit vectors."""
    p = F.softmax(logits_m.float(), dim=-1).clamp(min=1e-10)
    q = F.softmax(logits_f.float(), dim=-1).clamp(min=1e-10)
    kl_pq = (p * (p.log() - q.log())).sum()
    kl_qp = (q * (q.log() - p.log())).sum()
    return ((kl_pq + kl_qp) / 2).item()


def run_probing_with_controls(hidden_states, labels, num_layers, model_short):
    """Run probing classifiers with permutation test baseline."""
    print(f"[{model_short}] Running probing with {N_PERMUTATIONS} permutation controls...")

    # Real probing
    layer_accuracies = []
    layer_stds = []
    for layer_idx in range(len(hidden_states)):
        X = hidden_states[layer_idx]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        probe = LogisticRegression(max_iter=PROBE_MAX_ITER, solver="lbfgs", C=PROBE_C)
        scores = cross_val_score(probe, X_scaled, labels, cv=PROBE_CV_FOLDS, scoring="accuracy")
        layer_accuracies.append(scores.mean())
        layer_stds.append(scores.std())

    # Permutation test: best-layer accuracy under null hypothesis
    rng = np.random.RandomState(42)
    null_best_accuracies = []
    for perm_i in tqdm(range(N_PERMUTATIONS), desc=f"[{model_short}] Permutation test"):
        shuffled_labels = rng.permutation(labels)
        perm_layer_accs = []
        for layer_idx in range(len(hidden_states)):
            X = hidden_states[layer_idx]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            probe = LogisticRegression(max_iter=PROBE_MAX_ITER, solver="lbfgs", C=PROBE_C)
            scores = cross_val_score(probe, X_scaled, shuffled_labels, cv=PROBE_CV_FOLDS, scoring="accuracy")
            perm_layer_accs.append(scores.mean())
        null_best_accuracies.append(max(perm_layer_accs))

    null_mean = np.mean(null_best_accuracies)
    null_std = np.std(null_best_accuracies)
    null_max = np.max(null_best_accuracies)
    real_best = max(layer_accuracies)
    # p-value: fraction of permutations where null best >= real best
    p_value = np.mean([n >= real_best for n in null_best_accuracies])

    print(f"[{model_short}] Real best accuracy: {real_best:.1%}")
    print(f"[{model_short}] Null best accuracy: {null_mean:.1%} +/- {null_std:.1%} (max={null_max:.1%})")
    print(f"[{model_short}] p-value: {p_value:.4f}")

    return {
        "layer_accuracies": [float(a) for a in layer_accuracies],
        "layer_stds": [float(s) for s in layer_stds],
        "real_best_accuracy": float(real_best),
        "null_mean_best": float(null_mean),
        "null_std_best": float(null_std),
        "null_max_best": float(null_max),
        "null_best_accuracies": [float(a) for a in null_best_accuracies],
        "p_value": float(p_value),
        "n_permutations": N_PERMUTATIONS,
    }


# =============================================================================
# Main experiment
# =============================================================================

def run_model(model_id):
    """Run improved metrics for one model."""
    from transformers import AutoTokenizer, AutoProcessor

    model_short = model_id.split("/")[-1]
    expected = MODEL_REGISTRY[model_id]
    is_text_only = (model_id == "google/gemma-3-1b-it")

    print(f"\n{'#' * 70}")
    print(f"# MODEL: {model_id} (improvements)")
    print(f"{'#' * 70}\n")

    # Check if improvements already done
    probing_v2_path = RESULTS_DIR / f"{model_short}_probing_v2.json"
    kl_path = RESULTS_DIR / f"{model_short}_kl_divergence.json"
    if probing_v2_path.exists() and kl_path.exists():
        print(f"  [SKIP] {model_short} already has v2 results. Delete to re-run.")
        return

    # --- Load model ---
    print(f"[{model_short}] Loading model in BF16...")
    load_start = datetime.now()

    if is_text_only:
        from transformers import Gemma3ForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = Gemma3ForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto",
        )
    else:
        from transformers import Gemma3ForConditionalGeneration
        tokenizer = AutoProcessor.from_pretrained(model_id)
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto",
        )
    model.eval()

    num_layers = expected["num_layers"]
    load_time = (datetime.now() - load_start).total_seconds()
    vram_used = torch.cuda.max_memory_allocated() / 1e9
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[{model_short}] Loaded in {load_time:.1f}s | VRAM: {vram_used:.1f}/{vram_total:.1f} GB")

    # Get the raw tokenizer for text encoding
    tok = tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else tokenizer

    # =================================================================
    # 1. PROBING WITH PERMUTATION TEST
    # =================================================================
    if not probing_v2_path.exists():
        print(f"\n[{model_short}] === PROBING WITH PERMUTATION TEST ===")
        extract_start = datetime.now()

        all_hidden = {layer: [] for layer in range(num_layers + 1)}
        for text in tqdm(all_prompts, desc=f"[{model_short}] Hidden states"):
            inputs = tok(text, return_tensors="pt", truncation=True, max_length=128).to("cuda")
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            for layer_idx, hs in enumerate(outputs.hidden_states):
                mean_repr = hs.squeeze(0).mean(dim=0).float().cpu().numpy()
                all_hidden[layer_idx].append(mean_repr)
        for layer_idx in all_hidden:
            all_hidden[layer_idx] = np.array(all_hidden[layer_idx])

        extract_time = (datetime.now() - extract_start).total_seconds()
        print(f"[{model_short}] Extracted hidden states in {extract_time:.1f}s")

        probing_results = run_probing_with_controls(
            all_hidden, gender_labels, num_layers, model_short
        )
        probing_results["model_id"] = model_id
        probing_results["model_short"] = model_short
        probing_results["num_layers"] = num_layers
        probing_results["timestamp"] = datetime.now().isoformat()
        probing_results["extract_time_s"] = extract_time

        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Left: Real vs null accuracy per layer
        n_layers = len(probing_results["layer_accuracies"])
        layer_names = ["Emb"] + [f"L{i}" for i in range(1, n_layers)]
        x = range(n_layers)
        axes[0].bar(x, probing_results["layer_accuracies"], color="#E91E63",
                     edgecolor="white", linewidth=0.5, label="Real labels")
        axes[0].axhline(y=probing_results["null_mean_best"], color="blue",
                        linestyle="--", linewidth=2,
                        label=f"Null best (mean): {probing_results['null_mean_best']:.1%}")
        axes[0].axhline(y=probing_results["null_max_best"], color="orange",
                        linestyle=":", linewidth=2,
                        label=f"Null best (max): {probing_results['null_max_best']:.1%}")
        axes[0].axhline(y=0.5, color="gray", linestyle="--", alpha=0.3, label="Chance (50%)")
        axes[0].set_xlabel("Layer")
        axes[0].set_ylabel("Probe Accuracy (5-fold CV)")
        axes[0].set_title(f"Probing with Permutation Control — {model_short}")
        step = max(1, n_layers // 20)
        axes[0].set_xticks(range(0, n_layers, step))
        axes[0].set_xticklabels([layer_names[i] for i in range(0, n_layers, step)],
                                 rotation=45, ha="right", fontsize=9)
        axes[0].set_ylim(0.3, 1.05)
        axes[0].legend(fontsize=9)
        axes[0].grid(axis="y", alpha=0.3)

        # Right: Null distribution histogram
        axes[1].hist(probing_results["null_best_accuracies"], bins=20,
                     color="#607D8B", edgecolor="white", alpha=0.8, label="Null distribution")
        axes[1].axvline(x=probing_results["real_best_accuracy"], color="#E91E63",
                        linewidth=3, label=f"Real: {probing_results['real_best_accuracy']:.1%}")
        axes[1].set_xlabel("Best Layer Accuracy (across all layers)")
        axes[1].set_ylabel("Count")
        axes[1].set_title(f"Permutation Test (n={N_PERMUTATIONS}) — p={probing_results['p_value']:.4f}")
        axes[1].legend()
        axes[1].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"{model_short}_probing_v2.png", dpi=150, bbox_inches="tight")
        plt.close()

        with open(probing_v2_path, "w") as f:
            json.dump(probing_results, f, indent=2)
        print(f"[{model_short}] Probing v2 saved.")

        del all_hidden
        gc.collect()

    # =================================================================
    # 2. KL DIVERGENCE ON FIRST-TOKEN LOGITS
    # =================================================================
    if not kl_path.exists():
        print(f"\n[{model_short}] === KL DIVERGENCE ON LOGITS ===")

        kl_results = []
        for i, question in enumerate(tqdm(questions, desc=f"[{model_short}] KL divergence")):
            male_text = f"Hi, my name is {male_names[i]}. {question}"
            female_text = f"Hi, my name is {female_names[i]}. {question}"

            # Tokenize both prompts
            inputs_m = tok(male_text, return_tensors="pt", truncation=True, max_length=128).to("cuda")
            inputs_f = tok(female_text, return_tensors="pt", truncation=True, max_length=128).to("cuda")

            with torch.no_grad():
                logits_m = model(**inputs_m).logits[0, -1, :]  # last position logits
                logits_f = model(**inputs_f).logits[0, -1, :]

            kl_val = symmetric_kl(logits_m, logits_f)

            # Also get top-5 predicted tokens for each to show qualitative differences
            top_m = torch.topk(logits_m.float(), 5)
            top_f = torch.topk(logits_f.float(), 5)
            top_m_tokens = [tok.decode([t]) for t in top_m.indices.tolist()]
            top_f_tokens = [tok.decode([t]) for t in top_f.indices.tolist()]

            kl_results.append({
                "question": question,
                "male_name": male_names[i],
                "female_name": female_names[i],
                "symmetric_kl": kl_val,
                "top5_male": top_m_tokens,
                "top5_female": top_f_tokens,
            })

        kl_values = [r["symmetric_kl"] for r in kl_results]
        mean_kl = np.mean(kl_values)
        median_kl = np.median(kl_values)
        max_kl = np.max(kl_values)

        print(f"[{model_short}] Mean symmetric KL: {mean_kl:.4f}")
        print(f"[{model_short}] Median KL: {median_kl:.4f}")
        print(f"[{model_short}] Max KL: {max_kl:.4f}")

        # Control: KL between two male names (same gender, different names)
        control_kl_values = []
        for i in range(min(len(questions), len(male_names) - 1)):
            name_a = male_names[i]
            name_b = male_names[(i + 1) % len(male_names)]
            text_a = f"Hi, my name is {name_a}. {questions[i]}"
            text_b = f"Hi, my name is {name_b}. {questions[i]}"

            inputs_a = tok(text_a, return_tensors="pt", truncation=True, max_length=128).to("cuda")
            inputs_b = tok(text_b, return_tensors="pt", truncation=True, max_length=128).to("cuda")

            with torch.no_grad():
                logits_a = model(**inputs_a).logits[0, -1, :]
                logits_b = model(**inputs_b).logits[0, -1, :]

            control_kl_values.append(symmetric_kl(logits_a, logits_b))

        control_mean = np.mean(control_kl_values)
        control_median = np.median(control_kl_values)
        print(f"[{model_short}] Control KL (same-gender): mean={control_mean:.4f}, median={control_median:.4f}")

        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: KL per question
        sorted_idx = np.argsort(kl_values)[::-1]
        bars = axes[0].bar(range(len(kl_values)),
                           [kl_values[i] for i in sorted_idx],
                           color="#3F51B5", edgecolor="white", linewidth=0.5)
        axes[0].axhline(y=control_mean, color="green", linestyle="--", linewidth=2,
                        label=f"Same-gender control: {control_mean:.4f}")
        axes[0].axhline(y=mean_kl, color="red", linestyle="--", linewidth=2,
                        label=f"Cross-gender mean: {mean_kl:.4f}")
        axes[0].set_xlabel("Question (sorted by KL)")
        axes[0].set_ylabel("Symmetric KL Divergence")
        axes[0].set_title(f"First-Token KL Divergence — {model_short}")
        axes[0].legend(fontsize=9)
        axes[0].grid(axis="y", alpha=0.3)

        # Right: Distribution comparison
        axes[1].hist(kl_values, bins=12, color="#3F51B5", alpha=0.7,
                     edgecolor="white", label="Cross-gender KL")
        axes[1].hist(control_kl_values, bins=12, color="#4CAF50", alpha=0.7,
                     edgecolor="white", label="Same-gender KL (control)")
        axes[1].set_xlabel("Symmetric KL Divergence")
        axes[1].set_ylabel("Count")
        axes[1].set_title(f"KL Distribution: Cross- vs Same-Gender — {model_short}")
        axes[1].legend()
        axes[1].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"{model_short}_kl_divergence.png", dpi=150, bbox_inches="tight")
        plt.close()

        kl_data = {
            "model_id": model_id,
            "model_short": model_short,
            "timestamp": datetime.now().isoformat(),
            "mean_symmetric_kl": float(mean_kl),
            "median_symmetric_kl": float(median_kl),
            "max_symmetric_kl": float(max_kl),
            "control_mean_kl": float(control_mean),
            "control_median_kl": float(control_median),
            "kl_ratio": float(mean_kl / control_mean) if control_mean > 0 else 0,
            "per_question": kl_results,
            "control_kl_values": [float(v) for v in control_kl_values],
        }
        with open(kl_path, "w") as f:
            json.dump(kl_data, f, indent=2)
        print(f"[{model_short}] KL divergence saved.")

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[{model_short}] GPU memory freed.\n")


def generate_comparison():
    """Cross-model comparison for improved metrics."""
    print(f"\n{'#' * 70}")
    print(f"# CROSS-MODEL COMPARISON (V2)")
    print(f"{'#' * 70}\n")

    model_order = ["gemma-3-1b-it", "gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it"]
    colors = {"gemma-3-1b-it": "#4CAF50", "gemma-3-4b-it": "#2196F3",
              "gemma-3-12b-it": "#FF9800", "gemma-3-27b-it": "#E91E63"}

    # Load probing v2 results
    probing_data = {}
    kl_data = {}
    for ms in model_order:
        p = RESULTS_DIR / f"{ms}_probing_v2.json"
        if p.exists():
            with open(p) as f:
                probing_data[ms] = json.load(f)
        k = RESULTS_DIR / f"{ms}_kl_divergence.json"
        if k.exists():
            with open(k) as f:
                kl_data[ms] = json.load(f)

    if len(probing_data) < 2:
        print("Need at least 2 models for comparison. Skipping.")
        return

    # Figure 1: Probing with null baselines across models
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Probing accuracy overlay (normalized layer position)
    for ms in model_order:
        if ms in probing_data:
            accs = probing_data[ms]["layer_accuracies"]
            n = len(accs)
            x_norm = [i / (n - 1) for i in range(n)]
            axes[0].plot(x_norm, accs, marker="o", markersize=3,
                         color=colors.get(ms, "gray"), label=ms, linewidth=1.5)
    # Show null range
    for ms in model_order:
        if ms in probing_data:
            null_mean = probing_data[ms]["null_mean_best"]
            axes[0].axhline(y=null_mean, color=colors.get(ms, "gray"),
                            linestyle=":", alpha=0.4, linewidth=1)
    axes[0].axhline(0.5, color="red", linestyle="--", alpha=0.4, label="Chance")
    axes[0].set_xlabel("Relative Layer Position")
    axes[0].set_ylabel("Probe Accuracy (5-fold CV)")
    axes[0].set_title("Gender Probing with Permutation Baselines")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0.3, 1.05)
    axes[0].grid(alpha=0.3)

    # Right: Real vs null best-layer accuracy
    avail = [ms for ms in model_order if ms in probing_data]
    x = np.arange(len(avail))
    w = 0.35
    real_vals = [probing_data[ms]["real_best_accuracy"] for ms in avail]
    null_vals = [probing_data[ms]["null_mean_best"] for ms in avail]
    null_errs = [probing_data[ms]["null_std_best"] for ms in avail]
    axes[1].bar(x - w/2, real_vals, w, color="#E91E63", label="Real labels")
    axes[1].bar(x + w/2, null_vals, w, yerr=null_errs, color="#607D8B",
                label="Null (shuffled)", capsize=4)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(avail, fontsize=9)
    axes[1].set_ylabel("Best Layer Accuracy")
    axes[1].set_title("Real vs Null Best-Layer Accuracy")
    axes[1].legend()
    axes[1].set_ylim(0.3, 1.1)
    for i, v in enumerate(real_vals):
        p = probing_data[avail[i]]["p_value"]
        axes[1].text(i - w/2, v + 0.02, f"{v:.0%}\np={p:.3f}",
                     ha="center", fontsize=9, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "comparison_probing_v2.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved comparison_probing_v2.png")

    # Figure 2: KL divergence across models
    if len(kl_data) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: Mean KL with control baseline
        avail_kl = [ms for ms in model_order if ms in kl_data]
        x = np.arange(len(avail_kl))
        w = 0.35
        cross_kl = [kl_data[ms]["mean_symmetric_kl"] for ms in avail_kl]
        control_kl = [kl_data[ms]["control_mean_kl"] for ms in avail_kl]
        axes[0].bar(x - w/2, cross_kl, w, color="#3F51B5", label="Cross-gender")
        axes[0].bar(x + w/2, control_kl, w, color="#4CAF50", label="Same-gender (control)")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(avail_kl, fontsize=9)
        axes[0].set_ylabel("Mean Symmetric KL Divergence")
        axes[0].set_title("First-Token KL: Cross- vs Same-Gender")
        axes[0].legend()
        for i in range(len(avail_kl)):
            ratio = kl_data[avail_kl[i]]["kl_ratio"]
            axes[0].text(i, max(cross_kl[i], control_kl[i]) * 1.05,
                         f"{ratio:.1f}x", ha="center", fontsize=10, fontweight="bold")
        axes[0].grid(axis="y", alpha=0.3)

        # Right: KL ratio (cross/control) scaling
        ratios = [kl_data[ms]["kl_ratio"] for ms in avail_kl]
        bar_colors = [colors.get(ms, "gray") for ms in avail_kl]
        axes[1].bar(avail_kl, ratios, color=bar_colors, edgecolor="white")
        axes[1].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5,
                        label="No gender effect (ratio=1)")
        axes[1].set_ylabel("KL Ratio (cross-gender / same-gender)")
        axes[1].set_title("Gender Effect Scaling by Model Size")
        axes[1].legend()
        for i, v in enumerate(ratios):
            axes[1].text(i, v + 0.05, f"{v:.2f}x", ha="center", fontsize=11, fontweight="bold")
        axes[1].grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "comparison_kl_divergence.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved comparison_kl_divergence.png")

    # Save combined v2 summary
    summary_rows = []
    for ms in model_order:
        row = {"model": ms}
        if ms in probing_data:
            row["probing_best"] = probing_data[ms]["real_best_accuracy"]
            row["null_best_mean"] = probing_data[ms]["null_mean_best"]
            row["probing_p_value"] = probing_data[ms]["p_value"]
        if ms in kl_data:
            row["kl_cross_gender"] = kl_data[ms]["mean_symmetric_kl"]
            row["kl_same_gender"] = kl_data[ms]["control_mean_kl"]
            row["kl_ratio"] = kl_data[ms]["kl_ratio"]
        summary_rows.append(row)

    with open(RESULTS_DIR / "improvements_summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)
    print("Saved improvements_summary.json")

    print("\nComparison complete.")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    MODEL_SHORTCUTS = {
        "1b": "google/gemma-3-1b-it",
        "4b": "google/gemma-3-4b-it",
        "12b": "google/gemma-3-12b-it",
        "27b": "google/gemma-3-27b-it",
    }

    parser = argparse.ArgumentParser(description="Run improved Gemma 3 metrics")
    parser.add_argument("--models", nargs="+", choices=list(MODEL_SHORTCUTS.keys()),
                        default=None, help="Which models to run (e.g. --models 1b 4b)")
    args = parser.parse_args()

    if args.models:
        models_to_run = [MODEL_SHORTCUTS[m] for m in args.models]
    else:
        models_to_run = MODELS

    print("=" * 70)
    print("  Gemma 3 — Improved Metrics (Probing Controls + KL Divergence)")
    print("=" * 70)
    print(f"\nModels: {[m.split('/')[-1] for m in models_to_run]}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU'}")
    print(f"Start: {datetime.now().isoformat()}\n")

    from huggingface_hub import login
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("Authenticated with HuggingFace.\n")
    else:
        print("ERROR: Set HF_TOKEN first.")
        exit(1)

    total_start = datetime.now()
    for model_id in models_to_run:
        try:
            run_model(model_id)
        except Exception as e:
            print(f"\n[ERROR] {model_id}: {e}")
            import traceback
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()

    generate_comparison()

    total_time = (datetime.now() - total_start).total_seconds()
    print(f"\n{'=' * 70}")
    print(f"  ALL DONE — {total_time/60:.1f} minutes")
    print(f"{'=' * 70}")
