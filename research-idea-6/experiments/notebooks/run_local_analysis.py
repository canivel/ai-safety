#!/usr/bin/env python3
"""
Local CPU-only analysis: permutation tests + visualization.
Run AFTER run_gpu_only.py has saved hidden states and KL results.

Usage:
    python run_local_analysis.py
"""

import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

N_PERMUTATIONS = 100
PROBE_CV_FOLDS = 5
PROBE_C = 1.0
PROBE_MAX_ITER = 1000

RESULTS_DIR = Path("../results/gemma3_gender_detection")

MODEL_ORDER = ["gemma-3-1b-it", "gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it"]
COLORS = {"gemma-3-1b-it": "#4CAF50", "gemma-3-4b-it": "#2196F3",
           "gemma-3-12b-it": "#FF9800", "gemma-3-27b-it": "#E91E63"}

gender_labels = np.array([0] * 25 + [1] * 25)


def run_probing_with_permutation(model_short):
    """Run probing + permutation test from saved hidden states."""
    hs_path = RESULTS_DIR / f"{model_short}_hidden_states.npz"
    out_path = RESULTS_DIR / f"{model_short}_probing_v2.json"

    if out_path.exists():
        print(f"  [SKIP] {model_short} probing_v2 already exists.")
        return

    if not hs_path.exists():
        print(f"  [MISSING] {model_short} hidden states not found. Run run_gpu_only.py first.")
        return

    print(f"\n[{model_short}] Loading hidden states...")
    data = np.load(hs_path)
    num_layers = len(data.files)
    hidden_states = {i: data[f"layer_{i}"] for i in range(num_layers)}
    print(f"[{model_short}] {num_layers} layers, shape={hidden_states[0].shape}")

    # Real probing
    print(f"[{model_short}] Running probing...")
    layer_accuracies = []
    layer_stds = []
    for layer_idx in range(num_layers):
        X = hidden_states[layer_idx]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        probe = LogisticRegression(max_iter=PROBE_MAX_ITER, solver="lbfgs", C=PROBE_C)
        scores = cross_val_score(probe, X_scaled, gender_labels, cv=PROBE_CV_FOLDS, scoring="accuracy")
        layer_accuracies.append(scores.mean())
        layer_stds.append(scores.std())

    real_best = max(layer_accuracies)
    print(f"[{model_short}] Real best: {real_best:.1%}")

    # Permutation test
    print(f"[{model_short}] Running {N_PERMUTATIONS} permutations...")
    rng = np.random.RandomState(42)
    null_best_accuracies = []
    for _ in tqdm(range(N_PERMUTATIONS), desc=f"[{model_short}] Permutation"):
        shuffled = rng.permutation(gender_labels)
        perm_accs = []
        for layer_idx in range(num_layers):
            X = hidden_states[layer_idx]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            probe = LogisticRegression(max_iter=PROBE_MAX_ITER, solver="lbfgs", C=PROBE_C)
            scores = cross_val_score(probe, X_scaled, shuffled, cv=PROBE_CV_FOLDS, scoring="accuracy")
            perm_accs.append(scores.mean())
        null_best_accuracies.append(max(perm_accs))

    null_mean = float(np.mean(null_best_accuracies))
    null_std = float(np.std(null_best_accuracies))
    null_max = float(np.max(null_best_accuracies))
    p_value = float(np.mean([n >= real_best for n in null_best_accuracies]))

    print(f"[{model_short}] Null best: {null_mean:.1%} +/- {null_std:.1%} (max={null_max:.1%})")
    print(f"[{model_short}] p-value: {p_value:.4f}")

    results = {
        "model_short": model_short,
        "num_layers": num_layers,
        "timestamp": datetime.now().isoformat(),
        "layer_accuracies": [float(a) for a in layer_accuracies],
        "layer_stds": [float(s) for s in layer_stds],
        "real_best_accuracy": float(real_best),
        "null_mean_best": null_mean,
        "null_std_best": null_std,
        "null_max_best": null_max,
        "null_best_accuracies": [float(a) for a in null_best_accuracies],
        "p_value": p_value,
        "n_permutations": N_PERMUTATIONS,
    }

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    n = len(layer_accuracies)
    layer_names = ["Emb"] + [f"L{i}" for i in range(1, n)]
    axes[0].bar(range(n), layer_accuracies, color="#E91E63", edgecolor="white", linewidth=0.5, label="Real labels")
    axes[0].axhline(y=null_mean, color="blue", linestyle="--", linewidth=2,
                    label=f"Null best mean: {null_mean:.1%}")
    axes[0].axhline(y=null_max, color="orange", linestyle=":", linewidth=2,
                    label=f"Null best max: {null_max:.1%}")
    axes[0].axhline(y=0.5, color="gray", linestyle="--", alpha=0.3, label="Chance")
    axes[0].set_xlabel("Layer"); axes[0].set_ylabel("Accuracy (5-fold CV)")
    axes[0].set_title(f"Probing with Permutation Control — {model_short}")
    step = max(1, n // 20)
    axes[0].set_xticks(range(0, n, step))
    axes[0].set_xticklabels([layer_names[i] for i in range(0, n, step)], rotation=45, ha="right", fontsize=9)
    axes[0].set_ylim(0.3, 1.05); axes[0].legend(fontsize=9); axes[0].grid(axis="y", alpha=0.3)

    axes[1].hist(null_best_accuracies, bins=20, color="#607D8B", edgecolor="white", alpha=0.8, label="Null distribution")
    axes[1].axvline(x=real_best, color="#E91E63", linewidth=3, label=f"Real: {real_best:.1%}")
    axes[1].set_xlabel("Best Layer Accuracy"); axes[1].set_ylabel("Count")
    axes[1].set_title(f"Permutation Test (n={N_PERMUTATIONS}) — p={p_value:.4f}")
    axes[1].legend(); axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{model_short}_probing_v2.png", dpi=150, bbox_inches="tight")
    plt.close()

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{model_short}] Saved: {out_path.name}")


def generate_kl_charts():
    """Generate KL divergence comparison charts from saved data."""
    kl_data = {}
    for ms in MODEL_ORDER:
        p = RESULTS_DIR / f"{ms}_kl_divergence.json"
        if p.exists():
            with open(p) as f:
                kl_data[ms] = json.load(f)

    if len(kl_data) < 2:
        print("Need >= 2 KL results for comparison.")
        return

    avail = [ms for ms in MODEL_ORDER if ms in kl_data]
    print(f"\nGenerating KL charts for {len(avail)} models...")

    # Per-model charts
    for ms in avail:
        d = kl_data[ms]
        kl_values = [r["symmetric_kl"] for r in d["per_question"]]
        control_values = d["control_kl_values"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        sorted_idx = np.argsort(kl_values)[::-1]
        axes[0].bar(range(len(kl_values)), [kl_values[i] for i in sorted_idx],
                    color="#3F51B5", edgecolor="white", linewidth=0.5)
        axes[0].axhline(y=d["control_mean_kl"], color="green", linestyle="--", linewidth=2,
                        label=f"Same-gender: {d['control_mean_kl']:.4f}")
        axes[0].axhline(y=d["mean_symmetric_kl"], color="red", linestyle="--", linewidth=2,
                        label=f"Cross-gender: {d['mean_symmetric_kl']:.4f}")
        axes[0].set_xlabel("Question (sorted)"); axes[0].set_ylabel("Symmetric KL")
        axes[0].set_title(f"First-Token KL — {ms}"); axes[0].legend(fontsize=9)
        axes[0].grid(axis="y", alpha=0.3)

        axes[1].hist(kl_values, bins=12, color="#3F51B5", alpha=0.7, edgecolor="white", label="Cross-gender")
        axes[1].hist(control_values, bins=12, color="#4CAF50", alpha=0.7, edgecolor="white", label="Same-gender")
        axes[1].set_xlabel("Symmetric KL"); axes[1].set_ylabel("Count")
        axes[1].set_title(f"KL Distribution — {ms}"); axes[1].legend()
        axes[1].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"{ms}_kl_divergence.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Cross-model comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(avail))
    w = 0.35
    cross_kl = [kl_data[ms]["mean_symmetric_kl"] for ms in avail]
    control_kl = [kl_data[ms]["control_mean_kl"] for ms in avail]
    axes[0].bar(x - w/2, cross_kl, w, color="#3F51B5", label="Cross-gender")
    axes[0].bar(x + w/2, control_kl, w, color="#4CAF50", label="Same-gender (control)")
    axes[0].set_xticks(x); axes[0].set_xticklabels(avail, fontsize=9)
    axes[0].set_ylabel("Mean Symmetric KL"); axes[0].set_title("First-Token KL: Cross vs Same Gender")
    axes[0].legend()
    for i in range(len(avail)):
        r = kl_data[avail[i]]["kl_ratio"]
        axes[0].text(i, max(cross_kl[i], control_kl[i]) * 1.05, f"{r:.1f}x",
                     ha="center", fontsize=10, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)

    ratios = [kl_data[ms]["kl_ratio"] for ms in avail]
    axes[1].bar(avail, ratios, color=[COLORS.get(ms, "gray") for ms in avail], edgecolor="white")
    axes[1].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="No effect (1x)")
    axes[1].set_ylabel("KL Ratio (cross / same gender)")
    axes[1].set_title("Gender Effect by Model Size")
    axes[1].legend()
    for i, v in enumerate(ratios):
        axes[1].text(i, v + 0.05, f"{v:.2f}x", ha="center", fontsize=11, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "comparison_kl_divergence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved comparison_kl_divergence.png")


def generate_probing_comparison():
    """Cross-model probing comparison with null baselines."""
    probing_data = {}
    for ms in MODEL_ORDER:
        p = RESULTS_DIR / f"{ms}_probing_v2.json"
        if p.exists():
            with open(p) as f:
                probing_data[ms] = json.load(f)

    if len(probing_data) < 2:
        print("Need >= 2 probing_v2 results for comparison.")
        return

    avail = [ms for ms in MODEL_ORDER if ms in probing_data]
    print(f"Generating probing comparison for {len(avail)} models...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: overlay
    for ms in avail:
        accs = probing_data[ms]["layer_accuracies"]
        n = len(accs)
        x_norm = [i / (n - 1) for i in range(n)]
        axes[0].plot(x_norm, accs, marker="o", markersize=3,
                     color=COLORS.get(ms, "gray"), label=ms, linewidth=1.5)
    for ms in avail:
        axes[0].axhline(y=probing_data[ms]["null_mean_best"], color=COLORS.get(ms, "gray"),
                        linestyle=":", alpha=0.4)
    axes[0].axhline(0.5, color="red", linestyle="--", alpha=0.4, label="Chance")
    axes[0].set_xlabel("Relative Layer Position"); axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Probing with Permutation Baselines")
    axes[0].legend(fontsize=8); axes[0].set_ylim(0.3, 1.05); axes[0].grid(alpha=0.3)

    # Right: real vs null bars
    x = np.arange(len(avail))
    w = 0.35
    real_vals = [probing_data[ms]["real_best_accuracy"] for ms in avail]
    null_vals = [probing_data[ms]["null_mean_best"] for ms in avail]
    null_errs = [probing_data[ms]["null_std_best"] for ms in avail]
    axes[1].bar(x - w/2, real_vals, w, color="#E91E63", label="Real labels")
    axes[1].bar(x + w/2, null_vals, w, yerr=null_errs, color="#607D8B", label="Null (shuffled)", capsize=4)
    axes[1].set_xticks(x); axes[1].set_xticklabels(avail, fontsize=9)
    axes[1].set_ylabel("Best Layer Accuracy"); axes[1].set_title("Real vs Null Accuracy")
    axes[1].legend(); axes[1].set_ylim(0.3, 1.1)
    for i, v in enumerate(real_vals):
        p = probing_data[avail[i]]["p_value"]
        axes[1].text(i - w/2, v + 0.02, f"{v:.0%}\np={p:.3f}", ha="center", fontsize=9, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "comparison_probing_v2.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved comparison_probing_v2.png")

    # Save summary
    summary = []
    for ms in MODEL_ORDER:
        row = {"model": ms}
        if ms in probing_data:
            row["probing_best"] = probing_data[ms]["real_best_accuracy"]
            row["null_best_mean"] = probing_data[ms]["null_mean_best"]
            row["probing_p_value"] = probing_data[ms]["p_value"]
        kl_path = RESULTS_DIR / f"{ms}_kl_divergence.json"
        if kl_path.exists():
            with open(kl_path) as f:
                kl = json.load(f)
            row["kl_cross_gender"] = kl["mean_symmetric_kl"]
            row["kl_same_gender"] = kl["control_mean_kl"]
            row["kl_ratio"] = kl["kl_ratio"]
        summary.append(row)

    with open(RESULTS_DIR / "improvements_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved improvements_summary.json")


if __name__ == "__main__":
    print("=" * 70)
    print("  Local Analysis: Permutation Tests + Charts")
    print("=" * 70)

    t0 = datetime.now()

    # Run permutation tests
    for ms in MODEL_ORDER:
        hs_path = RESULTS_DIR / f"{ms}_hidden_states.npz"
        if hs_path.exists():
            run_probing_with_permutation(ms)

    # Generate all charts
    generate_kl_charts()
    generate_probing_comparison()

    print(f"\nAll done in {(datetime.now()-t0).total_seconds()/60:.1f} min")
