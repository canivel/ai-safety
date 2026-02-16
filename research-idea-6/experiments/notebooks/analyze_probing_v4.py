#!/usr/bin/env python3
"""
CPU-only: Analyze hidden states extracted from GPU.
Loads .npz files and runs all probing, permutation tests, and analysis.

Usage:
    python analyze_probing_v4.py qwen7b
    python analyze_probing_v4.py mistral7b
    python analyze_probing_v4.py gemma4b
    python analyze_probing_v4.py all        # run all available models
"""

import sys
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# === MODEL REGISTRY ===
MODEL_REGISTRY = {
    "qwen7b": {"model_short": "qwen2.5-7b-instruct", "model_id": "Qwen/Qwen2.5-7B-Instruct"},
    "mistral7b": {"model_short": "mistral-7b-instruct-v0.3", "model_id": "mistralai/Mistral-7B-Instruct-v0.3"},
    "gemma4b": {"model_short": "gemma-3-4b-it", "model_id": "google/gemma-3-4b-it"},
    "gemma1b": {"model_short": "gemma-3-1b-it", "model_id": "google/gemma-3-1b-it"},
    "gemma12b": {"model_short": "gemma-3-12b-it", "model_id": "google/gemma-3-12b-it"},
}

RESULTS_DIR = Path("../results/cross_family_probing_v4")

PROBE_CV_FOLDS = 5
PROBE_C = 1.0
PROBE_MAX_ITER = 1000
N_PERMUTATIONS = 100

# Dataset sizes (must match extraction script)
N_QUESTIONS = 200
N_MALE = 45
N_FEMALE = 45
N_AMBIGUOUS = 25

# Ambiguous names for labeling
ambiguous_names = [
    "Alex", "Jordan", "Taylor", "Morgan", "Casey",
    "Riley", "Avery", "Quinn", "Dakota", "Sage",
    "Blair", "Rowan", "Reese", "Cameron", "Hayden",
    "Skyler", "Finley", "Emery", "Peyton", "Addison",
    "Drew", "Jamie", "Jesse", "Pat", "Robin",
]


def run_probe(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    probe = LogisticRegression(max_iter=PROBE_MAX_ITER, solver="lbfgs", C=PROBE_C)
    scores = cross_val_score(probe, X_scaled, y, cv=PROBE_CV_FOLDS, scoring="accuracy")
    return float(scores.mean()), float(scores.std())


def run_permutation_test(X, y, n_perm=N_PERMUTATIONS):
    rng = np.random.RandomState(42)
    null_accs = []
    for _ in tqdm(range(n_perm), desc="Permutation test"):
        shuffled = rng.permutation(y)
        acc, _ = run_probe(X, shuffled)
        null_accs.append(acc)
    return null_accs


def analyze_model(model_key):
    cfg = MODEL_REGISTRY[model_key]
    model_short = cfg["model_short"]
    model_id = cfg["model_id"]

    hs_path = RESULTS_DIR / f"{model_short}_hidden_states.npz"
    steer_path = RESULTS_DIR / f"{model_short}_steering.json"

    if not hs_path.exists():
        print(f"  SKIP: {hs_path} not found")
        return None

    print(f"\n{'=' * 70}")
    print(f"  Analyzing: {model_short}")
    print(f"{'=' * 70}")

    t0 = datetime.now()

    # Load hidden states
    print("\n  Loading hidden states...")
    data = np.load(hs_path)
    total_layers = int(data["total_layers"][0])
    hidden_dim = int(data["hidden_dim"][0])
    gender_labels = data["gender_labels"]

    hs_last_token = {}
    hs_mean_question = {}
    hs_ambiguous = {}
    for layer in range(total_layers):
        hs_last_token[layer] = data[f"last_token_layer_{layer}"]
        hs_mean_question[layer] = data[f"question_only_layer_{layer}"]
        hs_ambiguous[layer] = data[f"ambiguous_layer_{layer}"]

    n_samples = hs_last_token[0].shape[0]
    print(f"  Shape: ({n_samples}, {hidden_dim}) | {total_layers} layers")

    # Load steering data
    steering_results = {}
    if steer_path.exists():
        with open(steer_path) as f:
            steering_results = json.load(f)
        print(f"  Steering data loaded (layer {steering_results.get('steer_layer', '?')})")

    # =========================================================
    # VARIANT A: Last-token probing
    # =========================================================
    print(f"\n[1/5] Variant A: Last-token probing ({total_layers} layers)...")
    variant_a = {"variant": "last_token", "layer_accuracies": [], "layer_stds": []}
    for layer in tqdm(range(total_layers), desc="Last-token probe"):
        acc, std = run_probe(hs_last_token[layer], gender_labels)
        variant_a["layer_accuracies"].append(acc)
        variant_a["layer_stds"].append(std)

    best_layer_a = int(np.argmax(variant_a["layer_accuracies"]))
    best_acc_a = variant_a["layer_accuracies"][best_layer_a]
    print(f"  Best: Layer {best_layer_a} = {best_acc_a:.1%}")

    print("  Running permutation test on best layer...")
    null_a = run_permutation_test(hs_last_token[best_layer_a], gender_labels)
    variant_a["best_layer"] = best_layer_a
    variant_a["best_accuracy"] = best_acc_a
    variant_a["null_mean"] = float(np.mean(null_a))
    variant_a["null_std"] = float(np.std(null_a))
    variant_a["null_max"] = float(np.max(null_a))
    variant_a["p_value"] = float(np.mean([n >= best_acc_a for n in null_a]))
    variant_a["embedding_accuracy"] = variant_a["layer_accuracies"][0]
    print(f"  Embedding: {variant_a['embedding_accuracy']:.1%}")
    print(f"  Null: {variant_a['null_mean']:.1%} +/- {variant_a['null_std']:.1%}")
    print(f"  p-value: {variant_a['p_value']:.4f}")

    # =========================================================
    # VARIANT B: Question-tokens only probing
    # =========================================================
    print(f"\n[2/5] Variant B: Question-only probing ({total_layers} layers)...")
    variant_b = {"variant": "question_tokens_only", "layer_accuracies": [], "layer_stds": []}
    for layer in tqdm(range(total_layers), desc="Question-only probe"):
        acc, std = run_probe(hs_mean_question[layer], gender_labels)
        variant_b["layer_accuracies"].append(acc)
        variant_b["layer_stds"].append(std)

    best_layer_b = int(np.argmax(variant_b["layer_accuracies"]))
    best_acc_b = variant_b["layer_accuracies"][best_layer_b]
    variant_b["best_layer"] = best_layer_b
    variant_b["best_accuracy"] = best_acc_b
    print("  Running permutation test on best layer...")
    null_b = run_permutation_test(hs_mean_question[best_layer_b], gender_labels)
    variant_b["null_mean"] = float(np.mean(null_b))
    variant_b["null_std"] = float(np.std(null_b))
    variant_b["null_max"] = float(np.max(null_b))
    variant_b["p_value"] = float(np.mean([n >= best_acc_b for n in null_b]))
    variant_b["embedding_accuracy"] = variant_b["layer_accuracies"][0]
    print(f"  Best: Layer {best_layer_b} = {best_acc_b:.1%}")
    print(f"  Embedding: {variant_b['embedding_accuracy']:.1%}")
    print(f"  p-value: {variant_b['p_value']:.4f}")

    # =========================================================
    # VARIANT C: Held-out name generalization
    # =========================================================
    print(f"\n[3/5] Variant C: Held-out name generalization...")
    train_name_idx = set(range(0, 35))
    test_name_idx = set(range(35, N_MALE))

    train_indices = []
    test_indices = []
    for i in range(n_samples):
        q_idx = i if i < N_QUESTIONS else i - N_QUESTIONS
        name_idx = q_idx % N_MALE
        if name_idx in train_name_idx:
            train_indices.append(i)
        elif name_idx in test_name_idx:
            test_indices.append(i)

    y_train = gender_labels[train_indices]
    y_test = gender_labels[test_indices]

    variant_c = {
        "variant": "held_out_names",
        "n_train": len(train_indices),
        "n_test": len(test_indices),
    }

    for hs_name, hs_dict in [("last_token", hs_last_token), ("question_only", hs_mean_question)]:
        train_accs = []
        test_accs = []
        for layer in range(total_layers):
            X_train = hs_dict[layer][train_indices]
            X_test = hs_dict[layer][test_indices]
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            probe = LogisticRegression(max_iter=PROBE_MAX_ITER, solver="lbfgs", C=PROBE_C)
            probe.fit(X_train_s, y_train)
            train_accs.append(float(probe.score(X_train_s, y_train)))
            test_accs.append(float(probe.score(X_test_s, y_test)))

        best_test_layer = int(np.argmax(test_accs))
        variant_c[f"{hs_name}_train_accs"] = train_accs
        variant_c[f"{hs_name}_test_accs"] = test_accs
        variant_c[f"{hs_name}_best_test_layer"] = best_test_layer
        variant_c[f"{hs_name}_best_test_acc"] = test_accs[best_test_layer]
        variant_c[f"{hs_name}_best_train_acc"] = train_accs[best_test_layer]
        print(f"  Held-out ({hs_name}): train={train_accs[best_test_layer]:.1%}, test={test_accs[best_test_layer]:.1%} at layer {best_test_layer}")

    # =========================================================
    # VARIANT D: Steering (loaded from GPU results)
    # =========================================================
    print(f"\n[4/5] Variant D: Steering (from GPU extraction)...")
    # Clean up steering_results to remove per-question arrays for the final JSON
    variant_d = {}
    if steering_results:
        variant_d["layer"] = steering_results.get("steer_layer", best_layer_a)
        for strength in [0.0, 1.0, 2.0, 5.0, 10.0]:
            key = f"strength_{strength}"
            if key in steering_results:
                d = steering_results[key]
                variant_d[key] = {
                    "cross_gender_kl": d["cross_gender_kl"],
                    "same_gender_kl": d["same_gender_kl"],
                    "ratio": d["ratio"],
                }
                print(f"  alpha={strength:.1f}: cross={d['cross_gender_kl']:.6f}, same={d['same_gender_kl']:.6f}, ratio={d['ratio']:.2f}x")
    else:
        print("  WARNING: No steering data found")

    # =========================================================
    # AMBIGUOUS NAME CONTROL
    # =========================================================
    print(f"\n[5/5] Ambiguous name control ({N_AMBIGUOUS} names)...")

    # Train probe on best layer using all gendered data
    X_best = hs_last_token[best_layer_a]
    scaler_best = StandardScaler()
    X_best_s = scaler_best.fit_transform(X_best)
    probe_best = LogisticRegression(max_iter=PROBE_MAX_ITER, solver="lbfgs", C=PROBE_C)
    probe_best.fit(X_best_s, gender_labels)

    ambiguous_control = {"layer": best_layer_a, "per_name": []}
    amb_predictions = []
    amb_confidences = []

    for i in range(N_AMBIGUOUS):
        h_last = hs_ambiguous[best_layer_a][i].reshape(1, -1)
        h_last_s = scaler_best.transform(h_last)

        pred = int(probe_best.predict(h_last_s)[0])
        prob = float(probe_best.predict_proba(h_last_s)[0, 1])

        amb_predictions.append(pred)
        amb_confidences.append(prob)

        name = ambiguous_names[i % N_AMBIGUOUS]
        ambiguous_control["per_name"].append({
            "name": name,
            "predicted_gender": "female" if pred == 1 else "male",
            "p_female": round(prob, 4),
        })

    n_pred_male = sum(1 for p in amb_predictions if p == 0)
    n_pred_female = sum(1 for p in amb_predictions if p == 1)
    mean_confidence = float(np.mean([abs(c - 0.5) for c in amb_confidences]))

    ambiguous_control["n_predicted_male"] = n_pred_male
    ambiguous_control["n_predicted_female"] = n_pred_female
    ambiguous_control["mean_deviation_from_chance"] = round(mean_confidence, 4)
    print(f"  Predictions: {n_pred_male} male, {n_pred_female} female")
    print(f"  Mean |P(female) - 0.5|: {mean_confidence:.4f}")

    # Add KL info from steering
    if "amb_kl_mean" in steering_results:
        ambiguous_control["mean_kl_between_ambiguous"] = steering_results["amb_kl_mean"]
        baseline_cross = steering_results.get("strength_0.0", {}).get("cross_gender_kl", 0)
        baseline_same = steering_results.get("strength_0.0", {}).get("same_gender_kl", 0)
        print(f"  KL between ambiguous pairs: {steering_results['amb_kl_mean']:.6f}")
        print(f"  (vs cross-gender: {baseline_cross:.6f}, same-gender: {baseline_same:.6f})")

    # =========================================================
    # SAVE FINAL RESULTS
    # =========================================================
    results = {
        "model": model_id,
        "model_short": model_short,
        "model_key": model_key,
        "timestamp": datetime.now().isoformat(),
        "dataset": {
            "n_questions": N_QUESTIONS,
            "n_male_names": N_MALE,
            "n_female_names": N_FEMALE,
            "n_ambiguous_names": N_AMBIGUOUS,
            "n_gendered_prompts": n_samples,
        },
        "total_layers": total_layers,
        "hidden_dim": hidden_dim,
        "variant_a_last_token": variant_a,
        "variant_b_question_only": variant_b,
        "variant_c_held_out": variant_c,
        "variant_d_steering": variant_d,
        "ambiguous_control": ambiguous_control,
    }

    out_path = RESULTS_DIR / f"{model_short}_probing_v4.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    elapsed = (datetime.now() - t0).total_seconds() / 60

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY — {model_short}")
    print(f"{'=' * 70}")
    print(f"\n  Variant A (last-token): {best_acc_a:.1%} at L{best_layer_a} (emb={variant_a['embedding_accuracy']:.1%}, p={variant_a['p_value']:.4f})")
    print(f"  Variant B (question-only): {best_acc_b:.1%} at L{best_layer_b} (emb={variant_b['embedding_accuracy']:.1%}, p={variant_b['p_value']:.4f})")
    for hs_name in ["last_token", "question_only"]:
        acc = variant_c[f"{hs_name}_best_test_acc"]
        layer = variant_c[f"{hs_name}_best_test_layer"]
        print(f"  Variant C ({hs_name}): test={acc:.1%} at L{layer}")
    if variant_d:
        for s in [0.0, 1.0, 2.0]:
            key = f"strength_{s}"
            if key in variant_d:
                d = variant_d[key]
                print(f"  Variant D (alpha={s:.0f}): ratio={d['ratio']:.2f}x")
    print(f"  Ambiguous: {n_pred_male}M/{n_pred_female}F, dev={mean_confidence:.4f}")
    print(f"\n  Time: {elapsed:.1f} min")
    print(f"  Saved: {out_path}")
    print(f"{'=' * 70}")

    return results


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python analyze_probing_v4.py <model_key|all>")
        print(f"  Available: {list(MODEL_REGISTRY.keys())}")
        sys.exit(1)

    target = sys.argv[1]

    if target == "all":
        keys = list(MODEL_REGISTRY.keys())
    elif target in MODEL_REGISTRY:
        keys = [target]
    else:
        print(f"ERROR: Unknown model '{target}'. Choose from: {list(MODEL_REGISTRY.keys())} or 'all'")
        sys.exit(1)

    all_results = {}
    for key in keys:
        result = analyze_model(key)
        if result:
            all_results[key] = result

    if len(all_results) > 1:
        print(f"\n\n{'=' * 70}")
        print(f"  CROSS-MODEL COMPARISON")
        print(f"{'=' * 70}")
        print(f"\n  {'Model':<25} {'Last-tok':<12} {'Q-only':<12} {'Held-out':<12} {'Steer 0->2':<15}")
        print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12} {'-'*15}")
        for key, r in all_results.items():
            lt = f"{r['variant_a_last_token']['best_accuracy']:.1%}"
            qo = f"{r['variant_b_question_only']['best_accuracy']:.1%}"
            ho = f"{r['variant_c_held_out'].get('last_token_best_test_acc', 0):.1%}"
            s0 = r['variant_d_steering'].get('strength_0.0', {}).get('ratio', 0)
            s2 = r['variant_d_steering'].get('strength_2.0', {}).get('ratio', 0)
            steer = f"{s0:.1f}x→{s2:.1f}x" if s0 > 0 else "N/A"
            print(f"  {r['model_short']:<25} {lt:<12} {qo:<12} {ho:<12} {steer:<15}")


if __name__ == "__main__":
    main()
