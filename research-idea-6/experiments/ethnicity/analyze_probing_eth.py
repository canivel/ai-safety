#!/usr/bin/env python3
"""
CPU-only: Analyze hidden states extracted from GPU for ETHNICITY probing.
Loads .npz files and runs all probing, permutation tests, and analysis.

Usage:
    python analyze_probing_eth.py gemma4b white_vs_black
    python analyze_probing_eth.py all white_vs_black
    python analyze_probing_eth.py all all           # all models, all comparisons
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.load_data import (
    load_ethnicity_names,
    load_model_registry,
    get_ethnicity_comparison,
    VALID_COMPARISONS,
)

RESULTS_BASE = Path(__file__).resolve().parent.parent / "results" / "ethnicity_probing"

PROBE_CV_FOLDS = 5
PROBE_C = 1.0
PROBE_MAX_ITER = 1000
N_PERMUTATIONS = 100

N_QUESTIONS = 200
N_PER_GROUP = 45
N_AMBIGUOUS = 25

eth_names_data = load_ethnicity_names()
amb_names = eth_names_data["ambiguous"]["names"]


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


def analyze_model(model_key, comparison):
    registry = load_model_registry()
    cfg = registry[model_key]
    model_short = cfg["model_short"]
    model_id = cfg["model_id"]

    ref_group, cmp_group = get_ethnicity_comparison(comparison)
    results_dir = RESULTS_BASE / comparison

    hs_path = results_dir / f"{model_short}_hidden_states.npz"
    steer_path = results_dir / f"{model_short}_steering.json"

    if not hs_path.exists():
        print(f"  SKIP: {hs_path} not found")
        return None

    print(f"\n{'=' * 70}")
    print(f"  Analyzing: {model_short} / {comparison}")
    print(f"{'=' * 70}")

    t0 = datetime.now()

    # Load hidden states
    print("\n  Loading hidden states...")
    data = np.load(hs_path)
    total_layers = int(data["total_layers"][0])
    hidden_dim = int(data["hidden_dim"][0])
    eth_labels = data["eth_labels"]

    # Gender annotations for confound check
    prompt_genders = data.get("prompt_genders", None)

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
    print(f"\n[1/6] Variant A: Last-token probing ({total_layers} layers)...")
    variant_a = {"variant": "last_token", "layer_accuracies": [], "layer_stds": []}
    for layer in tqdm(range(total_layers), desc="Last-token probe"):
        acc, std = run_probe(hs_last_token[layer], eth_labels)
        variant_a["layer_accuracies"].append(acc)
        variant_a["layer_stds"].append(std)

    best_layer_a = int(np.argmax(variant_a["layer_accuracies"]))
    best_acc_a = variant_a["layer_accuracies"][best_layer_a]
    print(f"  Best: Layer {best_layer_a} = {best_acc_a:.1%}")

    print("  Running permutation test on best layer...")
    null_a = run_permutation_test(hs_last_token[best_layer_a], eth_labels)
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
    print(f"\n[2/6] Variant B: Question-only probing ({total_layers} layers)...")
    variant_b = {"variant": "question_tokens_only", "layer_accuracies": [], "layer_stds": []}
    for layer in tqdm(range(total_layers), desc="Question-only probe"):
        acc, std = run_probe(hs_mean_question[layer], eth_labels)
        variant_b["layer_accuracies"].append(acc)
        variant_b["layer_stds"].append(std)

    best_layer_b = int(np.argmax(variant_b["layer_accuracies"]))
    best_acc_b = variant_b["layer_accuracies"][best_layer_b]
    variant_b["best_layer"] = best_layer_b
    variant_b["best_accuracy"] = best_acc_b
    print("  Running permutation test on best layer...")
    null_b = run_permutation_test(hs_mean_question[best_layer_b], eth_labels)
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
    print(f"\n[3/6] Variant C: Held-out name generalization...")
    train_name_idx = set(range(0, 35))
    test_name_idx = set(range(35, N_PER_GROUP))

    train_indices = []
    test_indices = []
    for i in range(n_samples):
        q_idx = i if i < N_QUESTIONS else i - N_QUESTIONS
        name_idx = q_idx % N_PER_GROUP
        if name_idx in train_name_idx:
            train_indices.append(i)
        elif name_idx in test_name_idx:
            test_indices.append(i)

    y_train = eth_labels[train_indices]
    y_test = eth_labels[test_indices]

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
    print(f"\n[4/6] Variant D: Steering (from GPU extraction)...")
    variant_d = {}
    if steering_results:
        variant_d["layer"] = steering_results.get("steer_layer", best_layer_a)
        for strength in [0.0, 1.0, 2.0, 5.0, 10.0]:
            key = f"strength_{strength}"
            if key in steering_results:
                d = steering_results[key]
                variant_d[key] = {
                    "cross_ethnicity_kl": d["cross_ethnicity_kl"],
                    "same_ethnicity_kl": d["same_ethnicity_kl"],
                    "ratio": d["ratio"],
                }
                print(f"  alpha={strength:.1f}: cross={d['cross_ethnicity_kl']:.6f}, same={d['same_ethnicity_kl']:.6f}, ratio={d['ratio']:.2f}x")
    else:
        print("  WARNING: No steering data found")

    # =========================================================
    # AMBIGUOUS NAME CONTROL
    # =========================================================
    print(f"\n[5/6] Ambiguous name control ({N_AMBIGUOUS} names)...")

    X_best = hs_last_token[best_layer_a]
    scaler_best = StandardScaler()
    X_best_s = scaler_best.fit_transform(X_best)
    probe_best = LogisticRegression(max_iter=PROBE_MAX_ITER, solver="lbfgs", C=PROBE_C)
    probe_best.fit(X_best_s, eth_labels)

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

        name = amb_names[i % N_AMBIGUOUS]
        ambiguous_control["per_name"].append({
            "name": name,
            "predicted_group": cmp_group if pred == 1 else ref_group,
            "p_comparison": round(prob, 4),
        })

    n_pred_ref = sum(1 for p in amb_predictions if p == 0)
    n_pred_cmp = sum(1 for p in amb_predictions if p == 1)
    mean_confidence = float(np.mean([abs(c - 0.5) for c in amb_confidences]))

    ambiguous_control["n_predicted_ref"] = n_pred_ref
    ambiguous_control["n_predicted_cmp"] = n_pred_cmp
    ambiguous_control["mean_deviation_from_chance"] = round(mean_confidence, 4)
    print(f"  Predictions: {n_pred_ref} {ref_group}, {n_pred_cmp} {cmp_group}")
    print(f"  Mean |P(cmp) - 0.5|: {mean_confidence:.4f}")

    if "amb_kl_mean" in steering_results:
        ambiguous_control["mean_kl_between_ambiguous"] = steering_results["amb_kl_mean"]

    # =========================================================
    # GENDER CONFOUND CHECK
    # =========================================================
    print(f"\n[6/6] Gender confound check...")
    gender_confound = {}

    if prompt_genders is not None:
        # Run ethnicity probing on same-gender subsets only
        for g_label, g_name in [(0, "male"), (1, "female")]:
            g_mask = prompt_genders == g_label
            g_indices = np.where(g_mask)[0]
            if len(g_indices) < 20:
                print(f"  SKIP {g_name}: only {len(g_indices)} samples")
                continue

            y_g = eth_labels[g_indices]
            # Check we have both classes
            if len(np.unique(y_g)) < 2:
                print(f"  SKIP {g_name}: only one class in subset")
                continue

            X_g = hs_last_token[best_layer_a][g_indices]
            acc_g, std_g = run_probe(X_g, y_g)
            gender_confound[f"{g_name}_only_accuracy"] = acc_g
            gender_confound[f"{g_name}_only_n_samples"] = len(g_indices)
            print(f"  {g_name}-only probing: {acc_g:.1%} ({len(g_indices)} samples)")

        # Also probe for gender on the ethnicity data
        if len(np.unique(prompt_genders)) == 2:
            X_gp = hs_last_token[best_layer_a]
            acc_gp, std_gp = run_probe(X_gp, prompt_genders)
            gender_confound["gender_probe_on_eth_data"] = acc_gp
            print(f"  Gender probe on ethnicity data: {acc_gp:.1%}")
    else:
        print("  WARNING: No gender annotations in data")

    # =========================================================
    # SAVE FINAL RESULTS
    # =========================================================
    results = {
        "model": model_id,
        "model_short": model_short,
        "model_key": model_key,
        "comparison": comparison,
        "ref_group": ref_group,
        "cmp_group": cmp_group,
        "timestamp": datetime.now().isoformat(),
        "dataset": {
            "n_questions": N_QUESTIONS,
            "n_ref_names": N_PER_GROUP,
            "n_cmp_names": N_PER_GROUP,
            "n_ambiguous_names": N_AMBIGUOUS,
            "n_eth_prompts": n_samples,
        },
        "total_layers": total_layers,
        "hidden_dim": hidden_dim,
        "variant_a_last_token": variant_a,
        "variant_b_question_only": variant_b,
        "variant_c_held_out": variant_c,
        "variant_d_steering": variant_d,
        "ambiguous_control": ambiguous_control,
        "gender_confound_check": gender_confound,
    }

    out_path = results_dir / f"{model_short}_probing_eth_{comparison}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    elapsed = (datetime.now() - t0).total_seconds() / 60

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY — {model_short} / {comparison}")
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
    print(f"  Ambiguous: {n_pred_ref} {ref_group}/{n_pred_cmp} {cmp_group}, dev={mean_confidence:.4f}")
    if gender_confound:
        for k, v in gender_confound.items():
            if "accuracy" in k:
                print(f"  Gender confound ({k}): {v:.1%}")
    print(f"\n  Time: {elapsed:.1f} min")
    print(f"  Saved: {out_path}")
    print(f"{'=' * 70}")

    return results


def main():
    if len(sys.argv) < 3:
        print(f"Usage: python analyze_probing_eth.py <model_key|all> <comparison|all>")
        print(f"  Comparisons: {VALID_COMPARISONS}")
        sys.exit(1)

    model_target = sys.argv[1]
    comp_target = sys.argv[2]

    registry = load_model_registry()

    if model_target == "all":
        model_keys = list(registry.keys())
    elif model_target in registry:
        model_keys = [model_target]
    else:
        print(f"ERROR: Unknown model '{model_target}'. Choose from: {list(registry.keys())} or 'all'")
        sys.exit(1)

    if comp_target == "all":
        comparisons = VALID_COMPARISONS
    elif comp_target in VALID_COMPARISONS:
        comparisons = [comp_target]
    else:
        print(f"ERROR: Unknown comparison '{comp_target}'. Choose from: {VALID_COMPARISONS} or 'all'")
        sys.exit(1)

    all_results = {}
    for comp in comparisons:
        for key in model_keys:
            result = analyze_model(key, comp)
            if result:
                all_results[(key, comp)] = result

    # Cross-comparison summary
    if len(all_results) > 1:
        print(f"\n\n{'=' * 70}")
        print(f"  CROSS-MODEL / CROSS-COMPARISON SUMMARY")
        print(f"{'=' * 70}")

        # Group by comparison
        for comp in comparisons:
            comp_results = {k: v for (k, c), v in all_results.items() if c == comp}
            if not comp_results:
                continue
            print(f"\n  --- {comp} ---")
            print(f"  {'Model':<25} {'Last-tok':<12} {'Q-only':<12} {'Held-out':<12} {'KL Ratio':<12}")
            print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
            for key, r in comp_results.items():
                lt = f"{r['variant_a_last_token']['best_accuracy']:.1%}"
                qo = f"{r['variant_b_question_only']['best_accuracy']:.1%}"
                ho = f"{r['variant_c_held_out'].get('last_token_best_test_acc', 0):.1%}"
                ratio = r['variant_d_steering'].get('strength_0.0', {}).get('ratio', 0)
                kl = f"{ratio:.2f}x" if ratio > 0 else "N/A"
                print(f"  {r['model_short']:<25} {lt:<12} {qo:<12} {ho:<12} {kl:<12}")


if __name__ == "__main__":
    main()
