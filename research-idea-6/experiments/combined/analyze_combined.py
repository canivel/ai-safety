#!/usr/bin/env python3
"""
CPU-only: Combined gender x ethnicity intersection analysis.

Loads hidden states from both the gender study and the ethnicity study,
then performs cross-axis probing, direction orthogonality, and cross-steering
ablation analyses.

Usage:
    python analyze_combined.py gemma4b white_vs_black
    python analyze_combined.py qwen7b white_vs_hispanic
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Shared imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.load_data import (
    load_gender_names,
    load_ethnicity_names,
    load_model_registry,
    get_ethnicity_comparison,
    VALID_COMPARISONS,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
GENDER_RESULTS_DIR = EXPERIMENTS_DIR / "results" / "cross_family_probing_v4"
ETH_RESULTS_DIR = EXPERIMENTS_DIR / "results" / "ethnicity_probing"
OUTPUT_DIR = EXPERIMENTS_DIR / "results" / "combined_analysis"

# ---------------------------------------------------------------------------
# Probe hyperparameters (same as both study scripts)
# ---------------------------------------------------------------------------
PROBE_CV_FOLDS = 5
PROBE_C = 1.0
PROBE_MAX_ITER = 1000

# ---------------------------------------------------------------------------
# Ethnic sub-group indices within the gender-study name list (per gender).
# Each gender list has 45 names.  The ordering in gender_names.json is:
#   0-24  Anglo/White
#   25-34 Hispanic
#   35-39 Asian
#   40-44 Other (MENA/African — classified as White under EEOC)
# Note: These ranges describe the GENDER study's name composition, not the
# EEOC-framed ethnicity study categories.
# ---------------------------------------------------------------------------
GENDER_NAME_ETH_RANGES = {
    "anglo":          range(0, 25),
    "hispanic":       range(25, 35),
    "asian":          range(35, 40),
    "other":          range(40, 45),
}

N_NAMES_PER_GENDER = 45
N_QUESTIONS = 200


# ===================================================================
# Helpers
# ===================================================================

def timestamp():
    return datetime.now().strftime("%H:%M:%S")


def run_probe_cv(X, y, folds=PROBE_CV_FOLDS):
    """5-fold CV probing accuracy, returns (mean, std)."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    probe = LogisticRegression(
        max_iter=PROBE_MAX_ITER, solver="lbfgs", C=PROBE_C,
    )
    scores = cross_val_score(probe, X_s, y, cv=folds, scoring="accuracy")
    return float(scores.mean()), float(scores.std())


def fit_probe_full(X, y):
    """Fit probe on all data, return (probe, scaler)."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    probe = LogisticRegression(
        max_iter=PROBE_MAX_ITER, solver="lbfgs", C=PROBE_C,
    )
    probe.fit(X_s, y)
    return probe, scaler


def extract_direction(probe, scaler):
    """Extract the linear direction from a trained probe, un-scaling it.

    The probe operates in scaled space: decision = w . ((x - mu) / sigma) + b
    So in original space the effective weight vector is w / sigma.
    We normalise to unit length.
    """
    raw = probe.coef_[0] / scaler.scale_
    return raw / np.linalg.norm(raw)


def project_out(X, direction):
    """Remove the component of X along `direction` (unit vector)."""
    # projection of each row onto direction
    dots = X @ direction  # (n,)
    return X - np.outer(dots, direction)


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def find_best_layer(data_dict, labels, total_layers):
    """Sweep layers, return (best_layer, best_acc, all_accs)."""
    accs = []
    for layer in range(total_layers):
        acc, _ = run_probe_cv(data_dict[layer], labels)
        accs.append(acc)
    best = int(np.argmax(accs))
    return best, accs[best], accs


# ===================================================================
# Ethnic annotation for gender-study names
# ===================================================================

def build_gender_study_eth_labels(n_samples, n_names=N_NAMES_PER_GENDER):
    """Return binary labels: 0 = Anglo, 1 = non-Anglo for each sample.

    The gender .npz has n_samples = N_QUESTIONS * 2  (male block, then female).
    Within each block the names cycle: sample i -> name index (i % n_names).
    """
    anglo_set = set(GENDER_NAME_ETH_RANGES["anglo"])
    labels = np.zeros(n_samples, dtype=int)
    half = n_samples // 2
    for i in range(n_samples):
        name_idx = i % n_names
        if name_idx not in anglo_set:
            labels[i] = 1
    return labels


# ===================================================================
# Main analysis
# ===================================================================

def analyze_combined(model_key, comparison):
    registry = load_model_registry()
    if model_key not in registry:
        print(f"ERROR: Unknown model key '{model_key}'. "
              f"Choose from: {list(registry.keys())}")
        sys.exit(1)

    cfg = registry[model_key]
    model_short = cfg["model_short"]
    ref_group, cmp_group = get_ethnicity_comparison(comparison)

    gender_hs_path = GENDER_RESULTS_DIR / f"{model_short}_hidden_states.npz"
    eth_hs_path = ETH_RESULTS_DIR / comparison / f"{model_short}_hidden_states.npz"

    print(f"\n{'=' * 70}")
    print(f"  Combined gender x ethnicity analysis")
    print(f"  Model: {model_short}  |  Comparison: {comparison}")
    print(f"{'=' * 70}")

    t0 = datetime.now()

    # ------------------------------------------------------------------
    # Load gender hidden states
    # ------------------------------------------------------------------
    print(f"\n[{timestamp()}] Loading gender hidden states...")
    if not gender_hs_path.exists():
        print(f"  SKIP: Gender data not found at {gender_hs_path}")
        print(f"  Cannot proceed without gender data.")
        return None
    gender_data = np.load(gender_hs_path)
    gender_total_layers = int(gender_data["total_layers"][0])
    gender_labels = gender_data["gender_labels"]  # 0=male, 1=female
    gender_hs = {}
    for layer in range(gender_total_layers):
        gender_hs[layer] = gender_data[f"last_token_layer_{layer}"]
    n_gender_samples = gender_hs[0].shape[0]
    gender_hidden_dim = gender_hs[0].shape[1]
    print(f"  Loaded: {n_gender_samples} samples, {gender_total_layers} layers, "
          f"dim={gender_hidden_dim}")

    # ------------------------------------------------------------------
    # Load ethnicity hidden states
    # ------------------------------------------------------------------
    print(f"[{timestamp()}] Loading ethnicity hidden states...")
    if not eth_hs_path.exists():
        print(f"  SKIP: Ethnicity data not found at {eth_hs_path}")
        print(f"  Cannot proceed without ethnicity data.")
        return None
    eth_data = np.load(eth_hs_path)
    eth_total_layers = int(eth_data["total_layers"][0])
    eth_labels = eth_data["eth_labels"]  # 0=ref, 1=cmp
    eth_prompt_genders = eth_data.get("prompt_genders", None)
    eth_hs = {}
    for layer in range(eth_total_layers):
        eth_hs[layer] = eth_data[f"last_token_layer_{layer}"]
    n_eth_samples = eth_hs[0].shape[0]
    eth_hidden_dim = eth_hs[0].shape[1]
    print(f"  Loaded: {n_eth_samples} samples, {eth_total_layers} layers, "
          f"dim={eth_hidden_dim}")

    # Sanity: both should share the same hidden dim
    if gender_hidden_dim != eth_hidden_dim:
        print(f"  WARNING: Hidden dim mismatch! gender={gender_hidden_dim}, "
              f"eth={eth_hidden_dim}")

    results = {
        "model": cfg["model_id"],
        "model_short": model_short,
        "model_key": model_key,
        "comparison": comparison,
        "ref_group": ref_group,
        "cmp_group": cmp_group,
        "timestamp": datetime.now().isoformat(),
        "gender_n_samples": n_gender_samples,
        "gender_total_layers": gender_total_layers,
        "eth_n_samples": n_eth_samples,
        "eth_total_layers": eth_total_layers,
        "hidden_dim": gender_hidden_dim,
    }

    # ==================================================================
    # ANALYSIS 1: Cross-axis probing
    # ==================================================================
    print(f"\n{'=' * 70}")
    print(f"  Analysis 1: Cross-axis probing")
    print(f"{'=' * 70}")

    cross_axis = {}

    # --- 1a: Ethnicity probe on gender-study data ---
    print(f"\n[{timestamp()}] 1a: Can we detect ethnicity in the gender experiment?")
    print(f"  Building Anglo vs non-Anglo labels from gender-study names...")
    gender_eth_labels = build_gender_study_eth_labels(n_gender_samples)
    n_anglo = int((gender_eth_labels == 0).sum())
    n_nonanglo = int((gender_eth_labels == 1).sum())
    print(f"  Anglo: {n_anglo}, non-Anglo: {n_nonanglo}")

    if len(np.unique(gender_eth_labels)) < 2:
        print(f"  SKIP: Only one class present")
        cross_axis["eth_probe_on_gender_data"] = {"skipped": True}
    else:
        print(f"  Sweeping {gender_total_layers} layers for ethnicity signal...")
        best_layer_1a, best_acc_1a, all_accs_1a = find_best_layer(
            gender_hs, gender_eth_labels, gender_total_layers,
        )
        print(f"  Best layer: {best_layer_1a}, accuracy: {best_acc_1a:.1%}")
        cross_axis["eth_probe_on_gender_data"] = {
            "task": "Anglo vs non-Anglo probe on gender-study hidden states",
            "best_layer": best_layer_1a,
            "best_accuracy": best_acc_1a,
            "layer_accuracies": all_accs_1a,
            "n_anglo": n_anglo,
            "n_nonanglo": n_nonanglo,
        }

    # --- 1b: Gender probe on ethnicity-study data ---
    print(f"\n[{timestamp()}] 1b: Can we detect gender in the ethnicity experiment?")
    if eth_prompt_genders is None:
        print(f"  SKIP: No prompt_genders in ethnicity data")
        cross_axis["gender_probe_on_eth_data"] = {"skipped": True}
    elif len(np.unique(eth_prompt_genders)) < 2:
        print(f"  SKIP: Only one gender class in ethnicity data")
        cross_axis["gender_probe_on_eth_data"] = {"skipped": True}
    else:
        n_male_eth = int((eth_prompt_genders == 0).sum())
        n_female_eth = int((eth_prompt_genders == 1).sum())
        print(f"  Male: {n_male_eth}, Female: {n_female_eth}")

        print(f"  Sweeping {eth_total_layers} layers for gender signal...")
        best_layer_1b, best_acc_1b, all_accs_1b = find_best_layer(
            eth_hs, eth_prompt_genders, eth_total_layers,
        )
        print(f"  Best layer: {best_layer_1b}, accuracy: {best_acc_1b:.1%}")
        cross_axis["gender_probe_on_eth_data"] = {
            "task": "Gender probe on ethnicity-study hidden states",
            "best_layer": best_layer_1b,
            "best_accuracy": best_acc_1b,
            "layer_accuracies": all_accs_1b,
            "n_male": n_male_eth,
            "n_female": n_female_eth,
        }

    results["analysis_1_cross_axis_probing"] = cross_axis

    # ==================================================================
    # ANALYSIS 2: Direction orthogonality
    # ==================================================================
    print(f"\n{'=' * 70}")
    print(f"  Analysis 2: Direction orthogonality")
    print(f"{'=' * 70}")

    direction_analysis = {}

    # Find best layer for gender probing (on gender data)
    print(f"\n[{timestamp()}] Finding best gender probing layer...")
    best_gender_layer, best_gender_acc, _ = find_best_layer(
        gender_hs, gender_labels, gender_total_layers,
    )
    print(f"  Gender best layer: {best_gender_layer}, acc: {best_gender_acc:.1%}")

    # Find best layer for ethnicity probing (on ethnicity data)
    print(f"[{timestamp()}] Finding best ethnicity probing layer...")
    best_eth_layer, best_eth_acc, _ = find_best_layer(
        eth_hs, eth_labels, eth_total_layers,
    )
    print(f"  Ethnicity best layer: {best_eth_layer}, acc: {best_eth_acc:.1%}")

    # Extract gender direction at best gender layer
    print(f"[{timestamp()}] Extracting gender direction (layer {best_gender_layer})...")
    gender_probe, gender_scaler = fit_probe_full(
        gender_hs[best_gender_layer], gender_labels,
    )
    gender_direction = extract_direction(gender_probe, gender_scaler)

    # Extract ethnicity direction at best ethnicity layer
    print(f"[{timestamp()}] Extracting ethnicity direction (layer {best_eth_layer})...")
    eth_probe, eth_scaler = fit_probe_full(
        eth_hs[best_eth_layer], eth_labels,
    )
    eth_direction = extract_direction(eth_probe, eth_scaler)

    # Compute cosine similarity
    cos_sim = cosine_similarity(gender_direction, eth_direction)
    print(f"\n  Cosine similarity between gender and ethnicity directions: {cos_sim:.4f}")
    print(f"  Interpretation: ", end="")
    if abs(cos_sim) < 0.1:
        print("NEAR-ORTHOGONAL -- signals are largely independent")
    elif abs(cos_sim) < 0.3:
        print("WEAKLY CORRELATED -- some shared variance")
    elif abs(cos_sim) < 0.6:
        print("MODERATELY CORRELATED -- notable overlap")
    else:
        print("STRONGLY CORRELATED -- directions are conflated")

    direction_analysis["gender_best_layer"] = best_gender_layer
    direction_analysis["gender_best_accuracy"] = best_gender_acc
    direction_analysis["eth_best_layer"] = best_eth_layer
    direction_analysis["eth_best_accuracy"] = best_eth_acc
    direction_analysis["cosine_similarity"] = cos_sim
    direction_analysis["abs_cosine_similarity"] = abs(cos_sim)

    results["analysis_2_direction_orthogonality"] = direction_analysis

    # ==================================================================
    # ANALYSIS 3: Cross-steering (ablation)
    # ==================================================================
    print(f"\n{'=' * 70}")
    print(f"  Analysis 3: Cross-steering ablation")
    print(f"{'=' * 70}")

    ablation = {}

    # --- 3a: Ablate gender direction from ethnicity data ---
    print(f"\n[{timestamp()}] 3a: Ablating gender direction from ethnicity data...")
    print(f"  Before ablation: ethnicity probe accuracy...")

    # Ethnicity probe on ethnicity data BEFORE ablation (at best eth layer)
    eth_before_acc, eth_before_std = run_probe_cv(
        eth_hs[best_eth_layer], eth_labels,
    )
    print(f"    Ethnicity accuracy (before): {eth_before_acc:.1%} +/- {eth_before_std:.1%}")

    # Ablate gender direction from the ethnicity hidden states
    # The gender direction lives in the same feature space if same model.
    # We use the gender direction extracted from best_gender_layer,
    # but apply it to the ethnicity data at best_eth_layer.
    # If layers differ, directions may not be perfectly aligned,
    # but this is a first-order approximation.
    eth_ablated = project_out(eth_hs[best_eth_layer], gender_direction)
    eth_after_acc, eth_after_std = run_probe_cv(eth_ablated, eth_labels)
    print(f"    Ethnicity accuracy (after gender ablation): {eth_after_acc:.1%} +/- {eth_after_std:.1%}")
    eth_drop = eth_before_acc - eth_after_acc
    print(f"    Drop: {eth_drop:+.1%}")

    ablation["eth_before_gender_ablation"] = {
        "layer": best_eth_layer,
        "accuracy": eth_before_acc,
        "std": eth_before_std,
    }
    ablation["eth_after_gender_ablation"] = {
        "layer": best_eth_layer,
        "accuracy": eth_after_acc,
        "std": eth_after_std,
        "accuracy_drop": eth_drop,
    }

    # --- 3b: Ablate ethnicity direction from gender data ---
    print(f"\n[{timestamp()}] 3b: Ablating ethnicity direction from gender data...")
    print(f"  Before ablation: gender probe accuracy...")

    gender_before_acc, gender_before_std = run_probe_cv(
        gender_hs[best_gender_layer], gender_labels,
    )
    print(f"    Gender accuracy (before): {gender_before_acc:.1%} +/- {gender_before_std:.1%}")

    gender_ablated = project_out(gender_hs[best_gender_layer], eth_direction)
    gender_after_acc, gender_after_std = run_probe_cv(gender_ablated, gender_labels)
    print(f"    Gender accuracy (after ethnicity ablation): {gender_after_acc:.1%} +/- {gender_after_std:.1%}")
    gender_drop = gender_before_acc - gender_after_acc
    print(f"    Drop: {gender_drop:+.1%}")

    ablation["gender_before_eth_ablation"] = {
        "layer": best_gender_layer,
        "accuracy": gender_before_acc,
        "std": gender_before_std,
    }
    ablation["gender_after_eth_ablation"] = {
        "layer": best_gender_layer,
        "accuracy": gender_after_acc,
        "std": gender_after_std,
        "accuracy_drop": gender_drop,
    }

    # Interpretation
    print(f"\n  Interpretation:")
    if abs(eth_drop) < 0.02 and abs(gender_drop) < 0.02:
        interp = "CAUSALLY INDEPENDENT -- ablating one axis does not affect the other"
    elif abs(eth_drop) < 0.05 and abs(gender_drop) < 0.05:
        interp = "MOSTLY INDEPENDENT -- minimal causal coupling"
    elif abs(eth_drop) > 0.1 or abs(gender_drop) > 0.1:
        interp = "CAUSALLY ENTANGLED -- removing one direction substantially affects the other"
    else:
        interp = "WEAKLY ENTANGLED -- some causal coupling between axes"
    print(f"    {interp}")
    ablation["interpretation"] = interp

    results["analysis_3_cross_steering_ablation"] = ablation

    # ==================================================================
    # SAVE
    # ==================================================================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{model_short}_combined_{comparison}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    elapsed = (datetime.now() - t0).total_seconds() / 60

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY -- {model_short} / {comparison}")
    print(f"{'=' * 70}")

    print(f"\n  Analysis 1: Cross-axis probing")
    ca = results["analysis_1_cross_axis_probing"]
    if not ca["eth_probe_on_gender_data"].get("skipped"):
        print(f"    Ethnicity in gender data: {ca['eth_probe_on_gender_data']['best_accuracy']:.1%} "
              f"at L{ca['eth_probe_on_gender_data']['best_layer']}")
    else:
        print(f"    Ethnicity in gender data: SKIPPED")
    if not ca["gender_probe_on_eth_data"].get("skipped"):
        print(f"    Gender in ethnicity data: {ca['gender_probe_on_eth_data']['best_accuracy']:.1%} "
              f"at L{ca['gender_probe_on_eth_data']['best_layer']}")
    else:
        print(f"    Gender in ethnicity data: SKIPPED")

    print(f"\n  Analysis 2: Direction orthogonality")
    da = results["analysis_2_direction_orthogonality"]
    print(f"    Gender direction: layer {da['gender_best_layer']} ({da['gender_best_accuracy']:.1%})")
    print(f"    Ethnicity direction: layer {da['eth_best_layer']} ({da['eth_best_accuracy']:.1%})")
    print(f"    Cosine similarity: {da['cosine_similarity']:.4f}")

    print(f"\n  Analysis 3: Cross-steering ablation")
    ab = results["analysis_3_cross_steering_ablation"]
    print(f"    Ethnicity acc: {ab['eth_before_gender_ablation']['accuracy']:.1%} "
          f"-> {ab['eth_after_gender_ablation']['accuracy']:.1%} "
          f"(drop: {ab['eth_after_gender_ablation']['accuracy_drop']:+.1%})")
    print(f"    Gender acc:    {ab['gender_before_eth_ablation']['accuracy']:.1%} "
          f"-> {ab['gender_after_eth_ablation']['accuracy']:.1%} "
          f"(drop: {ab['gender_after_eth_ablation']['accuracy_drop']:+.1%})")
    print(f"    {ab['interpretation']}")

    print(f"\n  Time: {elapsed:.1f} min")
    print(f"  Saved: {out_path}")
    print(f"{'=' * 70}")

    return results


def main():
    if len(sys.argv) < 3:
        print(f"Usage: python analyze_combined.py <model_key> <comparison>")
        print(f"  Example: python analyze_combined.py gemma4b white_vs_black")
        print(f"  Valid comparisons: {VALID_COMPARISONS}")
        registry = load_model_registry()
        print(f"  Valid model keys: {list(registry.keys())}")
        sys.exit(1)

    model_key = sys.argv[1]
    comparison = sys.argv[2]

    if comparison not in VALID_COMPARISONS:
        print(f"ERROR: Unknown comparison '{comparison}'. "
              f"Choose from: {VALID_COMPARISONS}")
        sys.exit(1)

    analyze_combined(model_key, comparison)


if __name__ == "__main__":
    main()
