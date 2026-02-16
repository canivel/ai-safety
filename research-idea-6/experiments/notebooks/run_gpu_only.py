#!/usr/bin/env python3
"""
GPU-only extraction: hidden states + KL divergence for all models.
Saves raw data so permutation tests can run locally (CPU-only).

Usage:
    python run_gpu_only.py --models 1b 4b 12b 27b
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import gc
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

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

RESULTS_DIR = Path("../results/gemma3_gender_detection")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset
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


def symmetric_kl(logits_m, logits_f):
    p = F.softmax(logits_m.float(), dim=-1).clamp(min=1e-10)
    q = F.softmax(logits_f.float(), dim=-1).clamp(min=1e-10)
    kl_pq = (p * (p.log() - q.log())).sum()
    kl_qp = (q * (q.log() - p.log())).sum()
    return ((kl_pq + kl_qp) / 2).item()


def run_model(model_id):
    from transformers import AutoTokenizer, AutoProcessor

    model_short = model_id.split("/")[-1]
    is_text_only = (model_id == "google/gemma-3-1b-it")
    num_layers = MODEL_REGISTRY[model_id]["num_layers"]

    hs_path = RESULTS_DIR / f"{model_short}_hidden_states.npz"
    kl_path = RESULTS_DIR / f"{model_short}_kl_divergence.json"

    if hs_path.exists() and kl_path.exists():
        print(f"  [SKIP] {model_short} — already extracted.")
        return

    print(f"\n{'#' * 70}")
    print(f"# {model_id}")
    print(f"{'#' * 70}\n")

    # Load model
    print(f"[{model_short}] Loading...")
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
    tok = tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else tokenizer

    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"[{model_short}] Loaded | VRAM: {vram:.1f} GB")

    # === 1. Extract hidden states ===
    if not hs_path.exists():
        print(f"[{model_short}] Extracting hidden states...")
        all_hidden = {layer: [] for layer in range(num_layers + 1)}
        for text in tqdm(all_prompts, desc=f"[{model_short}] Hidden states"):
            inputs = tok(text, return_tensors="pt", truncation=True, max_length=128).to("cuda")
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            for layer_idx, hs in enumerate(outputs.hidden_states):
                mean_repr = hs.squeeze(0).mean(dim=0).float().cpu().numpy()
                all_hidden[layer_idx].append(mean_repr)

        # Save as npz (compact)
        arrays = {f"layer_{i}": np.array(all_hidden[i]) for i in all_hidden}
        np.savez_compressed(hs_path, **arrays)
        print(f"[{model_short}] Hidden states saved → {hs_path.name}")

    # === 2. KL divergence on first-token logits ===
    if not kl_path.exists():
        print(f"[{model_short}] Computing KL divergence...")
        kl_results = []
        for i, question in enumerate(tqdm(questions, desc=f"[{model_short}] KL")):
            male_text = f"Hi, my name is {male_names[i]}. {question}"
            female_text = f"Hi, my name is {female_names[i]}. {question}"

            inputs_m = tok(male_text, return_tensors="pt", truncation=True, max_length=128).to("cuda")
            inputs_f = tok(female_text, return_tensors="pt", truncation=True, max_length=128).to("cuda")

            with torch.no_grad():
                logits_m = model(**inputs_m).logits[0, -1, :]
                logits_f = model(**inputs_f).logits[0, -1, :]

            kl_val = symmetric_kl(logits_m, logits_f)

            top_m = torch.topk(logits_m.float(), 5)
            top_f = torch.topk(logits_f.float(), 5)

            kl_results.append({
                "question": question,
                "male_name": male_names[i],
                "female_name": female_names[i],
                "symmetric_kl": kl_val,
                "top5_male": [tok.decode([t]) for t in top_m.indices.tolist()],
                "top5_female": [tok.decode([t]) for t in top_f.indices.tolist()],
            })

        # Same-gender control: male_name_A vs male_name_B for same question
        control_kl = []
        for i in range(len(questions)):
            name_a = male_names[i]
            name_b = male_names[(i + 1) % len(male_names)]
            text_a = f"Hi, my name is {name_a}. {questions[i]}"
            text_b = f"Hi, my name is {name_b}. {questions[i]}"

            inputs_a = tok(text_a, return_tensors="pt", truncation=True, max_length=128).to("cuda")
            inputs_b = tok(text_b, return_tensors="pt", truncation=True, max_length=128).to("cuda")

            with torch.no_grad():
                logits_a = model(**inputs_a).logits[0, -1, :]
                logits_b = model(**inputs_b).logits[0, -1, :]

            control_kl.append(symmetric_kl(logits_a, logits_b))

        kl_values = [r["symmetric_kl"] for r in kl_results]
        mean_kl = float(np.mean(kl_values))
        control_mean = float(np.mean(control_kl))

        print(f"[{model_short}] Cross-gender KL: {mean_kl:.4f}")
        print(f"[{model_short}] Same-gender KL:  {control_mean:.4f}")
        print(f"[{model_short}] Ratio: {mean_kl/control_mean:.2f}x")

        kl_data = {
            "model_id": model_id,
            "model_short": model_short,
            "timestamp": datetime.now().isoformat(),
            "mean_symmetric_kl": mean_kl,
            "median_symmetric_kl": float(np.median(kl_values)),
            "max_symmetric_kl": float(np.max(kl_values)),
            "control_mean_kl": control_mean,
            "control_median_kl": float(np.median(control_kl)),
            "kl_ratio": mean_kl / control_mean if control_mean > 0 else 0,
            "per_question": kl_results,
            "control_kl_values": [float(v) for v in control_kl],
        }
        with open(kl_path, "w") as f:
            json.dump(kl_data, f, indent=2)
        print(f"[{model_short}] KL saved → {kl_path.name}")

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[{model_short}] Done.\n")


if __name__ == "__main__":
    MODEL_SHORTCUTS = {
        "1b": "google/gemma-3-1b-it",
        "4b": "google/gemma-3-4b-it",
        "12b": "google/gemma-3-12b-it",
        "27b": "google/gemma-3-27b-it",
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=list(MODEL_SHORTCUTS.keys()), default=None)
    args = parser.parse_args()

    models_to_run = [MODEL_SHORTCUTS[m] for m in args.models] if args.models else MODELS

    print("=" * 70)
    print("  GPU-only extraction: Hidden States + KL Divergence")
    print("=" * 70)
    print(f"Models: {[m.split('/')[-1] for m in models_to_run]}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    from huggingface_hub import login
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:
        print("ERROR: Set HF_TOKEN"); exit(1)

    t0 = datetime.now()
    for model_id in models_to_run:
        try:
            run_model(model_id)
        except Exception as e:
            print(f"[ERROR] {model_id}: {e}")
            import traceback; traceback.print_exc()
            gc.collect(); torch.cuda.empty_cache()

    print(f"\nDone in {(datetime.now()-t0).total_seconds()/60:.1f} min")
    print(f"Now copy results locally and run: python run_local_analysis.py")
