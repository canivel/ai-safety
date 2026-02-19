#!/usr/bin/env python3
"""
Quick sweep: teacher-forcing response-level KL at multiple ablation strengths
for ETHNICITY probing (adapted from gender version).

For each question:
  1. Generate ONE response (reference-group name, normal condition)
  2. Teacher-force [ref_prompt + R] and [cmp_prompt + R] at each strength
  3. Report per-strength mean KL across all response positions

Usage:
    python run_kl_strength_sweep_eth.py <model_key> <comparison>
    python run_kl_strength_sweep_eth.py gemma4b white_vs_black
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
    print(f"Usage: python run_kl_strength_sweep_eth.py <model_key> <comparison>")
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

# Strengths to sweep -- key range around the optimal removal point
STRENGTHS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
N_QUESTIONS_SWEEP = 30  # Enough for statistical stability
MAX_TOKENS = 256

# === DATASET ===
questions = load_questions()
eth_data = load_ethnicity_names()

ref_names = eth_data[REF_GROUP]["names"]
cmp_names = eth_data[CMP_GROUP]["names"]

N_Q = len(questions)
N_REF = len(ref_names)
N_CMP = len(cmp_names)

ref_prompts = [f"Hi, my name is {ref_names[i % N_REF]}. {questions[i]}" for i in range(N_Q)]
cmp_prompts = [f"Hi, my name is {cmp_names[i % N_CMP]}. {questions[i]}" for i in range(N_Q)]
all_ethnicity_prompts = ref_prompts + cmp_prompts
ethnicity_labels = np.array([0] * N_Q + [1] * N_Q)


def symmetric_kl(logits_a, logits_b):
    p = F.softmax(logits_a.float(), dim=-1).clamp(min=1e-10)
    q = F.softmax(logits_b.float(), dim=-1).clamp(min=1e-10)
    kl_pq = (p * (p.log() - q.log())).sum()
    kl_qp = (q * (q.log() - p.log())).sum()
    return ((kl_pq + kl_qp) / 2).item()


def make_hook(direction, strength):
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hs = output[0]
        else:
            hs = output
        proj = torch.einsum('...d,d->...', hs.float(), direction.float())
        hs_modified = hs.float() - strength * proj.unsqueeze(-1) * direction.float()
        if isinstance(output, tuple):
            return (hs_modified.to(hs.dtype),) + output[1:]
        return hs_modified.to(hs.dtype)
    return hook_fn


def load_model_and_tokenizer(model_id, loader):
    from huggingface_hub import login
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    processor = None
    if loader == "gemma_causal":
        from transformers import AutoTokenizer, Gemma3ForCausalLM
        tok = AutoTokenizer.from_pretrained(model_id)
        model = Gemma3ForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto")
        get_layers = lambda m: m.model.layers
        get_embed = lambda m: m.model.embed_tokens
    elif loader == "gemma_conditional":
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_id)
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto")
        tok = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
        get_layers = lambda m: m.model.language_model.layers
        get_embed = lambda m: m.model.language_model.embed_tokens
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto")
        get_layers = lambda m: m.model.layers
        get_embed = lambda m: m.model.embed_tokens
        processor = None

    model.eval()
    if hasattr(model.config, 'num_hidden_layers'):
        num_layers = model.config.num_hidden_layers
    elif hasattr(model.config, 'text_config'):
        num_layers = model.config.text_config.num_hidden_layers
    else:
        raise ValueError("Cannot find num_hidden_layers")

    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return model, tok, processor, get_layers, get_embed, num_layers


def generate_response(model, tok, processor, text, loader, max_new_tokens=256):
    messages = [{"role": "user", "content": text}]
    if loader == "gemma_conditional" and processor is not None:
        mm_messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        try:
            inputs = processor.apply_chat_template(
                mm_messages, return_tensors="pt",
                add_generation_prompt=True, return_dict=True, tokenize=True)
            inputs = {k: v.to("cuda") for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            input_len = inputs["input_ids"].shape[1]
        except Exception:
            input_ids = tok.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor([input_ids])
            inputs = {"input_ids": input_ids.to("cuda")}
            input_len = inputs["input_ids"].shape[1]
    else:
        input_ids = tok.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor([input_ids])
        inputs = {"input_ids": input_ids.to("cuda")}
        input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    response_ids = output_ids[0, input_len:].cpu()
    return response_ids


def main():
    t0 = datetime.now()
    print("=" * 70)
    print("  KL Strength Sweep — Ethnicity (Teacher-Forcing Response-Level KL)")
    print(f"  Model:      {MODEL_ID}")
    print(f"  Comparison: {COMPARISON} ({REF_GROUP} vs {CMP_GROUP})")
    print(f"  Strengths:  {STRENGTHS}")
    print(f"  Questions:  {N_QUESTIONS_SWEEP}")
    print("=" * 70)

    # Load model
    print(f"\n[1/5] Loading model...")
    model, tok, processor, get_layers, get_embed, num_layers = \
        load_model_and_tokenizer(MODEL_ID, LOADER)
    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Loaded | VRAM: {vram:.1f} GB")

    layers = get_layers(model)
    target_module = layers[BEST_PROBE_LAYER - 1] if BEST_PROBE_LAYER > 0 else get_embed(model)

    # Extract hidden states and get ethnicity direction
    print(f"\n[2/5] Getting ethnicity direction (layer {BEST_PROBE_LAYER})...")
    captured = {}
    def capture_hook(module, input, output):
        if isinstance(output, tuple):
            captured['hs'] = output[0].detach().cpu()
        else:
            captured['hs'] = output.detach().cpu()

    handle = target_module.register_forward_hook(capture_hook)
    hs_list = []
    for text in tqdm(all_ethnicity_prompts, desc="Extracting HS"):
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=128).to("cuda")
        with torch.no_grad():
            model(**inputs)
        hs_list.append(captured['hs'].squeeze(0).float()[-1].numpy())
    handle.remove()
    hs_array = np.array(hs_list)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(hs_array)
    probe = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
    probe.fit(X, ethnicity_labels)
    eth_dir = probe.coef_[0] / scaler.scale_
    eth_dir = eth_dir / np.linalg.norm(eth_dir)
    eth_dir_tensor = torch.tensor(eth_dir, dtype=torch.bfloat16).to("cuda")

    # Random direction control
    rng = np.random.RandomState(42)
    rand_dir = rng.randn(len(eth_dir)).astype(np.float32)
    rand_dir = rand_dir / np.linalg.norm(rand_dir)
    rand_dir_tensor = torch.tensor(rand_dir, dtype=torch.bfloat16).to("cuda")

    del hs_list, hs_array, X
    gc.collect()
    print(f"  Ethnicity direction extracted.")

    # Generate baseline responses
    print(f"\n[3/5] Generating {N_QUESTIONS_SWEEP} baseline responses...")
    responses = []
    for q_idx in tqdm(range(N_QUESTIONS_SWEEP), desc="Generating"):
        r_ids = generate_response(
            model, tok, processor, ref_prompts[q_idx], LOADER, MAX_TOKENS)
        responses.append(r_ids)

    # Sweep strengths — ethnicity direction
    print(f"\n[4/5] Sweeping {len(STRENGTHS)} strengths (ethnicity direction)...")
    results = {
        "model": MODEL_SHORT,
        "comparison": COMPARISON,
        "ref_group": REF_GROUP,
        "cmp_group": CMP_GROUP,
        "strengths": {},
        "random_control": {},
    }

    for strength in STRENGTHS:
        if strength > 0:
            hook_handle = target_module.register_forward_hook(
                make_hook(eth_dir_tensor, strength))

        first_token_kls = []
        response_kls = []

        for q_idx in range(N_QUESTIONS_SWEEP):
            ref_text = ref_prompts[q_idx]
            cmp_text = cmp_prompts[q_idx]
            r_ids = responses[q_idx]

            # Tokenize prompts
            p_ids_ref = tok(ref_text, return_tensors="pt", truncation=True, max_length=128)["input_ids"]
            p_ids_cmp = tok(cmp_text, return_tensors="pt", truncation=True, max_length=128)["input_ids"]

            # Teacher-forcing: [prompt + response]
            full_ref = torch.cat([p_ids_ref[0], r_ids]).unsqueeze(0).to("cuda")
            full_cmp = torch.cat([p_ids_cmp[0], r_ids]).unsqueeze(0).to("cuda")

            with torch.no_grad():
                logits_ref = model(input_ids=full_ref).logits[0]
                logits_cmp = model(input_ids=full_cmp).logits[0]

            plen_ref = p_ids_ref.shape[1]
            plen_cmp = p_ids_cmp.shape[1]
            rlen = len(r_ids)

            # First-token KL
            ft_kl = symmetric_kl(logits_ref[plen_ref - 1], logits_cmp[plen_cmp - 1])
            first_token_kls.append(ft_kl)

            # Per-position response KL
            pos_kls = []
            for t in range(rlen):
                pr = plen_ref - 1 + t
                pc = plen_cmp - 1 + t
                if pr < logits_ref.shape[0] and pc < logits_cmp.shape[0]:
                    pos_kls.append(symmetric_kl(logits_ref[pr], logits_cmp[pc]))
            response_kls.append(float(np.mean(pos_kls)) if pos_kls else 0.0)

        if strength > 0:
            hook_handle.remove()

        mean_ft = float(np.mean(first_token_kls))
        mean_resp = float(np.mean(response_kls))

        results["strengths"][str(strength)] = {
            "mean_first_token_kl": mean_ft,
            "mean_response_kl": mean_resp,
            "first_token_kl_values": [float(v) for v in first_token_kls],
            "response_kl_values": [float(v) for v in response_kls],
        }

        print(f"  [eth]  a={strength:4.1f}: FT_KL={mean_ft:.6f}  Resp_KL={mean_resp:.6f}")

    # Sweep strengths — random direction control
    print(f"\n[5/5] Sweeping {len(STRENGTHS)} strengths (random direction control)...")

    for strength in STRENGTHS:
        if strength > 0:
            hook_handle = target_module.register_forward_hook(
                make_hook(rand_dir_tensor, strength))

        first_token_kls = []
        response_kls = []

        for q_idx in range(N_QUESTIONS_SWEEP):
            ref_text = ref_prompts[q_idx]
            cmp_text = cmp_prompts[q_idx]
            r_ids = responses[q_idx]

            p_ids_ref = tok(ref_text, return_tensors="pt", truncation=True, max_length=128)["input_ids"]
            p_ids_cmp = tok(cmp_text, return_tensors="pt", truncation=True, max_length=128)["input_ids"]

            full_ref = torch.cat([p_ids_ref[0], r_ids]).unsqueeze(0).to("cuda")
            full_cmp = torch.cat([p_ids_cmp[0], r_ids]).unsqueeze(0).to("cuda")

            with torch.no_grad():
                logits_ref = model(input_ids=full_ref).logits[0]
                logits_cmp = model(input_ids=full_cmp).logits[0]

            plen_ref = p_ids_ref.shape[1]
            plen_cmp = p_ids_cmp.shape[1]
            rlen = len(r_ids)

            ft_kl = symmetric_kl(logits_ref[plen_ref - 1], logits_cmp[plen_cmp - 1])
            first_token_kls.append(ft_kl)

            pos_kls = []
            for t in range(rlen):
                pr = plen_ref - 1 + t
                pc = plen_cmp - 1 + t
                if pr < logits_ref.shape[0] and pc < logits_cmp.shape[0]:
                    pos_kls.append(symmetric_kl(logits_ref[pr], logits_cmp[pc]))
            response_kls.append(float(np.mean(pos_kls)) if pos_kls else 0.0)

        if strength > 0:
            hook_handle.remove()

        mean_ft = float(np.mean(first_token_kls))
        mean_resp = float(np.mean(response_kls))

        results["random_control"][str(strength)] = {
            "mean_first_token_kl": mean_ft,
            "mean_response_kl": mean_resp,
            "first_token_kl_values": [float(v) for v in first_token_kls],
            "response_kl_values": [float(v) for v in response_kls],
        }

        print(f"  [rand] a={strength:4.1f}: FT_KL={mean_ft:.6f}  Resp_KL={mean_resp:.6f}")

    # Save
    path = RESULTS_DIR / f"{MODEL_SHORT}_kl_strength_sweep.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {path}")

    elapsed = (datetime.now() - t0).total_seconds() / 60
    print(f"  Time: {elapsed:.1f} min")


if __name__ == "__main__":
    main()
