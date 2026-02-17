#!/usr/bin/env python3
"""
Quick sweep: teacher-forcing response-level KL at multiple ablation strengths.

For each question:
  1. Generate ONE response (male name, normal condition)
  2. Teacher-force [male_prompt + R] and [female_prompt + R] at each strength
  3. Report per-strength mean KL across all response positions

This extends the existing first-token steering data to show that response-level
KL follows the same U-shaped dose-response curve.

Usage:
    python run_kl_strength_sweep.py gemma4b
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

MODEL_REGISTRY = {
    "gemma4b": {
        "model_id": "google/gemma-3-4b-it",
        "model_short": "gemma-3-4b-it",
        "gated": True,
        "loader": "gemma_conditional",
        "best_probe_layer": 17,
    },
    "mistral7b": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "model_short": "mistral-7b-instruct-v0.3",
        "gated": True,
        "loader": "auto",
        "best_probe_layer": 15,
    },
}

if len(sys.argv) < 2:
    print(f"Usage: python run_kl_strength_sweep.py <model_key>")
    sys.exit(1)

MODEL_KEY = sys.argv[1]
MODEL_CFG = MODEL_REGISTRY[MODEL_KEY]
MODEL_ID = MODEL_CFG["model_id"]
MODEL_SHORT = MODEL_CFG["model_short"]
LOADER = MODEL_CFG.get("loader", "auto")
BEST_PROBE_LAYER = MODEL_CFG["best_probe_layer"]

RESULTS_DIR = Path("../results/causal_mediation")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Strengths to sweep — key range around the optimal removal point
STRENGTHS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
N_QUESTIONS_SWEEP = 30  # Enough for statistical stability
MAX_TOKENS = 256

# Dataset (first 100 questions only — sufficient for sweep)
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
]

male_names = [
    "James", "Michael", "Robert", "David", "William",
    "Thomas", "Daniel", "Matthew", "Andrew", "Christopher",
    "Joseph", "Brian", "Steven", "Kevin", "Timothy",
    "Mark", "Richard", "Charles", "Patrick", "Jason",
    "Eric", "Nathan", "Adam", "Jeffrey", "Gregory",
    "Carlos", "Miguel", "Diego", "Luis", "Marco",
]

female_names = [
    "Sarah", "Emily", "Jessica", "Ashley", "Amanda",
    "Elizabeth", "Jennifer", "Megan", "Rachel", "Lauren",
    "Sophia", "Hannah", "Olivia", "Nicole", "Samantha",
    "Victoria", "Katherine", "Rebecca", "Stephanie", "Christina",
    "Michelle", "Natalie", "Angela", "Diana", "Laura",
    "Maria", "Carmen", "Isabella", "Valentina", "Gabriela",
]

N_Q = len(questions)
N_M = len(male_names)
N_F = len(female_names)

male_prompts = [f"Hi, my name is {male_names[i % N_M]}. {questions[i]}" for i in range(N_Q)]
female_prompts = [f"Hi, my name is {female_names[i % N_F]}. {questions[i]}" for i in range(N_Q)]
all_gendered = male_prompts + female_prompts
gender_labels = np.array([0] * N_Q + [1] * N_Q)


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
    if loader == "gemma_conditional":
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
    print("  KL Strength Sweep (Teacher-Forcing Response-Level KL)")
    print(f"  Model: {MODEL_ID}")
    print(f"  Strengths: {STRENGTHS}")
    print(f"  Questions: {N_QUESTIONS_SWEEP}")
    print("=" * 70)

    # Load model
    print(f"\n[1/4] Loading model...")
    model, tok, processor, get_layers, get_embed, num_layers = \
        load_model_and_tokenizer(MODEL_ID, LOADER)
    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Loaded | VRAM: {vram:.1f} GB")

    layers = get_layers(model)
    target_module = layers[BEST_PROBE_LAYER - 1] if BEST_PROBE_LAYER > 0 else get_embed(model)

    # Extract hidden states and get gender direction
    print(f"\n[2/4] Getting gender direction (layer {BEST_PROBE_LAYER})...")
    captured = {}
    def capture_hook(module, input, output):
        if isinstance(output, tuple):
            captured['hs'] = output[0].detach().cpu()
        else:
            captured['hs'] = output.detach().cpu()

    handle = target_module.register_forward_hook(capture_hook)
    hs_list = []
    for text in tqdm(all_gendered, desc="Extracting HS"):
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
    probe.fit(X, gender_labels)
    gender_dir = probe.coef_[0] / scaler.scale_
    gender_dir = gender_dir / np.linalg.norm(gender_dir)
    gender_dir_tensor = torch.tensor(gender_dir, dtype=torch.bfloat16).to("cuda")
    del hs_list, hs_array, X
    gc.collect()
    print(f"  Direction extracted.")

    # Generate baseline responses
    print(f"\n[3/4] Generating {N_QUESTIONS_SWEEP} baseline responses...")
    responses = []
    for q_idx in tqdm(range(N_QUESTIONS_SWEEP), desc="Generating"):
        r_ids = generate_response(
            model, tok, processor, male_prompts[q_idx], LOADER, MAX_TOKENS)
        responses.append(r_ids)

    # Sweep strengths
    print(f"\n[4/4] Sweeping {len(STRENGTHS)} strengths...")
    results = {"model": MODEL_SHORT, "strengths": {}}

    for strength in STRENGTHS:
        if strength > 0:
            hook_handle = target_module.register_forward_hook(
                make_hook(gender_dir_tensor, strength))

        first_token_kls = []
        response_kls = []

        for q_idx in range(N_QUESTIONS_SWEEP):
            m_text = male_prompts[q_idx]
            f_text = female_prompts[q_idx]
            r_ids = responses[q_idx]

            # Tokenize prompts
            p_ids_m = tok(m_text, return_tensors="pt", truncation=True, max_length=128)["input_ids"]
            p_ids_f = tok(f_text, return_tensors="pt", truncation=True, max_length=128)["input_ids"]

            # Teacher-forcing: [prompt + response]
            full_m = torch.cat([p_ids_m[0], r_ids]).unsqueeze(0).to("cuda")
            full_f = torch.cat([p_ids_f[0], r_ids]).unsqueeze(0).to("cuda")

            with torch.no_grad():
                logits_m = model(input_ids=full_m).logits[0]
                logits_f = model(input_ids=full_f).logits[0]

            plen_m = p_ids_m.shape[1]
            plen_f = p_ids_f.shape[1]
            rlen = len(r_ids)

            # First-token KL
            ft_kl = symmetric_kl(logits_m[plen_m - 1], logits_f[plen_f - 1])
            first_token_kls.append(ft_kl)

            # Per-position response KL
            pos_kls = []
            for t in range(rlen):
                pm = plen_m - 1 + t
                pf = plen_f - 1 + t
                if pm < logits_m.shape[0] and pf < logits_f.shape[0]:
                    pos_kls.append(symmetric_kl(logits_m[pm], logits_f[pf]))
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

        print(f"  α={strength:4.1f}: FT_KL={mean_ft:.6f}  Resp_KL={mean_resp:.6f}")

    # Save
    path = RESULTS_DIR / f"{MODEL_SHORT}_kl_strength_sweep.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {path}")

    elapsed = (datetime.now() - t0).total_seconds() / 60
    print(f"  Time: {elapsed:.1f} min")


if __name__ == "__main__":
    main()
