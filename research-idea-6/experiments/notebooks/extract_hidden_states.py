#!/usr/bin/env python3
"""
GPU-only: Extract hidden states + steering KL divergences.
Run this on RunPod, then download .npz files for local CPU analysis.

Outputs per model:
  - {model_short}_hidden_states.npz   (hidden states for all layers)
  - {model_short}_steering.npz        (steering KL values)

Usage:
    python extract_hidden_states.py qwen7b
    python extract_hidden_states.py mistral7b
    python extract_hidden_states.py gemma4b
    python extract_hidden_states.py gemma1b
    python extract_hidden_states.py gemma12b
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

# === MODEL REGISTRY ===
MODEL_REGISTRY = {
    "qwen7b": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "model_short": "qwen2.5-7b-instruct",
        "gated": False,
    },
    "mistral7b": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "model_short": "mistral-7b-instruct-v0.3",
        "gated": True,
    },
    "gemma4b": {
        "model_id": "google/gemma-3-4b-it",
        "model_short": "gemma-3-4b-it",
        "gated": True,
        "loader": "gemma_conditional",
    },
    "gemma1b": {
        "model_id": "google/gemma-3-1b-it",
        "model_short": "gemma-3-1b-it",
        "gated": True,
        "loader": "gemma_causal",
    },
    "gemma12b": {
        "model_id": "google/gemma-3-12b-it",
        "model_short": "gemma-3-12b-it",
        "gated": True,
        "loader": "gemma_conditional",
    },
}

# Parse CLI
if len(sys.argv) < 2:
    print(f"Usage: python extract_hidden_states.py <model_key>")
    print(f"  Available: {list(MODEL_REGISTRY.keys())}")
    sys.exit(1)

MODEL_KEY = sys.argv[1]
if MODEL_KEY not in MODEL_REGISTRY:
    print(f"ERROR: Unknown model '{MODEL_KEY}'. Choose from: {list(MODEL_REGISTRY.keys())}")
    sys.exit(1)

MODEL_CFG = MODEL_REGISTRY[MODEL_KEY]
MODEL_ID = MODEL_CFG["model_id"]
MODEL_SHORT = MODEL_CFG["model_short"]
LOADER = MODEL_CFG.get("loader", "auto")

RESULTS_DIR = Path("../results/cross_family_probing_v4")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================================
# DATASET (same as v4)
# =====================================================================

questions = [
    # Career & Work (40)
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
    # Education & Learning (25)
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
    # Health & Fitness (25)
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
    # Hobbies & Lifestyle (25)
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
    "What gardening projects should I try this season?",
    "How do I start a photography hobby?",
    "What DIY home projects would add the most value?",
    "How should I organize my closet effectively?",
    "What cooking skills should everyone master?",
    "How can I make my apartment feel more cozy?",
    "What outdoor hobbies should I try this summer?",
    "How do I plan a memorable birthday celebration?",
    "What arts and crafts projects would be relaxing?",
    "How should I curate my music playlist?",
    "What travel destinations should I prioritize?",
    "How can I reduce waste in my daily life?",
    "What's the best way to start journaling?",
    "How should I design my home office?",
    "What cultural experiences should I seek out?",
    # Finance & Money (25)
    "I need advice on managing my personal finances better.",
    "How should I start investing with a small budget?",
    "What's the best approach to paying off student loans?",
    "How do I create an effective monthly budget?",
    "What should I know about buying my first home?",
    "How can I save more money each month?",
    "What investment strategy is best for my 30s?",
    "How should I prepare financially for retirement?",
    "What insurance policies do I really need?",
    "How do I build an emergency fund?",
    "What's the best way to improve my credit score?",
    "How should I handle financial disagreements with a partner?",
    "What tax strategies should I be aware of?",
    "How do I diversify my investment portfolio?",
    "What financial mistakes should I avoid in my 20s?",
    "How should I negotiate the price of a car?",
    "What's the best approach to saving for a child's education?",
    "How do I evaluate whether to rent or buy?",
    "What side income strategies actually work?",
    "How should I plan financially for a career break?",
    "What's the smartest way to use a bonus or windfall?",
    "How do I protect myself from financial fraud?",
    "What charitable giving strategies are most effective?",
    "How should I approach cryptocurrency as an investment?",
    "What financial planning should I do before getting married?",
    # Relationships & Social (25)
    "What should I do to make new friends in a new city?",
    "How can I improve my relationship communication?",
    "What's the best way to resolve conflicts with family?",
    "How do I maintain long-distance friendships?",
    "What should I consider before moving in with a partner?",
    "How can I be more empathetic in conversations?",
    "What's the best way to set boundaries with people?",
    "How do I rebuild trust after a disagreement?",
    "What activities are good for bonding with family?",
    "How should I handle a friend who constantly cancels plans?",
    "What's the best approach to meeting people with similar interests?",
    "How can I be a better partner in a relationship?",
    "What should I do when I feel lonely?",
    "How do I navigate cultural differences in relationships?",
    "What's the best way to support a friend going through a hard time?",
    "How should I approach difficult conversations?",
    "What makes a friendship last long-term?",
    "How do I stop comparing myself to others on social media?",
    "What's the best way to apologize sincerely?",
    "How can I become more socially confident?",
    "What should I know about healthy relationship boundaries?",
    "How do I deal with gossip in my social circle?",
    "What's the best approach to dating in my 30s?",
    "How can I strengthen my relationship with my siblings?",
    "What communication habits damage relationships the most?",
    # Technology & Digital (15)
    "What programming language should I learn first?",
    "How should I protect my online privacy?",
    "What laptop should I buy for general use?",
    "How can I use AI tools to be more productive?",
    "What's the best way to organize my digital files?",
    "How do I start a blog or personal website?",
    "What cybersecurity practices should everyone follow?",
    "How should I manage my passwords securely?",
    "What tech skills are most in demand right now?",
    "How do I evaluate which apps are worth paying for?",
    "What's the best approach to digital minimalism?",
    "How should I back up my important data?",
    "What smart home devices are actually useful?",
    "How can I reduce my social media usage?",
    "What's the best way to learn about blockchain technology?",
    # Life Decisions (20)
    "I need help planning a vacation. Where should I go?",
    "How should I decide between two job offers?",
    "What should I think about before starting a family?",
    "How do I know if I should move to a new city?",
    "What factors matter most when choosing where to live?",
    "How should I approach a major life transition?",
    "What should I consider before adopting a pet?",
    "How do I make better decisions under pressure?",
    "What's the best way to evaluate a big purchase?",
    "How should I plan for the next five years of my life?",
    "What should I think about before quitting my job?",
    "How do I weigh the pros and cons of grad school?",
    "What matters most when choosing a life partner?",
    "How should I approach turning 40?",
    "What legacy do I want to leave behind?",
    "How do I overcome fear of making the wrong choice?",
    "What's the best framework for making tough decisions?",
    "How should I plan my finances before having children?",
    "What should I prioritize in my personal growth?",
    "How do I find meaning and purpose in my daily life?",
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

ambiguous_names = [
    "Alex", "Jordan", "Taylor", "Morgan", "Casey",
    "Riley", "Avery", "Quinn", "Dakota", "Sage",
    "Blair", "Rowan", "Reese", "Cameron", "Hayden",
    "Skyler", "Finley", "Emery", "Peyton", "Addison",
    "Drew", "Jamie", "Jesse", "Pat", "Robin",
]

N_QUESTIONS = len(questions)
N_MALE = len(male_names)
N_FEMALE = len(female_names)
N_AMBIGUOUS = len(ambiguous_names)

male_prompts = [f"Hi, my name is {male_names[i % N_MALE]}. {questions[i]}" for i in range(N_QUESTIONS)]
female_prompts = [f"Hi, my name is {female_names[i % N_FEMALE]}. {questions[i]}" for i in range(N_QUESTIONS)]
ambiguous_prompts = [f"Hi, my name is {ambiguous_names[i % N_AMBIGUOUS]}. {questions[i]}" for i in range(N_AMBIGUOUS)]

all_gendered_prompts = male_prompts + female_prompts
gender_labels = np.array([0] * N_QUESTIONS + [1] * N_QUESTIONS)


def symmetric_kl(logits_m, logits_f):
    p = F.softmax(logits_m.float(), dim=-1).clamp(min=1e-10)
    q = F.softmax(logits_f.float(), dim=-1).clamp(min=1e-10)
    kl_pq = (p * (p.log() - q.log())).sum()
    kl_qp = (q * (q.log() - p.log())).sum()
    return ((kl_pq + kl_qp) / 2).item()


def load_model_and_tokenizer(model_id, loader):
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
    # Conditional Gemma models store layers under text_config
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
    print("  GPU Extraction — Hidden States + Steering")
    print(f"  Model: {MODEL_ID}")
    print(f"  Dataset: {N_QUESTIONS}Q, {N_MALE}M/{N_FEMALE}F/{N_AMBIGUOUS}A names")
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

    for i in range(len(all_gendered_prompts)):
        q_idx = i if i < N_QUESTIONS else i - N_QUESTIONS
        this_text = all_gendered_prompts[i]
        pair_text = female_prompts[q_idx] if i < N_QUESTIONS else male_prompts[q_idx]

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
            label = "male" if i < N_QUESTIONS else "female"
            print(f"  [{label}] name_pos=[{name_start}:{name_end}], decoded='{decoded}'")

    name_lens = [nb[1] - nb[0] for nb in name_boundaries]
    print(f"  Name token lengths: {sorted(set(name_lens))} (unique)")

    # =========================================================
    # EXTRACT HIDDEN STATES — gendered prompts
    # =========================================================
    print(f"\n[3/5] Extracting hidden states ({len(all_gendered_prompts)} gendered prompts)...")

    hs_last_token = {layer: [] for layer in range(total_layers)}
    hs_mean_question = {layer: [] for layer in range(total_layers)}

    for i, text in enumerate(tqdm(all_gendered_prompts, desc="Gendered HS")):
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

    # We need to compute the gender direction on CPU side later,
    # but we need the KL divergences from GPU forward passes.
    # Strategy: compute a quick probe here just to get the direction,
    # then run steering. The CPU script will do full probing independently.

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # Quick probe at each layer to find best layer for steering
    print("  Finding best probe layer (quick scan)...")
    best_acc = 0.0
    best_layer = 0
    for layer in range(total_layers):
        scaler = StandardScaler()
        X_s = scaler.fit_transform(hs_last_token[layer])
        probe = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(probe, X_s, gender_labels, cv=5, scoring="accuracy")
        acc = scores.mean()
        if acc > best_acc:
            best_acc = acc
            best_layer = layer
    print(f"  Best probe layer: {best_layer} ({best_acc:.1%})")

    # Get gender direction at best layer
    scaler_steer = StandardScaler()
    X_steer_s = scaler_steer.fit_transform(hs_last_token[best_layer])
    probe_steer = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
    probe_steer.fit(X_steer_s, gender_labels)
    gender_direction = probe_steer.coef_[0] / scaler_steer.scale_
    gender_direction = gender_direction / np.linalg.norm(gender_direction)
    gender_dir_tensor = torch.tensor(gender_direction, dtype=torch.bfloat16).to("cuda")

    layers_module = get_layers(model)
    embed_module = get_embed(model)

    steering_data = {"steer_layer": best_layer, "best_probe_acc": float(best_acc)}

    steer_q_indices = list(range(0, min(50, N_QUESTIONS)))

    for strength in [0.0, 1.0, 2.0, 5.0, 10.0]:
        kl_cross = []
        kl_same = []

        for i in steer_q_indices:
            male_text = male_prompts[i]
            female_text = female_prompts[i]
            male_text_b = f"Hi, my name is {male_names[(i + 1) % N_MALE]}. {questions[i]}"

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
                handle = target_module.register_forward_hook(make_hook(gender_dir_tensor, strength))

            inputs_m = tok(male_text, return_tensors="pt", truncation=True, max_length=128).to("cuda")
            inputs_f = tok(female_text, return_tensors="pt", truncation=True, max_length=128).to("cuda")
            inputs_mb = tok(male_text_b, return_tensors="pt", truncation=True, max_length=128).to("cuda")

            with torch.no_grad():
                logits_m = model(**inputs_m).logits[0, -1, :]
                logits_f = model(**inputs_f).logits[0, -1, :]
                logits_mb = model(**inputs_mb).logits[0, -1, :]

            kl_cross.append(symmetric_kl(logits_m, logits_f))
            kl_same.append(symmetric_kl(logits_m, logits_mb))

            if strength > 0:
                handle.remove()

        mean_cross = float(np.mean(kl_cross))
        mean_same = float(np.mean(kl_same))
        ratio = mean_cross / mean_same if mean_same > 0 else 0

        steering_data[f"strength_{strength}"] = {
            "cross_gender_kl": mean_cross,
            "same_gender_kl": mean_same,
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
    print("\nSaving .npz files...")

    # Hidden states: save per-layer arrays
    hs_save = {}
    for layer in range(total_layers):
        hs_save[f"last_token_layer_{layer}"] = hs_last_token[layer]
        hs_save[f"question_only_layer_{layer}"] = hs_mean_question[layer]
        hs_save[f"ambiguous_layer_{layer}"] = hs_ambiguous[layer]
    hs_save["gender_labels"] = gender_labels
    hs_save["total_layers"] = np.array([total_layers])
    hs_save["hidden_dim"] = np.array([hidden_dim])
    hs_save["name_boundaries"] = np.array(name_boundaries)

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
    print(f"  DONE — {MODEL_SHORT}")
    print(f"  Time: {elapsed:.1f} min")
    print(f"  Files: {hs_path.name}, {steer_path.name}")
    print(f"{'=' * 70}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
