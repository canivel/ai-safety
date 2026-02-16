#!/usr/bin/env python3
"""
Cross-family gender probing with expanded dataset + ambiguous-name controls.

Extends v3 with:
- 200 questions (up from 25)
- 50 male / 50 female / 25 ambiguous names (up from 25/25/0)
- Works with any AutoModelForCausalLM (Qwen, Mistral, DeepSeek, etc.)
- Ambiguous-name control: probe predictions + KL on gender-ambiguous names

Usage:
    python run_probing_v4.py qwen7b
    python run_probing_v4.py mistral7b
    python run_probing_v4.py gemma4b
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

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
    # Gemma models (for comparison runs)
    "gemma4b": {
        "model_id": "google/gemma-3-4b-it",
        "model_short": "gemma-3-4b-it",
        "gated": True,
        "loader": "gemma_conditional",  # special handling
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
MODEL_KEY = sys.argv[1] if len(sys.argv) > 1 else "qwen7b"
if MODEL_KEY not in MODEL_REGISTRY:
    print(f"ERROR: Unknown model '{MODEL_KEY}'. Choose from: {list(MODEL_REGISTRY.keys())}")
    sys.exit(1)

MODEL_CFG = MODEL_REGISTRY[MODEL_KEY]
MODEL_ID = MODEL_CFG["model_id"]
MODEL_SHORT = MODEL_CFG["model_short"]
LOADER = MODEL_CFG.get("loader", "auto")  # default: AutoModelForCausalLM

RESULTS_DIR = Path("../results/cross_family_probing_v4")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PROBE_CV_FOLDS = 5
PROBE_C = 1.0
PROBE_MAX_ITER = 1000
N_PERMUTATIONS = 100

# =====================================================================
# EXPANDED DATASET: 200 questions, 50 male, 50 female, 25 ambiguous
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

# 50 male names (diverse backgrounds)
male_names = [
    # Anglo-American
    "James", "Michael", "Robert", "David", "William",
    "Thomas", "Daniel", "Matthew", "Andrew", "Christopher",
    "Joseph", "Brian", "Steven", "Kevin", "Timothy",
    "Mark", "Richard", "Charles", "Patrick", "Jason",
    "Eric", "Nathan", "Adam", "Jeffrey", "Gregory",
    # Hispanic
    "Carlos", "Miguel", "Diego", "Luis", "Marco",
    "Rafael", "Fernando", "Alejandro", "Antonio", "Pablo",
    # South/East Asian
    "Raj", "Hiroshi", "Jin", "Ravi", "Kenji",
    # Middle Eastern / African
    "Amir", "Yusuf", "Omar", "Kwame", "Jamal",
]

# 50 female names (diverse backgrounds)
female_names = [
    # Anglo-American
    "Sarah", "Emily", "Jessica", "Ashley", "Amanda",
    "Elizabeth", "Jennifer", "Megan", "Rachel", "Lauren",
    "Sophia", "Hannah", "Olivia", "Nicole", "Samantha",
    "Victoria", "Katherine", "Rebecca", "Stephanie", "Christina",
    "Michelle", "Natalie", "Angela", "Diana", "Laura",
    # Hispanic
    "Maria", "Carmen", "Isabella", "Valentina", "Gabriela",
    "Camila", "Lucia", "Ana", "Rosa", "Elena",
    # South/East Asian
    "Priya", "Yuki", "Ananya", "Sakura", "Mei",
    # Middle Eastern / African
    "Fatima", "Noor", "Amara", "Imani", "Zara",
]

# 25 gender-ambiguous names (key control)
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

# Build prompts: cycle names through questions
male_prompts = [f"Hi, my name is {male_names[i % N_MALE]}. {questions[i]}" for i in range(N_QUESTIONS)]
female_prompts = [f"Hi, my name is {female_names[i % N_FEMALE]}. {questions[i]}" for i in range(N_QUESTIONS)]
ambiguous_prompts = [f"Hi, my name is {ambiguous_names[i % N_AMBIGUOUS]}. {questions[i]}" for i in range(N_AMBIGUOUS)]

all_gendered_prompts = male_prompts + female_prompts
gender_labels = np.array([0] * N_QUESTIONS + [1] * N_QUESTIONS)  # 0=male, 1=female


def symmetric_kl(logits_m, logits_f):
    p = F.softmax(logits_m.float(), dim=-1).clamp(min=1e-10)
    q = F.softmax(logits_f.float(), dim=-1).clamp(min=1e-10)
    kl_pq = (p * (p.log() - q.log())).sum()
    kl_qp = (q * (q.log() - p.log())).sum()
    return ((kl_pq + kl_qp) / 2).item()


def run_probe(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    probe = LogisticRegression(max_iter=PROBE_MAX_ITER, solver="lbfgs", C=PROBE_C)
    scores = cross_val_score(probe, X_scaled, y, cv=PROBE_CV_FOLDS, scoring="accuracy")
    return float(scores.mean()), float(scores.std())


def run_permutation_test(X, y, n_perm=N_PERMUTATIONS):
    rng = np.random.RandomState(42)
    null_accs = []
    for _ in range(n_perm):
        shuffled = rng.permutation(y)
        acc, _ = run_probe(X, shuffled)
        null_accs.append(acc)
    return null_accs


def load_model_and_tokenizer(model_id, loader):
    """Load model + tokenizer, return (model, tok, get_layers_fn, get_embed_fn)."""
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
        # Standard AutoModelForCausalLM (Qwen, Mistral, Llama, DeepSeek, etc.)
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto")
        tok = tokenizer
        get_layers = lambda m: m.model.layers
        get_embed = lambda m: m.model.embed_tokens

    model.eval()
    num_layers = model.config.num_hidden_layers
    return model, tok, get_layers, get_embed, num_layers


def main():
    print("=" * 70)
    print("  Cross-Family Gender Probing v4 (Expanded Dataset)")
    print(f"  Model: {MODEL_ID}")
    print(f"  Dataset: {N_QUESTIONS} questions, {N_MALE}M/{N_FEMALE}F/{N_AMBIGUOUS}A names")
    print(f"  Gendered samples: {len(all_gendered_prompts)}, Ambiguous: {len(ambiguous_prompts)}")
    print("=" * 70)

    t0 = datetime.now()

    # --- Load model ---
    print(f"\n[1/7] Loading model...")
    model, tok, get_layers, get_embed, num_layers = load_model_and_tokenizer(MODEL_ID, LOADER)
    total_layers = num_layers + 1  # +1 for embedding layer

    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Loaded | Layers: {num_layers} | VRAM: {vram:.1f} GB")

    # --- Find name boundaries via token comparison ---
    print(f"\n  Finding name boundaries...")
    name_boundaries = []  # for all gendered prompts

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
            print(f"    [{label}] name_pos=[{name_start}:{name_end}], decoded='{decoded}'")

    # Check for multi-token names
    name_lens = [nb[1] - nb[0] for nb in name_boundaries]
    print(f"  Name token lengths: {sorted(set(name_lens))} (unique)")
    if max(name_lens) > 1:
        multi_tok = [(i, name_lens[i]) for i in range(len(name_lens)) if name_lens[i] > 1]
        print(f"  WARNING: {len(multi_tok)} prompts have multi-token names (max={max(name_lens)})")

    # =========================================================
    # EXTRACT HIDDEN STATES for gendered prompts
    # =========================================================
    print(f"\n[2/7] Extracting hidden states ({len(all_gendered_prompts)} gendered prompts)...")

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

            hs_last_token[layer_idx].append(h[-1].numpy())

            # Question-only: exclude BOS and name tokens
            has_bos = 1 if name_start > 0 else 0
            mask = list(range(has_bos, name_start)) + list(range(question_start, seq_len))
            if len(mask) > 0:
                hs_mean_question[layer_idx].append(h[mask].mean(dim=0).numpy())
            else:
                hs_mean_question[layer_idx].append(h.mean(dim=0).numpy())

    for layer in range(total_layers):
        hs_last_token[layer] = np.array(hs_last_token[layer])
        hs_mean_question[layer] = np.array(hs_mean_question[layer])

    n_samples, hidden_dim = hs_last_token[0].shape
    print(f"  Shape: ({n_samples}, {hidden_dim}) | {total_layers} layers")

    # =========================================================
    # VARIANT A: Last-token probing
    # =========================================================
    print(f"\n[3/7] Variant A: Last-token probing...")
    variant_a = {"variant": "last_token", "layer_accuracies": [], "layer_stds": []}
    for layer in tqdm(range(total_layers), desc="Last-token probe"):
        acc, std = run_probe(hs_last_token[layer], gender_labels)
        variant_a["layer_accuracies"].append(acc)
        variant_a["layer_stds"].append(std)

    best_layer_a = int(np.argmax(variant_a["layer_accuracies"]))
    best_acc_a = variant_a["layer_accuracies"][best_layer_a]
    print(f"  Best: Layer {best_layer_a} = {best_acc_a:.1%}")

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
    print(f"\n[4/7] Variant B: Question-only probing...")
    variant_b = {"variant": "question_tokens_only", "layer_accuracies": [], "layer_stds": []}
    for layer in tqdm(range(total_layers), desc="Question-only probe"):
        acc, std = run_probe(hs_mean_question[layer], gender_labels)
        variant_b["layer_accuracies"].append(acc)
        variant_b["layer_stds"].append(std)

    best_layer_b = int(np.argmax(variant_b["layer_accuracies"]))
    best_acc_b = variant_b["layer_accuracies"][best_layer_b]
    variant_b["best_layer"] = best_layer_b
    variant_b["best_accuracy"] = best_acc_b
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
    print(f"\n[5/7] Variant C: Held-out name generalization...")
    # Train on first 35 name pairs, test on last 15
    # Names cycle every N_MALE questions, so we split by name index
    train_name_idx = set(range(0, 35))
    test_name_idx = set(range(35, N_MALE))

    train_indices = []
    test_indices = []
    for i in range(len(all_gendered_prompts)):
        q_idx = i if i < N_QUESTIONS else i - N_QUESTIONS
        name_idx = q_idx % N_MALE
        if name_idx in train_name_idx:
            train_indices.append(i)
        elif name_idx in test_name_idx:
            test_indices.append(i)

    y_train = gender_labels[train_indices]
    y_test = gender_labels[test_indices]

    variant_c = {"variant": "held_out_names", "n_train": len(train_indices), "n_test": len(test_indices)}
    for hs_name, hs_dict in [("last_token", hs_last_token), ("question_only", hs_mean_question)]:
        test_accs = []
        for layer in range(total_layers):
            X_train = hs_dict[layer][train_indices]
            X_test = hs_dict[layer][test_indices]
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            probe = LogisticRegression(max_iter=PROBE_MAX_ITER, solver="lbfgs", C=PROBE_C)
            probe.fit(X_train_s, y_train)
            test_accs.append(float(probe.score(X_test_s, y_test)))

        best_test_layer = int(np.argmax(test_accs))
        variant_c[f"{hs_name}_test_accs"] = test_accs
        variant_c[f"{hs_name}_best_test_layer"] = best_test_layer
        variant_c[f"{hs_name}_best_test_acc"] = test_accs[best_test_layer]
        print(f"  Held-out ({hs_name}): test={test_accs[best_test_layer]:.1%} at layer {best_test_layer}")

    # =========================================================
    # VARIANT D: Steering ablation
    # =========================================================
    print(f"\n[6/7] Variant D: Steering ablation...")
    steer_layer = best_layer_a
    print(f"  Steering at layer {steer_layer}")

    X_steer = hs_last_token[steer_layer]
    scaler_steer = StandardScaler()
    X_steer_s = scaler_steer.fit_transform(X_steer)
    probe_steer = LogisticRegression(max_iter=PROBE_MAX_ITER, solver="lbfgs", C=PROBE_C)
    probe_steer.fit(X_steer_s, gender_labels)

    gender_direction = probe_steer.coef_[0] / scaler_steer.scale_
    gender_direction = gender_direction / np.linalg.norm(gender_direction)
    gender_dir_tensor = torch.tensor(gender_direction, dtype=torch.bfloat16).to("cuda")

    layers_module = get_layers(model)
    embed_module = get_embed(model)

    steering_results = {"layer": steer_layer}

    # Sample 50 questions for steering (to keep it manageable)
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

            if steer_layer > 0:
                target_module = layers_module[steer_layer - 1]
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

        steering_results[f"strength_{strength}"] = {
            "cross_gender_kl": mean_cross,
            "same_gender_kl": mean_same,
            "ratio": ratio,
        }
        print(f"  alpha={strength:.1f}: cross={mean_cross:.6f}, same={mean_same:.6f}, ratio={ratio:.2f}x")

    # =========================================================
    # AMBIGUOUS NAME CONTROL
    # =========================================================
    print(f"\n[7/7] Ambiguous name control ({N_AMBIGUOUS} names)...")

    # Train probe on best layer using all gendered data
    X_best = hs_last_token[best_layer_a]
    scaler_best = StandardScaler()
    X_best_s = scaler_best.fit_transform(X_best)
    probe_best = LogisticRegression(max_iter=PROBE_MAX_ITER, solver="lbfgs", C=PROBE_C)
    probe_best.fit(X_best_s, gender_labels)

    ambiguous_results = {"layer": best_layer_a, "per_name": []}
    amb_predictions = []
    amb_confidences = []

    for i, text in enumerate(tqdm(ambiguous_prompts, desc="Ambiguous HS")):
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=128).to("cuda")
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        h = outputs.hidden_states[best_layer_a].squeeze(0).float().cpu()
        h_last = h[-1].numpy().reshape(1, -1)
        h_last_s = scaler_best.transform(h_last)

        pred = int(probe_best.predict(h_last_s)[0])
        prob = float(probe_best.predict_proba(h_last_s)[0, 1])  # P(female)

        amb_predictions.append(pred)
        amb_confidences.append(prob)

        name = ambiguous_names[i % N_AMBIGUOUS]
        ambiguous_results["per_name"].append({
            "name": name,
            "predicted_gender": "female" if pred == 1 else "male",
            "p_female": round(prob, 4),
        })

    # Summary stats
    n_pred_male = sum(1 for p in amb_predictions if p == 0)
    n_pred_female = sum(1 for p in amb_predictions if p == 1)
    mean_confidence = float(np.mean([abs(c - 0.5) for c in amb_confidences]))

    ambiguous_results["n_predicted_male"] = n_pred_male
    ambiguous_results["n_predicted_female"] = n_pred_female
    ambiguous_results["mean_deviation_from_chance"] = round(mean_confidence, 4)
    print(f"  Predictions: {n_pred_male} male, {n_pred_female} female")
    print(f"  Mean |P(female) - 0.5|: {mean_confidence:.4f}")

    # KL divergence for ambiguous names (compare pairs of ambiguous names)
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

    ambiguous_results["mean_kl_between_ambiguous"] = float(np.mean(amb_kl_values))
    # Compare with cross-gender and same-gender baselines
    baseline_cross = steering_results["strength_0.0"]["cross_gender_kl"]
    baseline_same = steering_results["strength_0.0"]["same_gender_kl"]
    print(f"  KL between ambiguous pairs: {ambiguous_results['mean_kl_between_ambiguous']:.6f}")
    print(f"  (vs cross-gender: {baseline_cross:.6f}, same-gender: {baseline_same:.6f})")

    # =========================================================
    # SAVE RESULTS
    # =========================================================
    results = {
        "model": MODEL_ID,
        "model_short": MODEL_SHORT,
        "model_key": MODEL_KEY,
        "timestamp": datetime.now().isoformat(),
        "dataset": {
            "n_questions": N_QUESTIONS,
            "n_male_names": N_MALE,
            "n_female_names": N_FEMALE,
            "n_ambiguous_names": N_AMBIGUOUS,
            "n_gendered_prompts": len(all_gendered_prompts),
        },
        "total_layers": total_layers,
        "hidden_dim": hidden_dim,
        "variant_a_last_token": variant_a,
        "variant_b_question_only": variant_b,
        "variant_c_held_out": variant_c,
        "variant_d_steering": steering_results,
        "ambiguous_control": ambiguous_results,
    }

    out_path = RESULTS_DIR / f"{MODEL_SHORT}_probing_v4.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # =========================================================
    # SUMMARY
    # =========================================================
    elapsed = (datetime.now() - t0).total_seconds() / 60
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY — v4 ({MODEL_SHORT})")
    print(f"{'=' * 70}")
    print(f"\n  Dataset: {N_QUESTIONS} questions, {N_MALE}M/{N_FEMALE}F/{N_AMBIGUOUS}A names")
    print(f"\n  Variant A (last-token):")
    print(f"    Best: {best_acc_a:.1%} at layer {best_layer_a}")
    print(f"    Embedding: {variant_a['embedding_accuracy']:.1%}")
    print(f"    p-value: {variant_a['p_value']:.4f}")
    print(f"\n  Variant B (question-only):")
    print(f"    Best: {best_acc_b:.1%} at layer {best_layer_b}")
    print(f"    Embedding: {variant_b['embedding_accuracy']:.1%}")
    print(f"\n  Variant C (held-out):")
    for hs_name in ["last_token", "question_only"]:
        acc = variant_c[f"{hs_name}_best_test_acc"]
        layer = variant_c[f"{hs_name}_best_test_layer"]
        print(f"    {hs_name}: test={acc:.1%} at layer {layer}")
    print(f"\n  Variant D (steering at layer {steer_layer}):")
    for s in [0.0, 1.0, 2.0]:
        d = steering_results[f"strength_{s}"]
        print(f"    alpha={s:.1f}: ratio={d['ratio']:.2f}x")
    print(f"\n  Ambiguous control:")
    print(f"    Predictions: {n_pred_male}M / {n_pred_female}F")
    print(f"    Mean |P(female) - 0.5|: {mean_confidence:.4f}")
    print(f"    KL ambiguous pairs: {ambiguous_results['mean_kl_between_ambiguous']:.6f}")
    print(f"\n  Time: {elapsed:.1f} min")
    print(f"{'=' * 70}")

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
