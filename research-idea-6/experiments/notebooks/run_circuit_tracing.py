#!/usr/bin/env python3
"""
Experiment 2: Attention Head Circuit Tracing

Which attention heads propagate gender information from name tokens to
question tokens?

Method:
  Phase 1 — Attention pattern extraction:
    1. Load model with eager attention (required for attention weight access)
    2. Run 100 prompts (50M + 50F) with output_attentions=True
    3. For each head, compute mean attention from question tokens → name tokens
    4. Rank heads by attention to name tokens and by gender-differential attention

  Phase 2 — Causal ablation verification:
    1. Zero out top-N name-attending heads via forward hooks
    2. Re-extract hidden states for all 400 prompts
    3. Re-run probing classifier
    4. If probing accuracy drops → those heads causally propagate gender info

Usage:
    python run_circuit_tracing.py gemma4b
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

# Parse CLI
if len(sys.argv) < 2:
    print(f"Usage: python run_circuit_tracing.py <model_key>")
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
BEST_PROBE_LAYER = MODEL_CFG["best_probe_layer"]
# NUM_HEADS and HEAD_DIM will be auto-detected after model loading

RESULTS_DIR = Path("../results/circuit_tracing")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================================
# DATASET (same as v4 probing)
# =====================================================================

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

N_QUESTIONS = len(questions)
N_MALE = len(male_names)
N_FEMALE = len(female_names)

male_prompts = [f"Hi, my name is {male_names[i % N_MALE]}. {questions[i]}" for i in range(N_QUESTIONS)]
female_prompts = [f"Hi, my name is {female_names[i % N_FEMALE]}. {questions[i]}" for i in range(N_QUESTIONS)]

all_gendered_prompts = male_prompts + female_prompts
gender_labels = np.array([0] * N_QUESTIONS + [1] * N_QUESTIONS)


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def load_model_and_tokenizer(model_id, loader, eager_attention=False):
    """Load model with optional eager attention for attention weight extraction."""
    from huggingface_hub import login
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    extra_kwargs = {}
    if eager_attention:
        extra_kwargs["attn_implementation"] = "eager"

    if loader == "gemma_causal":
        from transformers import AutoTokenizer, Gemma3ForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = Gemma3ForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto",
            **extra_kwargs)
        tok = tokenizer
        get_layers = lambda m: m.model.layers
        get_embed = lambda m: m.model.embed_tokens
    elif loader == "gemma_conditional":
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration
        tokenizer = AutoProcessor.from_pretrained(model_id)
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto",
            **extra_kwargs)
        tok = tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else tokenizer
        get_layers = lambda m: m.model.language_model.layers
        get_embed = lambda m: m.model.language_model.embed_tokens
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto",
            **extra_kwargs)
        tok = tokenizer
        get_layers = lambda m: m.model.layers
        get_embed = lambda m: m.model.embed_tokens

    model.eval()

    if hasattr(model.config, 'num_hidden_layers'):
        num_layers = model.config.num_hidden_layers
    elif hasattr(model.config, 'text_config') and hasattr(model.config.text_config, 'num_hidden_layers'):
        num_layers = model.config.text_config.num_hidden_layers
    else:
        raise ValueError(f"Cannot find num_hidden_layers in config: {model.config}")

    # Auto-detect attention head dimensions from config and o_proj weight
    cfg = model.config.text_config if hasattr(model.config, 'text_config') else model.config
    num_attn_heads = cfg.num_attention_heads
    head_dim = getattr(cfg, 'head_dim', None)
    # Verify against o_proj weight shape
    layer0 = get_layers(model)[0]
    o_proj_in = layer0.self_attn.o_proj.in_features
    if head_dim is None:
        head_dim = o_proj_in // num_attn_heads
    # Double-check: num_heads * head_dim must equal o_proj input
    if num_attn_heads * head_dim != o_proj_in:
        print(f"  WARNING: {num_attn_heads} heads * {head_dim} dim = {num_attn_heads * head_dim} "
              f"!= o_proj in_features {o_proj_in}")
        # Fall back to deriving from o_proj
        head_dim = o_proj_in // num_attn_heads
        print(f"  Using derived head_dim={head_dim}")

    return model, tok, get_layers, get_embed, num_layers, num_attn_heads, head_dim


def find_name_boundaries(tok, all_prompts, male_prompts, female_prompts, n_questions):
    """Find name token positions by comparing male/female token sequences."""
    boundaries = []
    for i in range(len(all_prompts)):
        q_idx = i if i < n_questions else i - n_questions
        this_text = all_prompts[i]
        pair_text = female_prompts[q_idx] if i < n_questions else male_prompts[q_idx]

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
        boundaries.append((name_start, name_end))

    return boundaries


def make_head_ablation_hook(target_heads, num_heads, head_dim):
    """
    Create a forward_pre_hook on o_proj that zeros specific attention heads.

    The input to o_proj is (batch, seq, num_heads * head_dim).
    We reshape, zero target heads, reshape back.
    """
    def hook_fn(module, input):
        x = input[0]  # (batch, seq, hidden_dim)
        batch, seq, hidden = x.shape
        x = x.view(batch, seq, num_heads, head_dim)
        for head_idx in target_heads:
            x[:, :, head_idx, :] = 0.0
        return (x.view(batch, seq, hidden),) + input[1:]
    return hook_fn


# =====================================================================
# MAIN
# =====================================================================

def main():
    t0 = datetime.now()
    print("=" * 70)
    print("  Experiment 2: Attention Head Circuit Tracing")
    print(f"  Model: {MODEL_ID}")
    print(f"  Best probe layer: {BEST_PROBE_LAYER}")
    print("=" * 70)

    # --- Step 1: Load model with eager attention ---
    print(f"\n[1/6] Loading model with eager attention...")
    model, tok, get_layers, get_embed, num_layers, NUM_HEADS, HEAD_DIM = \
        load_model_and_tokenizer(MODEL_ID, LOADER, eager_attention=True)

    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Loaded | Layers: {num_layers} | VRAM: {vram:.1f} GB")
    print(f"  Attention: {NUM_HEADS} heads × {HEAD_DIM} dim (o_proj input: {NUM_HEADS * HEAD_DIM})")

    # --- Step 2: Find name boundaries ---
    print(f"\n[2/6] Finding name boundaries...")
    name_boundaries = find_name_boundaries(
        tok, all_gendered_prompts, male_prompts, female_prompts, N_QUESTIONS)
    name_lens = [nb[1] - nb[0] for nb in name_boundaries]
    print(f"  Name token lengths: {sorted(set(name_lens))} unique values")

    # --- Step 3: Extract attention patterns ---
    N_ATTN_SUBSET = 50  # Questions for attention analysis
    n_prompts = N_ATTN_SUBSET * 2  # male + female
    print(f"\n[3/6] Extracting attention patterns ({n_prompts} prompts)...")

    # Store per-head attention scores: (prompt, layer, head)
    # Score = mean attention from question tokens to name tokens
    attn_to_name = np.zeros((n_prompts, num_layers, NUM_HEADS))
    attn_from_last = np.zeros((n_prompts, num_layers, NUM_HEADS))

    # Select subset: first N_ATTN_SUBSET male + first N_ATTN_SUBSET female
    subset_indices = list(range(N_ATTN_SUBSET)) + \
                     list(range(N_QUESTIONS, N_QUESTIONS + N_ATTN_SUBSET))

    for p_idx, prompt_idx in enumerate(tqdm(subset_indices, desc="Attention")):
        text = all_gendered_prompts[prompt_idx]
        name_start, name_end = name_boundaries[prompt_idx]

        inputs = tok(text, return_tensors="pt", truncation=True, max_length=128).to("cuda")
        seq_len = inputs["input_ids"].shape[1]
        question_start = min(name_end, seq_len)

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        # outputs.attentions: tuple of (1, num_heads, seq_len, seq_len) per layer
        attentions = outputs.attentions
        if attentions is None:
            print(f"  WARNING: No attention weights returned for prompt {prompt_idx}")
            continue

        for layer_idx in range(min(len(attentions), num_layers)):
            attn = attentions[layer_idx][0].float().cpu()  # (num_heads, seq, seq)
            n_heads_actual = attn.shape[0]

            for h in range(min(n_heads_actual, NUM_HEADS)):
                # Question tokens attending to name tokens
                if question_start < seq_len and name_start < name_end:
                    attn_qn = attn[h, question_start:, name_start:name_end]
                    attn_to_name[p_idx, layer_idx, h] = attn_qn.mean().item()

                # Last token attending to name tokens
                if name_start < name_end:
                    attn_ln = attn[h, -1, name_start:name_end]
                    attn_from_last[p_idx, layer_idx, h] = attn_ln.mean().item()

        # Free attention memory
        del outputs, attentions
        torch.cuda.empty_cache()

    # Split into male/female
    male_attn_name = attn_to_name[:N_ATTN_SUBSET]  # (50, layers, heads)
    female_attn_name = attn_to_name[N_ATTN_SUBSET:]

    male_attn_last = attn_from_last[:N_ATTN_SUBSET]
    female_attn_last = attn_from_last[N_ATTN_SUBSET:]

    # --- Step 4: Rank heads ---
    print(f"\n[4/6] Ranking heads by name attention and gender differential...")

    # Mean attention to name (across all prompts)
    mean_attn = attn_to_name.mean(axis=0)  # (layers, heads)
    mean_attn_last_token = attn_from_last.mean(axis=0)

    # Gender differential: |mean_male - mean_female|
    gender_diff = np.abs(male_attn_name.mean(axis=0) - female_attn_name.mean(axis=0))
    gender_diff_last = np.abs(male_attn_last.mean(axis=0) - female_attn_last.mean(axis=0))

    # Flatten and rank
    head_scores = []
    for layer in range(num_layers):
        for head in range(NUM_HEADS):
            head_scores.append({
                "layer": layer,
                "head": head,
                "mean_attn_to_name": float(mean_attn[layer, head]),
                "mean_attn_last_to_name": float(mean_attn_last_token[layer, head]),
                "gender_diff_question": float(gender_diff[layer, head]),
                "gender_diff_last": float(gender_diff_last[layer, head]),
            })

    # Sort by attention from question→name (most attentive first)
    by_attn = sorted(head_scores, key=lambda x: x["mean_attn_to_name"], reverse=True)
    print(f"\n  Top 10 heads by question→name attention:")
    for i, h in enumerate(by_attn[:10]):
        print(f"    {i+1}. Layer {h['layer']:2d} Head {h['head']:2d}: "
              f"attn={h['mean_attn_to_name']:.4f} "
              f"diff={h['gender_diff_question']:.4f}")

    # Sort by gender differential
    by_diff = sorted(head_scores, key=lambda x: x["gender_diff_question"], reverse=True)
    print(f"\n  Top 10 heads by gender-differential attention:")
    for i, h in enumerate(by_diff[:10]):
        print(f"    {i+1}. Layer {h['layer']:2d} Head {h['head']:2d}: "
              f"diff={h['gender_diff_question']:.4f} "
              f"attn={h['mean_attn_to_name']:.4f}")

    # Sort by last-token attention to name
    by_last = sorted(head_scores, key=lambda x: x["mean_attn_last_to_name"], reverse=True)
    print(f"\n  Top 10 heads by last-token→name attention:")
    for i, h in enumerate(by_last[:10]):
        print(f"    {i+1}. Layer {h['layer']:2d} Head {h['head']:2d}: "
              f"last_attn={h['mean_attn_last_to_name']:.4f}")

    # --- Step 5: Ablation verification ---
    print(f"\n[5/6] Ablation verification (re-probing after zeroing top heads)...")

    layers_module = get_layers(model)

    # Baseline: extract hidden states at best probe layer (no ablation)
    print(f"  Extracting baseline hidden states at layer {BEST_PROBE_LAYER}...")
    captured = {}
    def capture_hook(module, input, output):
        if isinstance(output, tuple):
            captured['hs'] = output[0].detach().cpu()
        else:
            captured['hs'] = output.detach().cpu()

    target_layer_module = layers_module[BEST_PROBE_LAYER - 1] if BEST_PROBE_LAYER > 0 \
        else get_embed(model)

    handle_capture = target_layer_module.register_forward_hook(capture_hook)

    hs_baseline = []
    for text in tqdm(all_gendered_prompts, desc="Baseline HS"):
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=128).to("cuda")
        with torch.no_grad():
            model(**inputs)
        hs = captured['hs'].squeeze(0).float()
        hs_baseline.append(hs[-1].numpy())

    handle_capture.remove()
    hs_baseline = np.array(hs_baseline)

    # Train baseline probe
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score

    scaler_base = StandardScaler()
    X_base = scaler_base.fit_transform(hs_baseline)
    probe_base = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
    scores_base = cross_val_score(probe_base, X_base, gender_labels, cv=5, scoring="accuracy")
    baseline_acc = scores_base.mean()
    print(f"  Baseline probe accuracy: {baseline_acc:.1%}")

    # Ablation experiments: zero top-N heads and re-probe
    ablation_results = []
    top_heads_by_attn = [(h["layer"], h["head"]) for h in by_attn]

    for n_ablate in [1, 3, 5, 10, 20]:
        heads_to_ablate = top_heads_by_attn[:n_ablate]
        print(f"\n  Ablating top {n_ablate} heads...")

        # Group heads by layer for efficient hooking
        heads_by_layer = {}
        for layer, head in heads_to_ablate:
            if layer not in heads_by_layer:
                heads_by_layer[layer] = []
            heads_by_layer[layer].append(head)

        # Register hooks on o_proj of affected layers
        ablation_handles = []
        for layer_idx, head_list in heads_by_layer.items():
            self_attn = layers_module[layer_idx].self_attn
            hook = make_head_ablation_hook(head_list, NUM_HEADS, HEAD_DIM)
            h = self_attn.o_proj.register_forward_pre_hook(hook)
            ablation_handles.append(h)

        # Re-capture with probe layer hook
        handle_capture = target_layer_module.register_forward_hook(capture_hook)

        hs_ablated = []
        for text in tqdm(all_gendered_prompts, desc=f"Ablated HS (top-{n_ablate})"):
            inputs = tok(text, return_tensors="pt", truncation=True, max_length=128).to("cuda")
            with torch.no_grad():
                model(**inputs)
            hs = captured['hs'].squeeze(0).float()
            hs_ablated.append(hs[-1].numpy())

        handle_capture.remove()
        for h in ablation_handles:
            h.remove()

        hs_ablated = np.array(hs_ablated)

        # Re-probe
        scaler_abl = StandardScaler()
        X_abl = scaler_abl.fit_transform(hs_ablated)
        probe_abl = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
        scores_abl = cross_val_score(probe_abl, X_abl, gender_labels, cv=5, scoring="accuracy")
        ablated_acc = scores_abl.mean()

        acc_drop = baseline_acc - ablated_acc
        ablation_results.append({
            "n_heads_ablated": n_ablate,
            "heads": [{"layer": l, "head": h} for l, h in heads_to_ablate],
            "accuracy": float(ablated_acc),
            "accuracy_drop": float(acc_drop),
            "accuracy_drop_pct": float(acc_drop / baseline_acc * 100) if baseline_acc > 0 else 0,
        })
        print(f"  Top-{n_ablate}: acc={ablated_acc:.1%} "
              f"(Δ={acc_drop:+.1%}, {acc_drop/baseline_acc*100:+.1f}% relative)")

    # --- Step 6: Save results ---
    print(f"\n[6/6] Saving results...")

    results = {
        "model": MODEL_SHORT,
        "model_id": MODEL_ID,
        "best_probe_layer": BEST_PROBE_LAYER,
        "num_layers": num_layers,
        "num_heads": NUM_HEADS,
        "head_dim": HEAD_DIM,
        "n_attention_prompts": n_prompts,
        "baseline_probe_accuracy": float(baseline_acc),
        # Top heads rankings
        "top_heads_by_question_to_name_attn": [
            {k: v for k, v in h.items()} for h in by_attn[:20]
        ],
        "top_heads_by_gender_differential": [
            {k: v for k, v in h.items()} for h in by_diff[:20]
        ],
        "top_heads_by_last_token_attn": [
            {k: v for k, v in h.items()} for h in by_last[:20]
        ],
        # Ablation results
        "ablation_results": ablation_results,
        # Full attention data (layer × head means)
        "mean_attn_to_name_per_head": mean_attn.tolist(),
        "gender_diff_per_head": gender_diff.tolist(),
        "mean_last_token_attn_per_head": mean_attn_last_token.tolist(),
    }

    path = RESULTS_DIR / f"{MODEL_SHORT}_circuit_tracing.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {path}")

    elapsed = (datetime.now() - t0).total_seconds() / 60
    print(f"\n{'=' * 70}")
    print(f"  RESULTS — {MODEL_SHORT}")
    print(f"{'=' * 70}")
    print(f"  Baseline probe: {baseline_acc:.1%}")
    for r in ablation_results:
        print(f"  Top-{r['n_heads_ablated']:2d} ablated: "
              f"{r['accuracy']:.1%} ({r['accuracy_drop_pct']:+.1f}%)")
    print(f"\n  Time: {elapsed:.1f} min")
    print(f"  VRAM peak: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    print(f"{'=' * 70}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
