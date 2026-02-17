#!/usr/bin/env python3
"""
Experiment 1: Causal Mediation on Full Responses

Does ablating the gender direction causally reduce how differently the model
processes male vs female prompts — measured via KL divergence?

Method:
  1. Generate a response R using the male prompt (normal condition)
  2. Teacher-force [male_prompt + R] and [female_prompt + R] through the model
  3. Compute per-token symmetric KL divergence at each response position
  4. Repeat under gender-ablated and random-ablated conditions
  5. If ablation reduces KL → causal evidence that the gender direction
     controls response-level divergence, not just first-token logits

Usage:
    python run_causal_mediation.py gemma4b
    python run_causal_mediation.py mistral7b
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

# Force unbuffered output (critical for RunPod SSH monitoring)
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
    "gemma12b": {
        "model_id": "google/gemma-3-12b-it",
        "model_short": "gemma-3-12b-it",
        "gated": True,
        "loader": "gemma_conditional",
        "best_probe_layer": 25,
    },
}

# Parse CLI
if len(sys.argv) < 2:
    print(f"Usage: python run_causal_mediation.py <model_key>")
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

RESULTS_DIR = Path("../results/causal_mediation")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================================
# DATASET (same as v4 probing)
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

def symmetric_kl(logits_a, logits_b):
    """Symmetric KL divergence between two logit vectors."""
    p = F.softmax(logits_a.float(), dim=-1).clamp(min=1e-10)
    q = F.softmax(logits_b.float(), dim=-1).clamp(min=1e-10)
    kl_pq = (p * (p.log() - q.log())).sum()
    kl_qp = (q * (q.log() - p.log())).sum()
    return ((kl_pq + kl_qp) / 2).item()


def jaccard_similarity(text_a, text_b):
    """Word-level Jaccard similarity between two texts."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def make_hook(direction, strength):
    """Create a forward hook that ablates a direction from hidden states."""
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
    """Load model and tokenizer, returning accessors for layers and embeddings."""
    from huggingface_hub import login
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    processor = None

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
        processor = AutoProcessor.from_pretrained(model_id)
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto")
        tok = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
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

    if hasattr(model.config, 'num_hidden_layers'):
        num_layers = model.config.num_hidden_layers
    elif hasattr(model.config, 'text_config') and hasattr(model.config.text_config, 'num_hidden_layers'):
        num_layers = model.config.text_config.num_hidden_layers
    else:
        raise ValueError(f"Cannot find num_hidden_layers in config: {model.config}")

    # Ensure pad_token is set for generation
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    return model, tok, processor, get_layers, get_embed, num_layers


def generate_response(model, tok, processor, text, loader, max_new_tokens=256):
    """Generate a response and return (text, response_token_ids)."""
    messages = [{"role": "user", "content": text}]

    if loader == "gemma_conditional" and processor is not None:
        # Use processor for multimodal Gemma models
        mm_messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        try:
            inputs = processor.apply_chat_template(
                mm_messages, return_tensors="pt",
                add_generation_prompt=True, return_dict=True, tokenize=True
            )
            inputs = {k: v.to("cuda") for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            input_len = inputs["input_ids"].shape[1]
        except Exception:
            # Fallback to inner tokenizer
            input_ids = tok.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            )
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor([input_ids])
            inputs = {"input_ids": input_ids.to("cuda")}
            input_len = inputs["input_ids"].shape[1]
    else:
        input_ids = tok.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor([input_ids])
        inputs = {"input_ids": input_ids.to("cuda")}
        input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )

    response_ids = output_ids[0, input_len:].cpu()
    response_text = tok.decode(response_ids, skip_special_tokens=True).strip()
    return response_text, response_ids


def compute_response_kl(model, tok, male_text, female_text, response_ids):
    """
    Teacher-forcing KL: compute per-token symmetric KL divergence between
    the model's predictions when processing [male_prompt + R] vs [female_prompt + R].

    The response tokens R are the same in both cases — only the name differs.
    """
    # Tokenize prompts
    prompt_ids_m = tok(male_text, return_tensors="pt", truncation=True, max_length=128)["input_ids"]
    prompt_ids_f = tok(female_text, return_tensors="pt", truncation=True, max_length=128)["input_ids"]

    # Concatenate prompt + response
    full_ids_m = torch.cat([prompt_ids_m[0], response_ids]).unsqueeze(0).to("cuda")
    full_ids_f = torch.cat([prompt_ids_f[0], response_ids]).unsqueeze(0).to("cuda")

    prompt_len_m = prompt_ids_m.shape[1]
    prompt_len_f = prompt_ids_f.shape[1]
    resp_len = len(response_ids)

    with torch.no_grad():
        logits_m = model(input_ids=full_ids_m).logits[0]  # (total_len, vocab)
        logits_f = model(input_ids=full_ids_f).logits[0]

    # KL at each response position
    # logits[prompt_len - 1 + t] predicts response token t
    kl_values = []
    for t in range(resp_len):
        pos_m = prompt_len_m - 1 + t
        pos_f = prompt_len_f - 1 + t
        if pos_m < logits_m.shape[0] and pos_f < logits_f.shape[0]:
            kl = symmetric_kl(logits_m[pos_m], logits_f[pos_f])
            kl_values.append(kl)

    first_token_kl = kl_values[0] if kl_values else 0.0
    mean_kl = float(np.mean(kl_values)) if kl_values else 0.0

    return {
        "first_token_kl": first_token_kl,
        "mean_response_kl": mean_kl,
        "kl_per_position": [float(v) for v in kl_values],
        "response_len_tokens": resp_len,
    }


# =====================================================================
# MAIN
# =====================================================================

def main():
    t0 = datetime.now()
    print("=" * 70)
    print("  Experiment 1: Causal Mediation on Full Responses")
    print(f"  Model: {MODEL_ID}")
    print(f"  Best probe layer: {BEST_PROBE_LAYER}")
    print(f"  Dataset: {N_QUESTIONS} questions")
    print("=" * 70)

    # --- Step 1: Load model ---
    print(f"\n[1/5] Loading model...")
    model, tok, processor, get_layers, get_embed, num_layers = \
        load_model_and_tokenizer(MODEL_ID, LOADER)

    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Loaded | Layers: {num_layers} | VRAM: {vram:.1f} GB")

    layers = get_layers(model)
    if BEST_PROBE_LAYER > 0:
        target_module = layers[BEST_PROBE_LAYER - 1]
    else:
        target_module = get_embed(model)

    # --- Step 2: Extract hidden states at best layer for gender direction ---
    print(f"\n[2/5] Extracting hidden states at layer {BEST_PROBE_LAYER} "
          f"({len(all_gendered_prompts)} prompts)...")

    captured = {}
    def capture_hook(module, input, output):
        if isinstance(output, tuple):
            captured['hs'] = output[0].detach().cpu()
        else:
            captured['hs'] = output.detach().cpu()

    handle = target_module.register_forward_hook(capture_hook)

    hs_list = []
    for text in tqdm(all_gendered_prompts, desc="Extracting HS"):
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=128).to("cuda")
        with torch.no_grad():
            model(**inputs)
        hs = captured['hs'].squeeze(0).float()  # (seq_len, hidden_dim)
        hs_list.append(hs[-1].numpy())  # last token

    handle.remove()
    hs_array = np.array(hs_list)  # (400, hidden_dim)
    print(f"  Shape: {hs_array.shape}")

    # --- Step 3: Train probe and extract gender direction ---
    print(f"\n[3/5] Training probe for gender direction...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score

    scaler = StandardScaler()
    X = scaler.fit_transform(hs_array)
    probe = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
    scores = cross_val_score(probe, X, gender_labels, cv=5, scoring="accuracy")
    probe_acc = scores.mean()
    print(f"  Probe accuracy at layer {BEST_PROBE_LAYER}: {probe_acc:.1%}")

    probe.fit(X, gender_labels)
    gender_direction = probe.coef_[0] / scaler.scale_
    gender_direction = gender_direction / np.linalg.norm(gender_direction)
    gender_dir_tensor = torch.tensor(gender_direction, dtype=torch.bfloat16).to("cuda")

    # Random direction for control
    np.random.seed(42)
    random_dir = np.random.randn(len(gender_direction))
    random_dir = random_dir / np.linalg.norm(random_dir)
    random_dir_tensor = torch.tensor(random_dir, dtype=torch.bfloat16).to("cuda")

    # Free HS memory
    del hs_list, hs_array, X
    gc.collect()

    # --- Step 4: Causal mediation experiment ---
    N_SUBSET = 50  # Number of questions for generation
    STRENGTH = 10.0
    MAX_TOKENS = 256

    print(f"\n[4/5] Causal mediation ({N_SUBSET} questions × 3 conditions)...")
    print(f"  Ablation strength: {STRENGTH}")
    print(f"  Max generation tokens: {MAX_TOKENS}")

    results_per_question = []

    for q_idx in range(N_SUBSET):
        q_start = datetime.now()
        male_text = male_prompts[q_idx]
        female_text = female_prompts[q_idx]

        q_result = {
            "question_idx": q_idx,
            "question": questions[q_idx],
            "male_name": male_names[q_idx % N_MALE],
            "female_name": female_names[q_idx % N_FEMALE],
        }

        # --- Normal condition ---
        # Generate male response (used as reference for teacher-forcing)
        R_m_text, R_m_ids = generate_response(
            model, tok, processor, male_text, LOADER, MAX_TOKENS)
        R_f_text, R_f_ids = generate_response(
            model, tok, processor, female_text, LOADER, MAX_TOKENS)

        q_result["male_response_normal"] = R_m_text
        q_result["female_response_normal"] = R_f_text
        q_result["jaccard_normal"] = jaccard_similarity(R_m_text, R_f_text)

        # Teacher-forcing KL (using male response as reference)
        kl_normal = compute_response_kl(model, tok, male_text, female_text, R_m_ids)
        q_result["kl_normal"] = kl_normal

        # --- Gender-ablated condition ---
        hook_handle = target_module.register_forward_hook(
            make_hook(gender_dir_tensor, STRENGTH))

        R_m_abl_text, R_m_abl_ids = generate_response(
            model, tok, processor, male_text, LOADER, MAX_TOKENS)
        R_f_abl_text, R_f_abl_ids = generate_response(
            model, tok, processor, female_text, LOADER, MAX_TOKENS)

        q_result["male_response_ablated"] = R_m_abl_text
        q_result["female_response_ablated"] = R_f_abl_text
        q_result["jaccard_ablated"] = jaccard_similarity(R_m_abl_text, R_f_abl_text)

        # Teacher-forcing KL with ablation (using normal male response)
        kl_ablated = compute_response_kl(model, tok, male_text, female_text, R_m_ids)
        q_result["kl_ablated"] = kl_ablated

        hook_handle.remove()

        # --- Random-ablated condition ---
        hook_handle = target_module.register_forward_hook(
            make_hook(random_dir_tensor, STRENGTH))

        # Teacher-forcing KL with random ablation (using normal male response)
        kl_random = compute_response_kl(model, tok, male_text, female_text, R_m_ids)
        q_result["kl_random"] = kl_random

        hook_handle.remove()

        # --- Convergence ratios ---
        # First-token KL convergence
        ft_norm = kl_normal["first_token_kl"]
        ft_abl = kl_ablated["first_token_kl"]
        ft_rand = kl_random["first_token_kl"]
        q_result["convergence_first_token"] = (
            (ft_norm - ft_abl) / ft_norm if ft_norm > 1e-10 else 0.0)
        q_result["convergence_first_token_random"] = (
            (ft_norm - ft_rand) / ft_norm if ft_norm > 1e-10 else 0.0)

        # Response-level KL convergence
        resp_norm = kl_normal["mean_response_kl"]
        resp_abl = kl_ablated["mean_response_kl"]
        resp_rand = kl_random["mean_response_kl"]
        q_result["convergence_response"] = (
            (resp_norm - resp_abl) / resp_norm if resp_norm > 1e-10 else 0.0)
        q_result["convergence_response_random"] = (
            (resp_norm - resp_rand) / resp_norm if resp_norm > 1e-10 else 0.0)

        results_per_question.append(q_result)

        elapsed_q = (datetime.now() - q_start).total_seconds()
        print(f"  Q{q_idx:2d}: FT_KL norm={ft_norm:.4f} abl={ft_abl:.4f} "
              f"rand={ft_rand:.4f} | "
              f"Resp_KL norm={resp_norm:.4f} abl={resp_abl:.4f} | "
              f"{elapsed_q:.0f}s")

        # Save incrementally
        _save_results(results_per_question, probe_acc)

    # --- Step 5: Aggregate metrics ---
    print(f"\n[5/5] Computing aggregate metrics...")

    ft_kl_normal = [r["kl_normal"]["first_token_kl"] for r in results_per_question]
    ft_kl_ablated = [r["kl_ablated"]["first_token_kl"] for r in results_per_question]
    ft_kl_random = [r["kl_random"]["first_token_kl"] for r in results_per_question]

    resp_kl_normal = [r["kl_normal"]["mean_response_kl"] for r in results_per_question]
    resp_kl_ablated = [r["kl_ablated"]["mean_response_kl"] for r in results_per_question]
    resp_kl_random = [r["kl_random"]["mean_response_kl"] for r in results_per_question]

    convergence_ft = [r["convergence_first_token"] for r in results_per_question]
    convergence_resp = [r["convergence_response"] for r in results_per_question]

    jaccard_normal = [r["jaccard_normal"] for r in results_per_question]
    jaccard_ablated = [r["jaccard_ablated"] for r in results_per_question]

    summary = {
        "model": MODEL_SHORT,
        "model_id": MODEL_ID,
        "best_probe_layer": BEST_PROBE_LAYER,
        "probe_accuracy": float(probe_acc),
        "ablation_strength": STRENGTH,
        "n_questions": N_SUBSET,
        "max_tokens": MAX_TOKENS,
        # First-token KL
        "mean_first_token_kl_normal": float(np.mean(ft_kl_normal)),
        "mean_first_token_kl_ablated": float(np.mean(ft_kl_ablated)),
        "mean_first_token_kl_random": float(np.mean(ft_kl_random)),
        # Response-level KL
        "mean_response_kl_normal": float(np.mean(resp_kl_normal)),
        "mean_response_kl_ablated": float(np.mean(resp_kl_ablated)),
        "mean_response_kl_random": float(np.mean(resp_kl_random)),
        # Convergence (how much ablation reduces KL)
        "mean_convergence_first_token": float(np.mean(convergence_ft)),
        "mean_convergence_response": float(np.mean(convergence_resp)),
        # Jaccard (supplementary)
        "mean_jaccard_normal": float(np.mean(jaccard_normal)),
        "mean_jaccard_ablated": float(np.mean(jaccard_ablated)),
    }

    _save_results(results_per_question, probe_acc, summary)

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"  RESULTS — {MODEL_SHORT}")
    print(f"{'=' * 70}")
    print(f"  First-token KL:")
    print(f"    Normal:  {summary['mean_first_token_kl_normal']:.6f}")
    print(f"    Ablated: {summary['mean_first_token_kl_ablated']:.6f} "
          f"(↓{summary['mean_convergence_first_token']:.1%})")
    print(f"    Random:  {summary['mean_first_token_kl_random']:.6f}")
    print(f"  Response-level KL:")
    print(f"    Normal:  {summary['mean_response_kl_normal']:.6f}")
    print(f"    Ablated: {summary['mean_response_kl_ablated']:.6f} "
          f"(↓{summary['mean_convergence_response']:.1%})")
    print(f"    Random:  {summary['mean_response_kl_random']:.6f}")
    print(f"  Jaccard similarity:")
    print(f"    Normal:  {summary['mean_jaccard_normal']:.4f}")
    print(f"    Ablated: {summary['mean_jaccard_ablated']:.4f}")

    elapsed = (datetime.now() - t0).total_seconds() / 60
    print(f"\n  Time: {elapsed:.1f} min")
    print(f"  VRAM peak: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    print(f"{'=' * 70}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


def _save_results(per_question, probe_acc, summary=None):
    """Save results incrementally."""
    output = {
        "model": MODEL_SHORT,
        "model_id": MODEL_ID,
        "best_probe_layer": BEST_PROBE_LAYER,
        "probe_accuracy": float(probe_acc),
        "per_question": per_question,
    }
    if summary:
        output["summary"] = summary

    path = RESULTS_DIR / f"{MODEL_SHORT}_causal_mediation.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)


if __name__ == "__main__":
    main()
