# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "transformer-lens @ git+https://github.com/TransformerLensOrg/TransformerLens",
#     "circuitsvis",
#     "huggingface-hub",
#     "datasets>=3.0.0",
#     "pandas",
#     "numpy",
#     "tqdm",
#     "pyarrow>=17.0.0,<18.0.0",
#     "matplotlib",
#     "accelerate",
#     "transformers",
# ]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu124" }
#
# [[tool.uv.index]]
# name = "pytorch-cu124"
# url = "https://download.pytorch.org/whl/cu124"
# explicit = true
# ///

# %% [markdown]
# # Exploration: KL Divergence Between Small and Large Models
#
# This script explores the distributional differences between small and large
# models on the Pile dataset. We compute per-token KL divergence to identify
# where the models diverge most significantly.
#
# **Goal**: Understand baseline differences between small and large models before
# investigating user-modeling specific behaviors.

# %% Imports
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from datasets import load_dataset
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
import gc

# %% CUDA Check
def enforce_cuda():
    """Ensure CUDA is available and being used. Exit if not."""
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        print("This script requires a GPU to run.")
        print("\nDebug info:")
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA built: {torch.version.cuda}")
        return None

    # Print GPU info
    print("=" * 60)
    print("GPU CONFIGURATION")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Current memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print("=" * 60)

    return "cuda"

DEVICE = enforce_cuda()

# %% Configuration
# Model configuration - TransformerLens model names
# Using Gemma models (require HF login with gated repo access)
SMALL_MODEL_NAME = "gemma-2-2b"    # 2B parameter model
BIG_MODEL_NAME = "gemma-2-9b"      # 9B parameter model

# Alternative: Pythia models (ungated, no HF login required)
# SMALL_MODEL_NAME = "pythia-410m"
# BIG_MODEL_NAME = "pythia-1.4b"

# Dataset configuration
DATASET_NAME = "NeelNanda/pile-10k"   # 10k subset of the Pile
MAX_TOKENS = 500                       # Max tokens per document
NUM_DOCUMENTS = 100                    # Set to None for all documents

# Processing configuration
CONTEXT_WINDOW = 5                     # Tokens before/after for context

# Output paths
OUTPUT_DIR = Path("../results")
FIGURES_DIR = Path("../figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# %% Load Small Model
print(f"Loading {SMALL_MODEL_NAME} with TransformerLens...")

small_model = HookedTransformer.from_pretrained(
    SMALL_MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device=DEVICE,
)

print(f"  ✓ Loaded {SMALL_MODEL_NAME}")
print(f"  GPU Memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# %% Load Big Model
print(f"Loading {BIG_MODEL_NAME} with TransformerLens...")

big_model = HookedTransformer.from_pretrained(
    BIG_MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device=DEVICE,
)

print(f"  ✓ Loaded {BIG_MODEL_NAME}")
print(f"  GPU Memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# %% Load Dataset
print(f"Loading dataset: {DATASET_NAME}")

dataset = load_dataset(DATASET_NAME, split="train")

if NUM_DOCUMENTS is not None:
    dataset = dataset.select(range(min(NUM_DOCUMENTS, len(dataset))))

print(f"  ✓ Loaded {len(dataset)} documents")

# Preview a sample
print("\nSample document preview:")
print(dataset[0]["text"][:300] + "...")

# %% Helper Functions
def compute_kl_divergence(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence: KL(P || Q) where P is the "true" distribution.

    Here we compute KL(big_model || small_model) to see where small model
    diverges from big model's predictions.
    """
    # Convert to log probabilities (use float32 for numerical stability)
    log_p = F.log_softmax(logits_p.float(), dim=-1)
    log_q = F.log_softmax(logits_q.float(), dim=-1)

    # Compute probabilities for P
    p = torch.exp(log_p)

    # KL(P || Q) = sum(P * (log P - log Q))
    kl_div = torch.sum(p * (log_p - log_q), dim=-1)

    return kl_div


def get_context_text(tokens: list, position: int, model, window: int = 5) -> tuple:
    """Get the text context around a token position."""
    seq_len = len(tokens)

    start_before = max(0, position - window)
    end_after = min(seq_len, position + window + 1)

    before_tokens = tokens[start_before:position]
    after_tokens = tokens[position + 1:end_after]

    before_text = model.tokenizer.decode(before_tokens, skip_special_tokens=True)
    after_text = model.tokenizer.decode(after_tokens, skip_special_tokens=True)

    return before_text, after_text


@torch.no_grad()
def process_document(
    text: str,
    small_model: HookedTransformer,
    big_model: HookedTransformer,
    max_tokens: int = 500,
    context_window: int = 5
) -> list:
    """Process a single document and compute per-token KL divergence."""
    # Tokenize
    tokens = small_model.to_tokens(text, truncate=True)

    # Truncate if needed
    if tokens.shape[1] > max_tokens:
        tokens = tokens[:, :max_tokens]

    seq_len = tokens.shape[1]

    if seq_len < 2:
        return []

    # Get logits from both models
    small_logits = small_model(tokens)[0]  # [seq_len, vocab_size]
    big_logits = big_model(tokens)[0]      # [seq_len, vocab_size]

    # Compute KL divergence: KL(big || small)
    kl_divs = compute_kl_divergence(big_logits, small_logits)

    # Get log probabilities for actual next tokens
    token_list = tokens[0].cpu().tolist()
    small_log_probs = F.log_softmax(small_logits.float(), dim=-1)
    big_log_probs = F.log_softmax(big_logits.float(), dim=-1)

    results = []

    for pos in range(seq_len - 1):
        next_token_id = token_list[pos + 1]
        current_token_id = token_list[pos]

        small_log_prob = small_log_probs[pos, next_token_id].item()
        big_log_prob = big_log_probs[pos, next_token_id].item()

        before_text, after_text = get_context_text(token_list, pos, small_model, context_window)

        results.append({
            "position": pos,
            "token_id": current_token_id,
            "token_text": small_model.tokenizer.decode([current_token_id]),
            "next_token_id": next_token_id,
            "next_token_text": small_model.tokenizer.decode([next_token_id]),
            "kl_divergence": kl_divs[pos].item(),
            "small_log_prob": small_log_prob,
            "big_log_prob": big_log_prob,
            "log_prob_diff": big_log_prob - small_log_prob,
            "context_before": before_text,
            "context_after": after_text,
        })

    return results

# %% Process Dataset
print(f"Processing {len(dataset)} documents with max {MAX_TOKENS} tokens each...")

all_results = []

for batch_idx, example in enumerate(tqdm(dataset, desc="Processing documents")):
    text = example["text"]

    try:
        doc_results = process_document(
            text,
            small_model,
            big_model,
            max_tokens=MAX_TOKENS,
            context_window=CONTEXT_WINDOW
        )

        for result in doc_results:
            result["batch_idx"] = batch_idx

        all_results.extend(doc_results)

    except Exception as e:
        print(f"Error processing document {batch_idx}: {e}")
        continue

    # Clear cache periodically
    if batch_idx % 100 == 0:
        torch.cuda.empty_cache()

# Create DataFrame
df = pd.DataFrame(all_results)

# Reorder columns
column_order = [
    "batch_idx", "position", "token_id", "token_text",
    "next_token_id", "next_token_text", "kl_divergence",
    "small_log_prob", "big_log_prob", "log_prob_diff",
    "context_before", "context_after",
]
df = df[column_order]

print(f"\n✓ Processed {len(df):,} tokens total")

# %% Save Results
parquet_path = OUTPUT_DIR / "kl_divergence_results.parquet"
csv_path = OUTPUT_DIR / "kl_divergence_results.csv"

df.to_parquet(parquet_path, index=False)
print(f"✓ Saved {parquet_path}")

df.to_csv(csv_path, index=False)
print(f"✓ Saved {csv_path}")

# %% Statistics
print("=" * 60)
print("KL DIVERGENCE STATISTICS")
print("=" * 60)
print(f"\nTotal tokens analyzed: {len(df):,}")
print(f"Documents processed: {df['batch_idx'].nunique()}")
print(f"\nKL Divergence (KL(big || small)):")
print(df["kl_divergence"].describe())

# %% Top Divergent Tokens
print("\n" + "=" * 60)
print("TOP 20 MOST DIVERGENT TOKENS")
print("=" * 60)

top_divergent = df.nlargest(20, "kl_divergence")[
    ["batch_idx", "position", "token_text", "next_token_text",
     "kl_divergence", "context_before", "context_after"]
]

for idx, row in top_divergent.iterrows():
    print(f"\nKL: {row['kl_divergence']:.4f}")
    print(f"  Context: ...{row['context_before']}[{row['token_text']}]{row['context_after']}...")
    print(f"  Next token: '{row['next_token_text']}'")
    print(f"  Doc: {row['batch_idx']}, Pos: {row['position']}")

# %% Visualizations
import matplotlib.pyplot as plt

# Plot 1: Distribution of KL divergence
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df["kl_divergence"], bins=100, edgecolor="black", alpha=0.7)
axes[0].set_xlabel("KL Divergence")
axes[0].set_ylabel("Count")
axes[0].set_title("Distribution of KL Divergence (KL(big || small))")
axes[0].set_yscale("log")

axes[1].hist(df["log_prob_diff"], bins=100, edgecolor="black", alpha=0.7, color="orange")
axes[1].set_xlabel("Log Prob Difference (big - small)")
axes[1].set_ylabel("Count")
axes[1].set_title("Difference in Log Probability of Next Token")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "kl_divergence_distribution.png", dpi=150)
plt.show()

# %% KL by Position
position_stats = df.groupby("position")["kl_divergence"].agg(["mean", "std", "median"]).reset_index()

plt.figure(figsize=(12, 5))
plt.plot(position_stats["position"], position_stats["mean"], label="Mean KL")
plt.fill_between(
    position_stats["position"],
    position_stats["mean"] - position_stats["std"],
    position_stats["mean"] + position_stats["std"],
    alpha=0.3
)
plt.xlabel("Position in Sequence")
plt.ylabel("KL Divergence")
plt.title("KL Divergence by Position (Mean ± Std)")
plt.legend()
plt.savefig(FIGURES_DIR / "kl_by_position.png", dpi=150)
plt.show()

# %% Correlation Plot
plt.figure(figsize=(8, 8))
plt.scatter(df["kl_divergence"], df["log_prob_diff"], alpha=0.1, s=1)
plt.xlabel("KL Divergence")
plt.ylabel("Log Prob Diff (big - small)")
plt.title("KL Divergence vs Log Prob Difference")
plt.savefig(FIGURES_DIR / "kl_vs_logprob.png", dpi=150)
plt.show()

correlation = df["kl_divergence"].corr(df["log_prob_diff"])
print(f"Correlation (KL vs log_prob_diff): {correlation:.4f}")

# %% Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"""
Models compared:
  - Small: {SMALL_MODEL_NAME}
  - Big: {BIG_MODEL_NAME}

Dataset: {DATASET_NAME}
  - Documents: {df['batch_idx'].nunique()}
  - Total tokens: {len(df):,}
  - Max tokens per doc: {MAX_TOKENS}

KL Divergence Statistics:
  - Mean: {df['kl_divergence'].mean():.4f}
  - Median: {df['kl_divergence'].median():.4f}
  - Max: {df['kl_divergence'].max():.4f}
  - 95th percentile: {df['kl_divergence'].quantile(0.95):.4f}
  - 99th percentile: {df['kl_divergence'].quantile(0.99):.4f}

Output saved to: {parquet_path}
""")

# %% Cleanup
del small_model
del big_model
gc.collect()
torch.cuda.empty_cache()
print(f"GPU memory after cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print("\n✓ Done!")
