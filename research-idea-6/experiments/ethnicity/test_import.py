#!/usr/bin/env python3
"""Quick test: can we load a model with device_map='auto'?"""
import sys
print(f"Python: {sys.executable}")

try:
    import accelerate
    print(f"accelerate: {accelerate.__version__}")
except ImportError:
    print("accelerate: NOT FOUND")

import transformers
print(f"transformers: {transformers.__version__}")

from transformers.utils import is_accelerate_available
print(f"is_accelerate_available: {is_accelerate_available()}")

import torch
print(f"torch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Try loading gemma1b (smallest model)
print("\nTrying to load gemma-3-1b-it...")
from transformers import AutoTokenizer, Gemma3ForCausalLM
tok = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
model = Gemma3ForCausalLM.from_pretrained(
    "google/gemma-3-1b-it", torch_dtype=torch.bfloat16, device_map="auto")
print(f"SUCCESS - model loaded, layers: {model.config.num_hidden_layers}")
del model
torch.cuda.empty_cache()
