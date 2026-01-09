"""
Model Diffing Pipeline for Sycophancy Research

This module provides the core infrastructure for running "stereo" analysis:
comparing Base model (gemma-2-2b) vs Chat model (gemma-2-2b-it) activations
and feeding them through a Cross-Coder to identify Chat-specific features.

Based on the methodology from:
- "Model Diffing" approach comparing Base vs RLHF-trained models
- Anthropic's work on Cross-Coders
- Open Source replications (e.g., Butanium/gemma-2-2b-crosscoder)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import json
from pathlib import Path
import warnings

# Conditional imports for flexibility
try:
    from transformer_lens import HookedTransformer
    TRANSFORMER_LENS_AVAILABLE = True
except ImportError:
    TRANSFORMER_LENS_AVAILABLE = False
    warnings.warn("transformer_lens not installed. Install with: pip install transformer_lens")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("transformers not installed. Install with: pip install transformers")


@dataclass
class ActivationCache:
    """Container for cached activations from a single forward pass."""
    residual_stream: Dict[int, torch.Tensor] = field(default_factory=dict)
    attention_patterns: Dict[int, torch.Tensor] = field(default_factory=dict)
    mlp_activations: Dict[int, torch.Tensor] = field(default_factory=dict)
    prompt: str = ""
    tokens: Optional[torch.Tensor] = None

    def get_layer(self, layer: int, component: str = "residual") -> torch.Tensor:
        """Get activations for a specific layer and component."""
        if component == "residual":
            return self.residual_stream.get(layer)
        elif component == "attention":
            return self.attention_patterns.get(layer)
        elif component == "mlp":
            return self.mlp_activations.get(layer)
        else:
            raise ValueError(f"Unknown component: {component}")


@dataclass
class DiffResult:
    """Result of comparing Base vs Chat model activations."""
    base_cache: ActivationCache
    chat_cache: ActivationCache
    prompt: str
    concatenated_activations: Dict[int, torch.Tensor] = field(default_factory=dict)
    cross_coder_output: Optional[torch.Tensor] = None
    active_latents: Optional[List[int]] = None
    latent_activations: Optional[torch.Tensor] = None


class StereoModelRunner:
    """
    Runs both Base and Chat models side-by-side on the same prompt.

    This is the core of the "stereo" setup needed for Model Diffing.
    """

    # Model identifiers
    BASE_MODEL = "google/gemma-2-2b"
    CHAT_MODEL = "google/gemma-2-2b-it"

    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_transformer_lens: bool = True,
        target_layers: Optional[List[int]] = None,
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize the stereo model runner.

        Args:
            device: Device to run models on ('cuda' or 'cpu')
            use_transformer_lens: Whether to use TransformerLens (recommended for hooks)
            target_layers: Which layers to cache activations for (default: [13] - middle layer)
            dtype: Data type for model weights
        """
        self.device = device
        self.use_transformer_lens = use_transformer_lens
        self.target_layers = target_layers or [13]  # Default to middle layer
        self.dtype = dtype

        self.base_model = None
        self.chat_model = None
        self.tokenizer = None

        self._is_loaded = False

    def load_models(self, hf_token: Optional[str] = None):
        """
        Load both Base and Chat models.

        Args:
            hf_token: HuggingFace token for accessing gated models (Gemma requires acceptance)
        """
        print(f"Loading models on {self.device}...")

        if self.use_transformer_lens and TRANSFORMER_LENS_AVAILABLE:
            self._load_with_transformer_lens(hf_token)
        elif TRANSFORMERS_AVAILABLE:
            self._load_with_transformers(hf_token)
        else:
            raise RuntimeError("No model loading library available. Install transformer_lens or transformers.")

        self._is_loaded = True
        print("Models loaded successfully!")

    def _load_with_transformer_lens(self, hf_token: Optional[str] = None):
        """Load models using TransformerLens for easy hooking."""
        print("Loading Base model (gemma-2-2b) with TransformerLens...")
        self.base_model = HookedTransformer.from_pretrained(
            self.BASE_MODEL,
            device=self.device,
            dtype=self.dtype,
            token=hf_token,
        )

        print("Loading Chat model (gemma-2-2b-it) with TransformerLens...")
        self.chat_model = HookedTransformer.from_pretrained(
            self.CHAT_MODEL,
            device=self.device,
            dtype=self.dtype,
            token=hf_token,
        )

        self.tokenizer = self.base_model.tokenizer

    def _load_with_transformers(self, hf_token: Optional[str] = None):
        """Load models using standard HuggingFace transformers."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("Loading Base model (gemma-2-2b) with transformers...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.BASE_MODEL,
            device_map=self.device,
            torch_dtype=self.dtype,
            token=hf_token,
        )

        print("Loading Chat model (gemma-2-2b-it) with transformers...")
        self.chat_model = AutoModelForCausalLM.from_pretrained(
            self.CHAT_MODEL,
            device_map=self.device,
            torch_dtype=self.dtype,
            token=hf_token,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.BASE_MODEL, token=hf_token)

    def run_stereo(self, prompt: str) -> Tuple[ActivationCache, ActivationCache]:
        """
        Run both models on the same prompt and cache activations.

        Args:
            prompt: Input prompt to run through both models

        Returns:
            Tuple of (base_cache, chat_cache) containing activations
        """
        if not self._is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        if self.use_transformer_lens:
            return self._run_stereo_transformer_lens(prompt)
        else:
            return self._run_stereo_transformers(prompt)

    def _run_stereo_transformer_lens(self, prompt: str) -> Tuple[ActivationCache, ActivationCache]:
        """Run stereo analysis using TransformerLens hooks."""
        # Tokenize
        tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # Hook names for residual stream
        hook_names = [f"blocks.{layer}.hook_resid_post" for layer in self.target_layers]

        # Run Base model with caching
        _, base_cache_dict = self.base_model.run_with_cache(
            tokens,
            names_filter=lambda name: name in hook_names,
            return_type="logits",
        )

        # Run Chat model with caching
        _, chat_cache_dict = self.chat_model.run_with_cache(
            tokens,
            names_filter=lambda name: name in hook_names,
            return_type="logits",
        )

        # Convert to ActivationCache format
        base_cache = ActivationCache(prompt=prompt, tokens=tokens)
        chat_cache = ActivationCache(prompt=prompt, tokens=tokens)

        for layer in self.target_layers:
            hook_name = f"blocks.{layer}.hook_resid_post"
            base_cache.residual_stream[layer] = base_cache_dict[hook_name]
            chat_cache.residual_stream[layer] = chat_cache_dict[hook_name]

        return base_cache, chat_cache

    def _run_stereo_transformers(self, prompt: str) -> Tuple[ActivationCache, ActivationCache]:
        """Run stereo analysis using standard transformers with hooks."""
        tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        base_cache = ActivationCache(prompt=prompt, tokens=tokens)
        chat_cache = ActivationCache(prompt=prompt, tokens=tokens)

        # For standard transformers, we need to use hooks
        def create_hook(cache: ActivationCache, layer: int):
            def hook(module, input, output):
                # For Gemma, output is typically (hidden_states, ...)
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                cache.residual_stream[layer] = hidden_states.detach()
            return hook

        # Register hooks for base model
        base_handles = []
        for layer in self.target_layers:
            handle = self.base_model.model.layers[layer].register_forward_hook(
                create_hook(base_cache, layer)
            )
            base_handles.append(handle)

        # Run base model
        with torch.no_grad():
            self.base_model(tokens)

        # Remove hooks
        for handle in base_handles:
            handle.remove()

        # Register hooks for chat model
        chat_handles = []
        for layer in self.target_layers:
            handle = self.chat_model.model.layers[layer].register_forward_hook(
                create_hook(chat_cache, layer)
            )
            chat_handles.append(handle)

        # Run chat model
        with torch.no_grad():
            self.chat_model(tokens)

        # Remove hooks
        for handle in chat_handles:
            handle.remove()

        return base_cache, chat_cache


class ModelDiffingPipeline:
    """
    Complete pipeline for Model Diffing analysis.

    This orchestrates:
    1. Running both models (StereoModelRunner)
    2. Concatenating activations
    3. Running through Cross-Coder
    4. Analyzing resulting latents
    """

    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        target_layer: int = 13,
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize the Model Diffing pipeline.

        Args:
            device: Device to run on
            target_layer: Which layer to analyze (13 is middle layer for Gemma 2 2B)
            dtype: Data type for computations
        """
        self.device = device
        self.target_layer = target_layer
        self.dtype = dtype

        self.stereo_runner = StereoModelRunner(
            device=device,
            target_layers=[target_layer],
            dtype=dtype,
        )

        self.cross_coder = None
        self._is_initialized = False

    def initialize(
        self,
        hf_token: Optional[str] = None,
        cross_coder_path: Optional[str] = None,
    ):
        """
        Initialize the pipeline by loading models and Cross-Coder.

        Args:
            hf_token: HuggingFace token for gated models
            cross_coder_path: Path or HuggingFace ID for Cross-Coder weights
        """
        # Load the stereo models
        self.stereo_runner.load_models(hf_token)

        # Load Cross-Coder if path provided
        if cross_coder_path:
            self._load_cross_coder(cross_coder_path)

        self._is_initialized = True

    def _load_cross_coder(self, path: str):
        """Load Cross-Coder from path or HuggingFace."""
        from .cross_coder import CrossCoderWrapper
        self.cross_coder = CrossCoderWrapper.from_pretrained(path, device=self.device)

    def analyze_prompt(self, prompt: str) -> DiffResult:
        """
        Run complete Model Diffing analysis on a single prompt.

        Args:
            prompt: Input prompt to analyze

        Returns:
            DiffResult containing all analysis outputs
        """
        if not self._is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        # Step 1: Run stereo analysis
        base_cache, chat_cache = self.stereo_runner.run_stereo(prompt)

        # Step 2: Get activations at target layer
        base_acts = base_cache.get_layer(self.target_layer)
        chat_acts = chat_cache.get_layer(self.target_layer)

        # Step 3: Concatenate activations (Cross-Coder format)
        # Shape: [batch, seq_len, 2 * d_model]
        concatenated = torch.cat([base_acts, chat_acts], dim=-1)

        result = DiffResult(
            base_cache=base_cache,
            chat_cache=chat_cache,
            prompt=prompt,
            concatenated_activations={self.target_layer: concatenated},
        )

        # Step 4: Run through Cross-Coder if available
        if self.cross_coder is not None:
            latent_acts = self.cross_coder.encode(concatenated)
            result.cross_coder_output = latent_acts
            result.latent_activations = latent_acts

            # Find active latents (above threshold)
            threshold = 0.1
            active_mask = latent_acts.abs() > threshold
            result.active_latents = active_mask.nonzero(as_tuple=True)[-1].unique().tolist()

        return result

    def analyze_prompt_pair(
        self,
        prompt_a: str,
        prompt_b: str,
        label_a: str = "A",
        label_b: str = "B",
    ) -> Dict:
        """
        Compare latent activations between two related prompts.

        This is the core analysis for finding User Modeling features:
        - Compare Novice vs Expert prompts
        - Compare Biased vs Neutral prompts
        - Look for latents that differentiate the two

        Args:
            prompt_a: First prompt (e.g., "novice" version)
            prompt_b: Second prompt (e.g., "expert" version)
            label_a: Label for first prompt
            label_b: Label for second prompt

        Returns:
            Dictionary with comparison results
        """
        result_a = self.analyze_prompt(prompt_a)
        result_b = self.analyze_prompt(prompt_b)

        comparison = {
            "prompt_a": prompt_a,
            "prompt_b": prompt_b,
            "label_a": label_a,
            "label_b": label_b,
            "result_a": result_a,
            "result_b": result_b,
        }

        if result_a.latent_activations is not None and result_b.latent_activations is not None:
            # Compute differential activation
            # We want latents that are MORE active in prompt_a than prompt_b
            diff = result_a.latent_activations - result_b.latent_activations

            # Find latents with largest positive difference (more active in A)
            # Average across sequence positions
            mean_diff = diff.mean(dim=(0, 1))  # [num_latents]

            # Top latents more active in prompt_a
            top_k = 20
            top_a_indices = mean_diff.topk(top_k).indices.tolist()
            top_a_values = mean_diff.topk(top_k).values.tolist()

            # Top latents more active in prompt_b
            top_b_indices = (-mean_diff).topk(top_k).indices.tolist()
            top_b_values = (-mean_diff).topk(top_k).values.tolist()

            comparison["differential_latents"] = {
                f"top_in_{label_a}": list(zip(top_a_indices, top_a_values)),
                f"top_in_{label_b}": list(zip(top_b_indices, top_b_values)),
                "mean_diff": mean_diff.cpu().numpy(),
            }

        return comparison

    def find_chat_specific_latents(
        self,
        prompt: str,
        threshold: float = 0.5,
    ) -> Dict:
        """
        Find latents that are specific to the Chat model (not in Base model).

        These are the "RLHF-induced" features, which include:
        - User modeling (novice/expert detection)
        - Safety behaviors
        - Helpfulness patterns
        - Sycophancy tendencies

        Args:
            prompt: Input prompt to analyze
            threshold: Activation threshold for considering a latent "active"

        Returns:
            Dictionary with chat-specific, base-specific, and shared latents
        """
        result = self.analyze_prompt(prompt)

        if self.cross_coder is None:
            raise RuntimeError("Cross-Coder not loaded. Provide cross_coder_path in initialize().")

        # The Cross-Coder is designed to separate features into:
        # - Shared (active for both models)
        # - Base-specific (active only in base model)
        # - Chat-specific (active only in chat model)

        # This depends on the specific Cross-Coder architecture
        # Typically, the latent space is partitioned or we can infer from activations

        chat_specific = self.cross_coder.get_chat_specific_latents(
            result.concatenated_activations[self.target_layer],
            threshold=threshold,
        )

        return {
            "prompt": prompt,
            "chat_specific_latents": chat_specific,
            "all_active_latents": result.active_latents,
            "raw_activations": result.latent_activations,
        }


def load_user_persona_dataset(path: str = "data/user_persona_prompts.json") -> Dict:
    """
    Load the user persona dataset for analysis.

    Args:
        path: Path to the JSON dataset file

    Returns:
        Loaded dataset as dictionary
    """
    with open(path, "r") as f:
        return json.load(f)


def run_day1_analysis(
    hf_token: str,
    output_dir: str = "results/day1",
    cross_coder_path: str = "Butanium/gemma-2-2b-crosscoder-l13",
):
    """
    Run the Day 1 analysis: Find Chat-specific User Modeling features.

    This implements the "Hunt" described in the research plan:
    1. Run prompts through both models
    2. Identify Chat-specific latents
    3. Look for latents that differentiate Novice vs Expert prompts

    Args:
        hf_token: HuggingFace token
        output_dir: Directory to save results
        cross_coder_path: Path to Cross-Coder weights
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Initialize pipeline
    pipeline = ModelDiffingPipeline(target_layer=13)
    pipeline.initialize(hf_token=hf_token, cross_coder_path=cross_coder_path)

    # Load dataset
    dataset = load_user_persona_dataset()

    results = []

    # Analyze expertise pairs
    for pair in dataset["prompt_pairs"]["expertise_level"]:
        print(f"\nAnalyzing pair: {pair['id']} ({pair['topic']})")

        comparison = pipeline.analyze_prompt_pair(
            pair["novice"]["prompt"],
            pair["expert"]["prompt"],
            label_a="novice",
            label_b="expert",
        )

        results.append({
            "pair_id": pair["id"],
            "category": pair["category"],
            "topic": pair["topic"],
            "differential_latents": comparison.get("differential_latents"),
        })

        # Look for the "User Model" feature
        if "differential_latents" in comparison:
            top_novice = comparison["differential_latents"]["top_in_novice"][:5]
            print(f"  Top latents for NOVICE prompt: {top_novice}")

    # Save results
    output_path = os.path.join(output_dir, "expertise_analysis.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")
    return results


if __name__ == "__main__":
    # Quick test
    print("Model Diffing Pipeline loaded successfully!")
    print(f"TransformerLens available: {TRANSFORMER_LENS_AVAILABLE}")
    print(f"Transformers available: {TRANSFORMERS_AVAILABLE}")
